"""Quality-evaluation orchestration.

Provides the product-facing ``oai quality <id>`` command and wraps
``inference/run-quality.sh`` using catalog data.  The catalog model ID is used
for Kubernetes/S3 labels while ``serving.served_name`` is sent to the vLLM
OpenAI endpoint; these differ for models such as Qwen3.5-4B.
"""

from __future__ import annotations

from . import catalog, paths, shell, ui


# ---------------------------------------------------------------------------
# Dataset sheets in the shared AutoQA workbook (multi-sheet .xlsx). Each entry
# maps a friendly key -> (worksheet name, quality config, result-tag segment).
#
# The two transcript sheets are compared SEPARATELY (lead's instruction), so each
# runs as its OWN Job with its own tag; that keeps their S3 result prefixes — and
# therefore the report rows — distinct. The QIDs sheet is a lookup table, not a
# scoring sheet, so it is never run on its own.
# ---------------------------------------------------------------------------
SHEETS: dict[str, dict[str, str]] = {
    # Test-ORG: original single-call transcripts.
    "org": {
        "sheet": "synthetic_autoqa_transcripts",
        "config": "configs/quality/autoqa_v1.yaml",
        "tag": "org",
    },
    # Test-XL: longer 2k-7k token prompts (several calls bundled per row).
    "xl": {
        "sheet": "synthetic_autoqa_transcripts-v2",
        "config": "configs/quality/autoqa_xl.yaml",
        "tag": "xl",
    },
}
# Default when the caller does not pick: run BOTH transcript sheets.
DEFAULT_SHEET_KEYS = ["org", "xl"]


def _resolve_sheet_keys(sheets: list[str] | None) -> list[str]:
    """Map user-facing sheet selectors to known keys.

    Accepts the friendly key (``org`` / ``xl``) OR the exact worksheet name, so
    ``--sheet synthetic_autoqa_transcripts-v2`` also works. ``None``/empty means
    run every transcript sheet.
    """
    if not sheets:
        return list(DEFAULT_SHEET_KEYS)
    name_to_key = {v["sheet"]: k for k, v in SHEETS.items()}
    resolved: list[str] = []
    for s in sheets:
        key = s.strip()
        if key in SHEETS:
            resolved.append(key)
        elif key in name_to_key:
            resolved.append(name_to_key[key])
        else:
            ui.fail(
                f"Unknown sheet '{s}'.",
                "Valid values: " + ", ".join(
                    f"{k} ({v['sheet']})" for k, v in SHEETS.items()
                ),
            )
    # De-duplicate while preserving order (e.g. --sheet org --sheet org).
    seen: set[str] = set()
    return [k for k in resolved if not (k in seen or seen.add(k))]


def run(
    model_id: str,
    *,
    tag: str | None = None,
    hw: str | None = None,
    quant: str | None = None,
    config: str | None = None,
    dataset_key: str | None = None,
    dataset_file: str | None = None,
    force_stage: bool = False,
    sheets: list[str] | None = None,
    service: str | None = None,
    served_name: str | None = None,
    dump_samples: bool = False,
    max_dump_samples: int = 200,
    dump_inputs: bool = False,
    run_timestamp: str | None = None,
    wait_for_marker: str | None = None,
    detach: bool = False,
) -> int:
    """Submit AutoQA quality Job(s) against an already-deployed model.

    By default this runs BOTH transcript sheets (Test-ORG and Test-XL) as
    separate Jobs so their metrics never mix. Pass ``sheets=["org"]`` (or the
    worksheet name) to run just one.

    ``wait_for_marker`` is an optional S3 key used by deploy orchestration to
    make quality the final stage after performance benchmarks. Standalone
    quality runs omit it and start immediately.
    """
    script = paths.ROOT / "inference" / "run-quality.sh"
    if not script.exists():
        ui.fail(
            "inference/run-quality.sh is missing.",
            "This is part of the base framework - check your checkout.",
        )

    if max_dump_samples < 1:
        ui.fail("--max-dump-samples must be at least 1.")

    if dump_inputs and not dump_samples:
        ui.fail(
            "--dump-inputs needs --dump-samples.",
            "Inputs are written into the samples file, so both are required.",
        )

    sheet_keys = _resolve_sheet_keys(sheets)

    # An explicit --config or --dataset-key only makes sense for a single sheet;
    # per-sheet defaults would otherwise be silently overridden for both.
    if len(sheet_keys) > 1 and (config or dataset_key):
        ui.fail(
            "--config / --dataset-key can only be used with a single --sheet.",
            "Run one sheet at a time when overriding the config or dataset key.",
        )

    rc_final = 0
    for key in sheet_keys:
        sheet_def = SHEETS[key]
        rc = _run_one_sheet(
            model_id,
            script=script,
            sheet_def=sheet_def,
            single_sheet=(len(sheet_keys) == 1),
            tag=tag,
            hw=hw,
            quant=quant,
            config=config,
            dataset_key=dataset_key,
            dataset_file=dataset_file,
            force_stage=force_stage,
            service=service,
            served_name=served_name,
            dump_samples=dump_samples,
            max_dump_samples=max_dump_samples,
            dump_inputs=dump_inputs,
            run_timestamp=run_timestamp,
            wait_for_marker=wait_for_marker,
            # When several sheets run in one invocation they must all be submitted
            # up front and continue in-cluster, so force detach for a multi-sheet
            # run regardless of the caller's preference.
            detach=detach or len(sheet_keys) > 1,
        )
        rc_final = rc_final or rc
    return rc_final


def _run_one_sheet(
    model_id: str,
    *,
    script,  # type: ignore[no-untyped-def]
    sheet_def: dict[str, str],
    single_sheet: bool,
    tag: str | None,
    hw: str | None,
    quant: str | None,
    config: str | None,
    dataset_key: str | None,
    dataset_file: str | None,
    force_stage: bool,
    service: str | None,
    served_name: str | None,
    dump_samples: bool,
    max_dump_samples: int,
    dump_inputs: bool,
    run_timestamp: str | None,
    wait_for_marker: str | None,
    detach: bool,
) -> int:
    spec = catalog.load(model_id)

    if tag and spec.is_neuron:
        ui.warn("--tag is not supported for Neuron models; using the untagged service.")
        tag = None

    selected_hw = hw or spec.preferred_instance or spec.hardware
    selected_quant = quant or spec.serving.quantization
    selected_served_name = served_name or spec.serving.served_name or spec.id
    selected_service = service or spec.service_name
    if tag and not service:
        selected_service = f"{selected_service}-{tag}"

    sheet_name = sheet_def["sheet"]
    # Result tag = the sheet segment, combined with any user tag, so Test-ORG and
    # Test-XL land under separate S3 prefixes even for the same deployment.
    sheet_seg = sheet_def["tag"]
    combined_tag = f"{tag}-{sheet_seg}" if tag else sheet_seg
    # The config defaults to the sheet's own config unless the caller overrode it
    # (only allowed for a single-sheet run, enforced by the caller).
    selected_config = config or sheet_def["config"]

    args = [
        "--model", spec.id,
        "--served-name", selected_served_name,
        "--hw", selected_hw,
        "--quant", selected_quant,
        "--svc", selected_service,
        "--config", selected_config,
        "--sheet", sheet_name,
        "--tag", combined_tag,
    ]
    if dataset_key:
        args += ["--dataset-key", dataset_key]
    if dataset_file:
        args += ["--dataset-file", dataset_file]
    if force_stage:
        args += ["--force-stage"]
    if run_timestamp:
        args += ["--run-timestamp", run_timestamp]
    if wait_for_marker:
        args += ["--wait-for-marker", wait_for_marker]
    if dump_samples:
        args += ["--dump-samples", "--max-dump-samples", str(max_dump_samples)]
        if dump_inputs:
            args += ["--dump-inputs"]
    if detach:
        args += ["--detach"]

    label = "Test-ORG" if sheet_seg == "org" else "Test-XL" if sheet_seg == "xl" else sheet_name
    ui.banner(f"Quality: {spec.id}  [{label}]{f'  (tag={tag})' if tag else ''}")
    ui.kv("Service", selected_service)
    ui.kv("Served model", selected_served_name)
    ui.kv("Hardware", selected_hw)
    ui.kv("Quantization", selected_quant)
    ui.kv("Sheet", sheet_name)
    ui.kv("Config", selected_config)
    ui.kv("Result tag", combined_tag)
    if dataset_file:
        ui.kv("Dataset file", f"{dataset_file} (auto-staged to S3 if missing)")
    ui.kv("Samples", f"dump first {max_dump_samples}" if dump_samples else "aggregate metrics only")
    if dump_inputs:
        ui.kv("Inputs", "INCLUDED in samples (call transcripts are persisted)")
    if wait_for_marker:
        ui.kv("Starts after", f"s3 marker {wait_for_marker}")

    rc = shell.run_bash(script, args)
    if rc != 0:
        ui.warn(f"Quality job for {label} reported non-zero (exit {rc}).")
        ui.hint("Check the Job logs and the S3 quality results path printed above.")
        return rc

    if detach:
        ui.info(f"Quality Job for {label} submitted in-cluster. It continues independently of this terminal.")
    else:
        ui.info(f"Quality evaluation for {label} finished. Results are in the results bucket (path shown above).")
    return 0
