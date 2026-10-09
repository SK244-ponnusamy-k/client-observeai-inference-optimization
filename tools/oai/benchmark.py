"""
Benchmark orchestration.

Wraps the existing in-cluster benchmark harness (inference/run-benchmark.sh),
driven by the model's catalog entry. Benchmarking is fully customizable:

  - Which profiles run comes from catalog `benchmark.profiles` (or --profile).
  - A custom dataset comes from catalog `benchmark.dataset` (or --dataset).
  - Cost math uses catalog `benchmark.instance_hourly_usd`, baked into the
    generated run manifest.

`oai deploy --benchmark` (or catalog benchmark.auto) calls run() right after a
successful deploy, so a user can onboard, deploy, and benchmark in one step.
"""

from __future__ import annotations

import os
import re

from . import catalog, paths, shell, ui
from .catalog import ModelSpec


def _read_config_env(key: str) -> str:
    """Read a variable value from config/config.env without requiring bash to
    source it. Handles `export KEY="value"` and `export KEY="${KEY:-default}"`
    forms. Returns "" when the key is not found or the file doesn't exist.
    """
    env_file = paths.CONFIG_ENV
    if not env_file.exists():
        return ""
    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        # Match: export KEY="value" or export KEY=value (with or without quotes)
        m = re.match(r'^export\s+' + re.escape(key) + r'\s*=\s*["\']?([^"\'$\n{][^"\'$\n]*?)["\']?\s*$', line)
        if m:
            return m.group(1).strip().strip("'\"")
        # Match the default-value form: export KEY="${KEY:-actual_default}"
        m2 = re.match(r'^export\s+' + re.escape(key) + r'\s*=\s*"\$\{' + re.escape(key) + r':-([^}]*)\}"', line)
        if m2:
            return m2.group(1).strip().strip("'\"")
    return ""

# Map a workload-profile path/name to the short name run-benchmark.sh expects.
_PROFILE_RE = re.compile(r"(realtime|batch)")


def _profile_short(profile_ref: str) -> str | None:
    m = _PROFILE_RE.search(profile_ref)
    return m.group(1) if m else None


def _selected_profiles(spec: ModelSpec, profile: str | None, skip_batch: bool) -> list[str]:
    if profile:
        short = _profile_short(profile)
        if not short:
            ui.fail(
                f"Unrecognised profile '{profile}'.",
                "Use 'realtime' or 'batch' (or a path like configs/workload_profiles/realtime_v1.yaml).",
            )
        return [short]
    shorts: list[str] = []
    for p in spec.benchmark.profiles:
        s = _profile_short(p)
        if s and s not in shorts:
            shorts.append(s)
    if skip_batch:
        shorts = [s for s in shorts if s != "batch"]
    return shorts or ["realtime"]


def _profile_arg(profiles: list[str]) -> str:
    """Collapse selected profiles into run-benchmark.sh's --profile value.

    ``both`` submits two separate Jobs. The batch Job waits for realtime's S3
    completion marker, so the profiles execute sequentially against one server.
    """
    has_rt = "realtime" in profiles
    has_batch = "batch" in profiles
    if has_rt and has_batch:
        return "both"
    if has_batch:
        return "batch"
    return "realtime"


def selected_profiles(spec: ModelSpec, profile: str | None, skip_batch: bool) -> list[str]:
    """Public normalized profile order used by deploy pipeline orchestration."""
    return _selected_profiles(spec, profile, skip_batch)


def completion_profile(spec: ModelSpec, profile: str | None, skip_batch: bool) -> str:
    """Profile whose completion marker represents the end of performance work."""
    profiles = selected_profiles(spec, profile, skip_batch)
    return "batch" if "batch" in profiles else "realtime"


def _manifest_hw(spec: ModelSpec, hw_override: str | None = None) -> str:
    """The <hw> token used in the manifest filename.

    When hw_override is given (the instance the model was actually deployed on,
    e.g. 'g7e.2xlarge'), use its family ('g7e') so the benchmark reads the cell
    that MATCHES the hardware — otherwise results get mislabeled with the
    catalog's default (preferred) instance. No override → catalog default.
    """
    if hw_override:
        return hw_override.split(".")[0]
    if spec.preferred_instance:
        return spec.preferred_instance.split(".")[0]
    return spec.hardware


def run(
    model_id: str,
    *,
    profile: str | None = None,
    dataset: str | None = None,
    skip_batch: bool = False,
    tag: str | None = None,
    hw: str | None = None,
    manifest: str | None = None,
    run_timestamp: str | None = None,
    auto_stop: bool = False,
    detach: bool = False,
    qid_cost: bool = False,
    qid_config: str = "",
    qid_sheet: str = "",
) -> int:
    spec = catalog.load(model_id)

    # A deploy-pipeline call always supplies run_timestamp (and never auto_stop,
    # since deploy submits its own autostop). Anything else is a STANDALONE run.
    from_deploy_pipeline = run_timestamp is not None

    # Standalone --auto-stop: we must know the run_timestamp up front so the
    # auto-stop Job can match the benchmark Job it should wait on. Generate one
    # here and run detached (auto-stop must survive the terminal closing).
    if auto_stop and run_timestamp is None:
        from datetime import datetime, timezone
        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        detach = True

    # Refuse a duplicate STANDALONE run for this model+tag (would hit the same
    # endpoint and contaminate results). Skipped for the deploy pipeline — deploy
    # already ran this guard, and the jobs it just submitted share this
    # run_timestamp, so re-checking would false-positive on our own jobs.
    # include_deployment=False: benchmarking runs AGAINST a live deployment.
    from . import preflight
    if not from_deploy_pipeline:
        preflight.fail_if_active(spec.id, tag, include_deployment=False)

    # Resolve the benchmark manifest, in priority order:
    #   1. Explicit --manifest (e.g. a Neuron/Trainium manifest you select by hand).
    #   2. Single dynamic GPU manifest <id>-gpu.yaml (the standard path — one file
    #      for ALL G instances; instance_type + cost resolve dynamically from --hw).
    #   3. Legacy per-hardware cell <id>-<hw>-<quant>.yaml (backward compatibility).
    #   4. Catalog default (spec.manifest_filename).
    hw_token = _manifest_hw(spec, hw)
    gpu_rel = f"configs/manifests/{spec.id}-gpu.yaml"
    legacy_rel = f"configs/manifests/{spec.id}-{hw_token}-{spec.serving.quantization}.yaml"
    default_rel = f"configs/manifests/{spec.manifest_filename}"

    if manifest:
        # User picked an explicit manifest (relative to repo root or absolute).
        manifest_rel = manifest if manifest.startswith("configs/") else f"configs/manifests/{manifest}"
    elif (paths.ROOT / gpu_rel).exists() and not spec.is_neuron:
        manifest_rel = gpu_rel
    elif (paths.ROOT / legacy_rel).exists():
        manifest_rel = legacy_rel
    else:
        if hw and not spec.is_neuron:
            ui.warn(
                f"No {gpu_rel} or per-hw {legacy_rel}; falling back to catalog default "
                f"{default_rel}. (Instance/cost still resolve dynamically from --hw.)"
            )
        manifest_rel = default_rel

    if not (paths.ROOT / manifest_rel).exists():
        ui.fail(
            f"Benchmark manifest not found: {manifest_rel}",
            f"Generate it first: oai model generate {model_id}",
        )

    if spec.benchmark.instance_hourly_usd <= 0:
        ui.warn("benchmark.instance_hourly_usd is 0 in the catalog - cost figures will be null.")
        ui.hint(f"Set it in catalog/models/{model_id}.yaml and re-run 'oai model generate {model_id}' for cost math.")

    profiles = _selected_profiles(spec, profile, skip_batch)
    profile_arg = _profile_arg(profiles)
    ds = dataset if dataset is not None else spec.benchmark.dataset
    # hw token (family, e.g. 'g7e') selects the manifest cell; the FULL instance
    # type (e.g. 'g7e.24xlarge') is passed separately for dynamic cost/label.
    hw_family = hw_token
    full_instance = hw if (hw and "." in hw) else None

    script = paths.ROOT / "inference" / "run-benchmark.sh"
    if not script.exists():
        ui.fail("inference/run-benchmark.sh is missing.", "This is part of the base framework - check your checkout.")

    # A --tag means we benchmarked a tagged (concurrent) deploy: point the runner
    # at that tagged service and isolate its S3 results so parallel runs don't
    # overwrite each other.
    svc = None
    if tag and not spec.is_neuron:
        svc = f"{spec.service_name}-{tag}"

    ui.banner(f"Benchmark: {spec.id}{f'  (tag={tag})' if tag else ''}")
    ui.kv("Manifest", manifest_rel)
    ui.kv("Profiles", f"{', '.join(profiles)}  (separate sequential Jobs)" if profile_arg == "both" else profile_arg)
    ui.kv("Dataset", ds or "built-in random (no PII)")
    if qid_cost:
        import os as _os  # noqa: PLC0415
        _ds_shown = ds or _os.environ.get("QID_COST_DATASET", "") or _read_config_env("QID_COST_DATASET") or "(not set)"
        _cfg_shown = qid_config or _os.environ.get("QID_COST_CONFIG", "") or _read_config_env("QID_COST_CONFIG") or "(not set)"
        ui.kv("QID-cost mode", "ON — per-QID benchmark across all concurrency levels")
        ui.kv("QID dataset", _ds_shown + (" (from config.env default)" if not ds else ""))
        ui.kv("QID config", _cfg_shown + (" (from config.env default)" if not qid_config else ""))
        ui.warn("QID-cost runs N_qids x N_concurrency bench invocations — may take significantly longer.")
        ui.warn("QID-cost runs N_qids x N_concurrency bench invocations — may take significantly longer.")

    args = ["--model", spec.id, "--profile", profile_arg, "--hw", hw_family, "--manifest", manifest_rel]
    if full_instance:
        # Pass the real deployed instance so cost/instance_type resolve dynamically
        # from the EC2 price book — no per-instance manifest edits needed.
        args += ["--instance", full_instance]
    if run_timestamp:
        args += ["--run-timestamp", run_timestamp]
    if detach:
        args += ["--detach"]
    if ds:
        args += ["--dataset", ds]
    if tag and svc:
        args += ["--tag", tag, "--svc", svc]
    if qid_cost:
        # Apply config.env defaults when the caller did not pass values
        # explicitly. After the first staging, `--qid-cost` alone is enough.
        import os as _os  # noqa: PLC0415
        resolved_dataset = ds or _os.environ.get("QID_COST_DATASET", "") or _read_config_env("QID_COST_DATASET")
        resolved_config  = qid_config or _os.environ.get("QID_COST_CONFIG", "") or _read_config_env("QID_COST_CONFIG")
        if not resolved_dataset:
            ui.fail(
                "--qid-cost requires a dataset.",
                "Pass --dataset <S3 key> or set QID_COST_DATASET in config/config.env.",
            )
        if not resolved_config:
            ui.fail(
                "--qid-cost requires a quality config.",
                "Pass --qid-config <path> or set QID_COST_CONFIG in config/config.env.",
            )
        if not ds and resolved_dataset:
            args += ["--dataset", resolved_dataset]
        args += ["--qid-cost"]
        if resolved_config:
            args += ["--qid-config", resolved_config]
        if qid_sheet:
            args += ["--qid-sheet", qid_sheet]

    rc = shell.run_bash(script, args)
    if rc != 0:
        # run-benchmark.sh exits 1 on SLO violation too - results may still be valid.
        ui.warn(f"Benchmark finished non-zero (exit {rc}).")
        ui.hint("This can mean an SLO threshold was not met (results still uploaded) or a real error.")
        ui.hint("Check the log above and the S3 results path it printed.")
        return rc

    if detach:
        ui.info("Performance Jobs submitted in-cluster. They continue independently of this terminal.")
    else:
        ui.info("Benchmark complete. Results are in the results bucket (path shown above).")

    # Standalone --auto-stop: submit an in-cluster autostop that waits for the
    # final benchmark Job (bench-jobs mode, kubectl-based) then stops the model.
    if auto_stop and not from_deploy_pipeline:
        last_profile = completion_profile(spec, profile, skip_batch)
        preflight.submit_autostop(
            spec.id, tag, run_timestamp,  # type: ignore[arg-type]
            wait_mode="bench-jobs",
            wait_for_profile=last_profile,
        )
    return 0
