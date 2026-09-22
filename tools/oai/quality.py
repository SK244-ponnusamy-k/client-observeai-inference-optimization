"""Quality-evaluation orchestration.

Provides the product-facing ``oai quality <id>`` command and wraps
``inference/run-quality.sh`` using catalog data.  The catalog model ID is used
for Kubernetes/S3 labels while ``serving.served_name`` is sent to the vLLM
OpenAI endpoint; these differ for models such as Qwen3.5-4B.
"""

from __future__ import annotations

from . import catalog, paths, shell, ui


def run(
    model_id: str,
    *,
    tag: str | None = None,
    hw: str | None = None,
    quant: str | None = None,
    config: str | None = None,
    dataset_key: str | None = None,
    service: str | None = None,
    served_name: str | None = None,
    dump_samples: bool = False,
    max_dump_samples: int = 200,
    dump_inputs: bool = False,
    run_timestamp: str | None = None,
    wait_for_marker: str | None = None,
    detach: bool = False,
) -> int:
    """Submit an AutoQA quality Job against an already-deployed model.

    ``wait_for_marker`` is an optional S3 key used by deploy orchestration to
    make quality the final stage after performance benchmarks.  Standalone
    quality runs omit it and start immediately.
    """
    spec = catalog.load(model_id)
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

    if tag and spec.is_neuron:
        ui.warn("--tag is not supported for Neuron models; using the untagged service.")
        tag = None

    selected_hw = hw or spec.preferred_instance or spec.hardware
    selected_quant = quant or spec.serving.quantization
    selected_served_name = served_name or spec.serving.served_name or spec.id
    selected_service = service or spec.service_name
    if tag and not service:
        selected_service = f"{selected_service}-{tag}"

    args = [
        "--model", spec.id,
        "--served-name", selected_served_name,
        "--hw", selected_hw,
        "--quant", selected_quant,
        "--svc", selected_service,
    ]
    if config:
        args += ["--config", config]
    if dataset_key:
        args += ["--dataset-key", dataset_key]
    if tag:
        args += ["--tag", tag]
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

    ui.banner(f"Quality: {spec.id}{f'  (tag={tag})' if tag else ''}")
    ui.kv("Service", selected_service)
    ui.kv("Served model", selected_served_name)
    ui.kv("Hardware", selected_hw)
    ui.kv("Quantization", selected_quant)
    ui.kv("Samples", f"dump first {max_dump_samples}" if dump_samples else "aggregate metrics only")
    if dump_inputs:
        ui.kv("Inputs", "INCLUDED in samples (call transcripts are persisted)")
    if wait_for_marker:
        ui.kv("Starts after", f"s3 marker {wait_for_marker}")

    rc = shell.run_bash(script, args)
    if rc != 0:
        ui.warn(f"Quality job submission/run reported non-zero (exit {rc}).")
        ui.hint("Check the Job logs and the S3 quality results path printed above.")
        return rc

    if detach:
        ui.info("Quality Job submitted in-cluster. It continues independently of this terminal.")
    else:
        ui.info("Quality evaluation finished. Results are in the results bucket (path shown above).")
    return 0
