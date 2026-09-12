"""
Interactive `oai model new` wizard.

Asks plain-language questions and writes a catalog entry. Designed so a
non-technical user never opens a YAML editor. Every answer has a safe default
shown in [brackets]; pressing Enter accepts it.
"""

from __future__ import annotations

import re

import yaml

from . import catalog, paths, ui

_GPU_DEFAULTS = ["g6e.2xlarge", "g6.2xlarge", "g5.2xlarge"]
_NEURON_DEFAULTS = ["trn2.48xlarge", "trn1.32xlarge"]


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    ans = input(f"  {prompt}{suffix}: ").strip()
    return ans or default


def _ask_bool(prompt: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    ans = input(f"  {prompt} [{d}]: ").strip().lower()
    if not ans:
        return default
    return ans in ("y", "yes")


def run() -> str:
    """Run the wizard, write the catalog file, return the model id."""
    ui.banner("Onboard a new model")
    print("  Answer a few questions. Press Enter to accept the [default].\n")

    hf_id = _ask("HuggingFace model id (e.g. org/model)")
    while "/" not in hf_id:
        ui.warn("That does not look like a HuggingFace id. Expected 'org/model'.")
        hf_id = _ask("HuggingFace model id (e.g. org/model)")

    suggested_id = re.sub(r"[^a-z0-9-]", "-", hf_id.split("/")[-1].lower()).strip("-")
    model_id = _ask("Short name for this model (folder + resource name)", suggested_id)
    display_name = _ask("Friendly display name", hf_id.split("/")[-1])

    hw = _ask("Run on 'gpu' (NVIDIA) or 'neuron' (Trainium)?", "gpu").lower()
    while hw not in ("gpu", "neuron"):
        hw = _ask("Please type 'gpu' or 'neuron'", "gpu").lower()

    gated = _ask_bool("Is this a gated model (needs a license / HF token)?", False)
    size_gb = _ask("Approximate download size in GB", "15")

    print("\n  Which instance types may it run on? (preferred first, comma-separated)")
    default_insts = ",".join(_NEURON_DEFAULTS if hw == "neuron" else _GPU_DEFAULTS)
    candidates = [c.strip() for c in _ask("Instances", default_insts).split(",") if c.strip()]

    quant = _ask("Quantization (mxfp4 | fp8 | w4a16 | bf16 | none)", "bf16" if hw == "neuron" else "mxfp4")
    max_len = _ask("Context length (max_model_len)", "16384")
    hourly = _ask("Instance cost in $/hr (for cost math; 0 to skip)", "0")

    auto_bench = _ask_bool("Benchmark automatically right after deploy?", False)

    data: dict = {
        "id": model_id,
        "display_name": display_name,
        "hardware": hw,
        "source": {"hf_id": hf_id, "gated": gated, "size_gb": float(size_gb)},
        "serving": {"quantization": quant, "max_model_len": int(max_len)},
        "instances": {"candidates": candidates},
        "benchmark": {"auto": auto_bench, "instance_hourly_usd": float(hourly)},
    }

    if hw == "neuron":
        cores = _ask("NeuronCores per replica (= tensor-parallel degree)", "8")
        data["instances"]["neuron_cores"] = int(cores)
        family = "trn2" if any(c.startswith("trn2") for c in candidates) else "trn1"
        data["neuron"] = {
            "source_folder": model_id,
            "sequence_length": int(max_len),
            "batch_size": 8,
            "auto_cast_type": "bf16",
            "instance_family": family,
        }
        node_group = _ask("Managed node-group fallback name (optional, Enter to skip)", "")
        if node_group:
            data["instances"]["node_group"] = node_group
    else:
        tp = _ask("GPUs per replica (tensor-parallel size)", "1")
        data["instances"]["tensor_parallel_size"] = int(tp)

    path = paths.catalog_path(model_id)
    if path.exists() and not _ask_bool(f"'{model_id}' already exists - overwrite it?", False):
        ui.fail("Cancelled - a catalog entry with that id already exists.", f"Pick a different id or edit {path}.")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# catalog/models/{model_id}.yaml\n"
        f"# Created by 'oai model new'. Edit freely; see catalog/SCHEMA.md for all options.\n\n"
    )
    path.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8", newline="\n")

    ui.info(f"Wrote catalog entry: {path.relative_to(paths.ROOT).as_posix()}")

    # Validate immediately and report any issues.
    spec = catalog.from_dict(data)
    problems = catalog.validate(spec)
    if problems:
        ui.warn("The entry has some issues to review:")
        for p in problems:
            ui.hint(p)
    else:
        ui.info("Entry looks valid.")

    ui.banner("Next steps")
    print(f"  oai model generate {model_id}    # create the deployment files")
    print(f"  oai download {model_id}          # pull weights into S3")
    print(f"  oai deploy {model_id}            # run it on the cluster")
    return model_id
