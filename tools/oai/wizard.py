"""
Interactive `oai model new` wizard.

Asks plain-language questions and writes a catalog entry. Designed so a
non-technical user never opens a YAML editor. Every answer has a safe default
shown in [brackets]; pressing Enter accepts it.

Two fields are always prompted because they cannot be safely defaulted:
  - source.s3_folder  : S3 folder name (defaults to model id, but must be
                        confirmed when the HF repo name differs from the id)
  - serving.extra_args: vLLM serving flags including the reasoning parser and
                        enable_thinking kwarg (user picks yes/no per flag)

All other fields (resources, benchmark.profiles, tags, served_name, tokenizer,
gpu_memory_utilization, max_num_seqs, max_num_batched_tokens) are written with
the same values every existing model uses — no prompts needed.
"""

from __future__ import annotations

import re

import yaml

from . import catalog, paths, ui

_GPU_DEFAULTS = ["g6e.2xlarge", "g6.2xlarge", "g5.2xlarge"]
_NEURON_DEFAULTS = ["trn2.3xlarge", "trn2.48xlarge", "trn1.32xlarge"]

# Reasoning parsers known to this framework — shown as a pick-list.
_REASONING_PARSERS = {
    "1": ("deepseek_r1",   "DeepSeek-R1 / Distill variants"),
    "2": ("qwen3",         "Qwen3 / Qwen3.5"),
    "3": ("gemma4",        "Gemma 4"),
    "4": ("openai_gptoss", "GPT-OSS (OpenAI)"),
    "5": ("none",          "No reasoning parser (non-reasoning model)"),
}


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


def _ask_reasoning_parser() -> str:
    """Present a numbered menu and return the chosen parser name (or 'none')."""
    print("\n  Reasoning parser — which model family is this?")
    for key, (parser, label) in _REASONING_PARSERS.items():
        print(f"    {key}) {label}  ({parser})")
    choice = input("  Pick [1-5]: ").strip()
    entry = _REASONING_PARSERS.get(choice)
    if entry:
        return entry[0]
    ui.warn(f"Unrecognised choice '{choice}' — defaulting to 'none'.")
    return "none"


def _build_extra_args(reasoning_parser: str, enable_thinking: bool) -> list[str]:
    """
    Assemble the extra_args list using the same pattern every existing model uses.

    Fixed flags (always included):
      --enable-chunked-prefill
      --enable-prefix-caching

    Reasoning ON  (parser selected + enable_thinking=true):
      --reasoning-parser=<parser>          <- active
      --default-chat-template-kwargs={"enable_thinking": true}

    Reasoning OFF (enable_thinking=false):
      # --reasoning-parser=<parser>        <- commented out, easy to re-enable
      --default-chat-template-kwargs={"enable_thinking": false}
    """
    args: list[str] = [
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
    ]
    if reasoning_parser != "none" and enable_thinking:
        # Reasoning ON — both parser and enable_thinking active
        args.append(f"--reasoning-parser={reasoning_parser}")
        args.append('--default-chat-template-kwargs={\\\"enable_thinking\\\": true}')
    elif reasoning_parser != "none" and not enable_thinking:
        # Reasoning OFF — parser commented out, enable_thinking false written explicitly
        args.append(f"# --reasoning-parser={reasoning_parser}")
        args.append('--default-chat-template-kwargs={\\\"enable_thinking\\\": false}')
    # parser == "none": non-reasoning model — no reasoning args at all
    return args


def run() -> str:
    """Run the wizard, write the catalog file, return the model id."""
    ui.banner("Onboard a new model")
    print("  Answer a few questions. Press Enter to accept the [default].\n")

    # ── identity ──────────────────────────────────────────────────────────────
    hf_id = _ask("HuggingFace model id (e.g. org/model)")
    while "/" not in hf_id:
        ui.warn("That does not look like a HuggingFace id. Expected 'org/model'.")
        hf_id = _ask("HuggingFace model id (e.g. org/model)")

    suggested_id = re.sub(r"[^a-z0-9-]", "-", hf_id.split("/")[-1].lower()).strip("-")
    model_id     = _ask("Short name for this model (folder + resource name)", suggested_id)
    display_name = _ask("Friendly display name", hf_id.split("/")[-1])

    hw = _ask("Run on 'gpu' (NVIDIA) or 'neuron' (Trainium)?", "gpu").lower()
    while hw not in ("gpu", "neuron"):
        hw = _ask("Please type 'gpu' or 'neuron'", "gpu").lower()

    gated   = _ask_bool("Is this a gated model (needs a license / HF token)?", False)
    size_gb = _ask("Approximate download size in GB", "15")

    # ── source.s3_folder (prompted — may differ from model_id) ───────────────
    print("\n  S3 folder where weights are stored.")
    print("  Press Enter if the folder name matches the model short name above.")
    s3_folder_raw = _ask("S3 folder name (leave blank = same as short name)", model_id)
    # Store only when it actually differs; at load time the catalog defaults to id.
    s3_folder = s3_folder_raw if s3_folder_raw != model_id else ""

    # ── instances ─────────────────────────────────────────────────────────────
    print("\n  Which instance types may it run on? (preferred first, comma-separated)")
    default_insts = ",".join(_NEURON_DEFAULTS if hw == "neuron" else _GPU_DEFAULTS)
    candidates = [c.strip() for c in _ask("Instances", default_insts).split(",") if c.strip()]

    # ── serving basics ────────────────────────────────────────────────────────
    quant   = _ask("Quantization (mxfp4 | fp8 | w4a16 | bf16 | none)", "bf16" if hw == "neuron" else "none")
    max_len = _ask("Context length (max_model_len)", "16384")

    # ── extra_args: reasoning parser + enable_thinking (always prompted) ──────
    reasoning_parser = _ask_reasoning_parser()
    enable_thinking  = False
    if reasoning_parser != "none":
        enable_thinking = _ask_bool(
            "Enable thinking (chain-of-thought) by default on every request?", True
        )
    extra_args = _build_extra_args(reasoning_parser, enable_thinking)

    # ── benchmark ─────────────────────────────────────────────────────────────
    hourly     = _ask("\n  Instance cost in $/hr (for cost math; 0 to skip)", "0")
    auto_bench = _ask_bool("Benchmark automatically right after deploy?", False)

    # ── assemble data dict ────────────────────────────────────────────────────
    # Fields NOT prompted — kept at the same values every existing model uses.
    # They are written explicitly so the file is self-documenting and the user
    # can tweak them without hunting for defaults.
    source_block: dict = {
        "hf_id":  hf_id,
        "gated":  gated,
        "size_gb": float(size_gb),
    }
    if s3_folder:
        source_block["s3_folder"] = s3_folder

    serving_block: dict = {
        "served_name":            model_id,
        "tokenizer":              hf_id,
        "quantization":           quant,
        "max_model_len":          int(max_len),
        "gpu_memory_utilization": 0.90,
        "max_num_seqs":           256,
        "max_num_batched_tokens": 8192,
        "extra_args":             extra_args,
    }

    # Resources — same defaults used by gpt-oss-20b (single-GPU TP=1 model).
    # Users with larger models should edit cpu/memory after generation.
    resources_block: dict = {
        "cpu_request":    "4",
        "cpu_limit":      "8",
        "memory_request": "24Gi",
        "memory_limit":   "30Gi",
        "vram_gb_required": max(int(float(size_gb) * 1.3), 8),
    }

    data: dict = {
        "id":           model_id,
        "display_name": display_name,
        "hardware":     hw,
        "source":       source_block,
        "serving":      serving_block,
        "resources":    resources_block,
        "instances":    {"candidates": candidates},
        "benchmark": {
            "auto": auto_bench,
            "profiles": [
                "configs/workload_profiles/realtime_v1.yaml",
                "configs/workload_profiles/batch_v1.yaml",
            ],
            "instance_hourly_usd": float(hourly),
        },
        "tags": {
            "Project":     "observeai-inference-optimization",
            "Engagement":  "shellkode-sow",
            "Environment": "sandbox",
            "ManagedBy":   "oai-cli",
            "Owner":       "TBD",
            "CostCenter":  "TBD",
        },
    }

    # ── hardware-specific extras ──────────────────────────────────────────────
    if hw == "neuron":
        cores = _ask("NeuronCores per replica (= tensor-parallel degree)", "8")
        data["instances"]["neuron_cores"] = int(cores)
        family = "trn2" if any(c.startswith("trn2") for c in candidates) else "trn1"
        data["neuron"] = {
            "source_folder":  model_id,
            "sequence_length": int(max_len),
            "batch_size":      8,
            "auto_cast_type":  "bf16",
            "instance_family": family,
        }
        node_group = _ask("Managed node-group fallback name (optional, Enter to skip)", "")
        if node_group:
            data["instances"]["node_group"] = node_group
    else:
        tp = _ask("GPUs per replica (tensor-parallel size)", "1")
        data["instances"]["tensor_parallel_size"] = int(tp)

    # ── write file ────────────────────────────────────────────────────────────
    path = paths.catalog_path(model_id)
    if path.exists() and not _ask_bool(f"\n  '{model_id}' already exists — overwrite it?", False):
        ui.fail("Cancelled - a catalog entry with that id already exists.", f"Pick a different id or edit {path}.")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# catalog/models/{model_id}.yaml\n"
        f"# Created by 'oai model new'. Edit freely; see catalog/SCHEMA.md for all options.\n\n"
    )
    path.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8", newline="\n")

    ui.info(f"Wrote catalog entry: {path.relative_to(paths.ROOT).as_posix()}")

    # Validate immediately and report any issues.
    spec     = catalog.from_dict(data)
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
