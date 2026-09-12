"""
Catalog entry model, loader, and validator.

A catalog entry (catalog/models/<id>.yaml) is the ONLY file a user edits to
onboard a model. This module turns that YAML into a validated `ModelSpec` with
all defaults resolved, so the generator and CLI never deal with raw dicts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import paths
from .ui import OaiError

_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_VALID_HARDWARE = ("gpu", "neuron")
_VALID_QUANT = ("mxfp4", "fp8", "w4a16", "bf16", "fp16", "none")

# Known GPU instance families and their per-card VRAM (GiB), used for sizing checks.
_GPU_VRAM = {
    "g5": 24,    # A10G
    "g6": 24,    # L4
    "g6e": 48,   # L40S
    "g7": 96,    # (Blackwell-class, custom AMI) - approximate
    "g7e": 96,
    "p4": 40,    # A100 40G
    "p5": 80,    # H100 80G
}


@dataclass
class Source:
    hf_id: str
    gated: bool = False
    size_gb: float = 0.0
    ignore_patterns: list[str] = field(default_factory=list)


@dataclass
class Serving:
    served_name: str = ""
    tokenizer: str = ""
    quantization: str = "none"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    extra_args: list[str] = field(default_factory=list)


@dataclass
class Resources:
    cpu_request: str = "4"
    cpu_limit: str = "8"
    memory_request: str = "24Gi"
    memory_limit: str = "30Gi"
    vram_gb_required: float = 0.0


@dataclass
class Instances:
    tensor_parallel_size: int = 1
    neuron_cores: int = 0
    candidates: list[str] = field(default_factory=list)
    node_group: str = ""


@dataclass
class Neuron:
    source_folder: str = ""
    neff_folder: str = ""
    sequence_length: int = 16384
    batch_size: int = 8
    auto_cast_type: str = "bf16"
    instance_family: str = "trn2"


@dataclass
class Benchmark:
    auto: bool = False
    profiles: list[str] = field(default_factory=list)
    instance_hourly_usd: float = 0.0
    dataset: str | None = None
    skip_batch: bool = False


@dataclass
class ModelSpec:
    id: str
    display_name: str
    hardware: str
    source: Source
    serving: Serving
    resources: Resources
    instances: Instances
    neuron: Neuron | None
    benchmark: Benchmark
    tags: dict[str, str] = field(default_factory=dict)

    # ---- derived names (single source of truth for resource naming) ----------
    @property
    def deployment_name(self) -> str:
        return f"oai-infopt-vllm-{self.id}"

    @property
    def service_name(self) -> str:
        return f"oai-infopt-vllm-{self.id}"

    @property
    def pvc_name(self) -> str:
        return f"oai-infopt-metadata-{self.id}"

    @property
    def download_job_name(self) -> str:
        return f"oai-infopt-download-{self.id}"

    @property
    def compile_job_name(self) -> str:
        return f"oai-infopt-neuron-compile-{self.id}"

    @property
    def is_neuron(self) -> bool:
        return self.hardware == "neuron"

    @property
    def s3_model_folder(self) -> str:
        """The S3 folder the deployment serves from."""
        if self.is_neuron and self.neuron:
            # Serve from the source (bf16) folder; NEFF cache is a serving-time detail.
            return self.neuron.source_folder or self.id
        return self.id

    @property
    def preferred_instance(self) -> str:
        return self.instances.candidates[0] if self.instances.candidates else ""

    @property
    def manifest_filename(self) -> str:
        hw = self.preferred_instance.split(".")[0] if self.preferred_instance else self.hardware
        quant = self.serving.quantization
        return f"{self.id}-{hw}-{quant}.yaml"


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d or d[key] in (None, ""):
        raise OaiError(
            f"Catalog entry is missing required field '{key}' ({ctx}).",
            f"Add '{key}:' to the {ctx} section. See catalog/SCHEMA.md.",
        )
    return d[key]


def from_dict(data: dict[str, Any]) -> ModelSpec:
    """Build a ModelSpec from a parsed catalog dict, applying defaults."""
    if not isinstance(data, dict):
        raise OaiError("Catalog file is not a valid YAML mapping.", "Check the file for YAML syntax errors.")

    model_id = _require(data, "id", "top level")
    hardware = _require(data, "hardware", "top level")

    src_raw = _require(data, "source", "top level")
    source = Source(
        hf_id=_require(src_raw, "hf_id", "source"),
        gated=bool(src_raw.get("gated", False)),
        size_gb=float(src_raw.get("size_gb", 0) or 0),
        ignore_patterns=list(src_raw.get("ignore_patterns", []) or []),
    )

    srv_raw = data.get("serving", {}) or {}
    serving = Serving(
        served_name=srv_raw.get("served_name") or model_id,
        tokenizer=srv_raw.get("tokenizer") or source.hf_id,
        quantization=srv_raw.get("quantization", "none"),
        max_model_len=int(srv_raw.get("max_model_len", 8192)),
        gpu_memory_utilization=float(srv_raw.get("gpu_memory_utilization", 0.90)),
        max_num_seqs=int(srv_raw.get("max_num_seqs", 256)),
        max_num_batched_tokens=int(srv_raw.get("max_num_batched_tokens", 8192)),
        extra_args=list(srv_raw.get("extra_args", []) or []),
    )

    res_raw = data.get("resources", {}) or {}
    # Derive a sensible VRAM requirement from model size if not given.
    default_vram = max(int(source.size_gb * 1.3) if source.size_gb else 0, 0)
    resources = Resources(
        cpu_request=str(res_raw.get("cpu_request", "4")),
        cpu_limit=str(res_raw.get("cpu_limit", "8")),
        memory_request=str(res_raw.get("memory_request", "24Gi")),
        memory_limit=str(res_raw.get("memory_limit", "30Gi")),
        vram_gb_required=float(res_raw.get("vram_gb_required", default_vram)),
    )

    inst_raw = data.get("instances", {}) or {}
    instances = Instances(
        tensor_parallel_size=int(inst_raw.get("tensor_parallel_size", 1)),
        neuron_cores=int(inst_raw.get("neuron_cores", 0)),
        candidates=list(inst_raw.get("candidates", []) or []),
        node_group=inst_raw.get("node_group", "") or "",
    )

    neuron = None
    if hardware == "neuron":
        neu_raw = data.get("neuron", {}) or {}
        source_folder = neu_raw.get("source_folder") or model_id
        neuron = Neuron(
            source_folder=source_folder,
            neff_folder=neu_raw.get("neff_folder") or f"{source_folder}-neuron",
            sequence_length=int(neu_raw.get("sequence_length", serving.max_model_len)),
            batch_size=int(neu_raw.get("batch_size", 8)),
            auto_cast_type=neu_raw.get("auto_cast_type", "bf16"),
            instance_family=neu_raw.get("instance_family", "trn2"),
        )

    bench_raw = data.get("benchmark", {}) or {}
    default_profiles = [
        "configs/workload_profiles/realtime_v1.yaml",
        "configs/workload_profiles/batch_v1.yaml",
    ]
    benchmark = Benchmark(
        auto=bool(bench_raw.get("auto", False)),
        profiles=list(bench_raw.get("profiles", default_profiles) or default_profiles),
        instance_hourly_usd=float(bench_raw.get("instance_hourly_usd", 0) or 0),
        dataset=bench_raw.get("dataset"),
        skip_batch=bool(bench_raw.get("skip_batch", False)),
    )

    return ModelSpec(
        id=model_id,
        display_name=data.get("display_name") or model_id,
        hardware=hardware,
        source=source,
        serving=serving,
        resources=resources,
        instances=instances,
        neuron=neuron,
        benchmark=benchmark,
        tags=dict(data.get("tags", {}) or {}),
    )


def load(model_id: str) -> ModelSpec:
    """Load and parse a catalog entry by id."""
    path = paths.catalog_path(model_id)
    if not path.exists():
        available = list_ids()
        hint = (
            f"Available models: {', '.join(available)}"
            if available
            else "No catalog entries yet. Run 'oai model new' to create one."
        )
        raise OaiError(f"No catalog entry found for '{model_id}' (looked in {path}).", hint)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise OaiError(f"Catalog file for '{model_id}' has a YAML error: {exc}", "Fix the YAML syntax and try again.")
    return from_dict(data)


def list_ids() -> list[str]:
    if not paths.CATALOG_DIR.exists():
        return []
    return sorted(p.stem for p in paths.CATALOG_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Validation - returns a list of human-readable problems (empty = all good).
# ---------------------------------------------------------------------------
def validate(spec: ModelSpec) -> list[str]:
    problems: list[str] = []

    if not _ID_RE.match(spec.id):
        problems.append(
            f"id '{spec.id}' is invalid - use lowercase letters, digits and dashes, starting with a letter."
        )

    if spec.hardware not in _VALID_HARDWARE:
        problems.append(f"hardware '{spec.hardware}' is invalid - must be one of {', '.join(_VALID_HARDWARE)}.")

    if "/" not in spec.source.hf_id:
        problems.append(
            f"source.hf_id '{spec.source.hf_id}' does not look like a HuggingFace id (expected 'org/model')."
        )

    if spec.serving.quantization not in _VALID_QUANT:
        problems.append(
            f"serving.quantization '{spec.serving.quantization}' is unusual - expected one of {', '.join(_VALID_QUANT)}."
        )

    if not spec.instances.candidates:
        problems.append("instances.candidates is empty - list at least one instance type to run on.")

    if spec.hardware == "gpu":
        for inst in spec.instances.candidates:
            fam = inst.split(".")[0]
            if fam not in _GPU_VRAM:
                problems.append(f"instances: '{inst}' is not a recognised GPU family ({fam}).")
                continue
            per_card = _GPU_VRAM[fam]
            total_vram = per_card * max(spec.instances.tensor_parallel_size, 1)
            if spec.resources.vram_gb_required and total_vram < spec.resources.vram_gb_required:
                problems.append(
                    f"instances: '{inst}' provides ~{total_vram} GiB VRAM "
                    f"(TP={spec.instances.tensor_parallel_size}) but the model needs "
                    f"~{spec.resources.vram_gb_required:.0f} GiB. It will not fit."
                )

    if spec.hardware == "neuron":
        if spec.instances.neuron_cores <= 0:
            problems.append("instances.neuron_cores must be > 0 for Neuron models (it is the tensor-parallel degree).")
        if spec.serving.quantization in ("mxfp4", "w4a16"):
            problems.append(
                f"serving.quantization '{spec.serving.quantization}' is not supported on Neuron - "
                "Neuron supports fp16/bf16 only. Set quantization to bf16 and compile from base weights."
            )
        if not spec.neuron:
            problems.append("Neuron model is missing the 'neuron:' compile block. See catalog/SCHEMA.md.")

    if spec.benchmark.instance_hourly_usd <= 0:
        problems.append(
            "benchmark.instance_hourly_usd is 0 - cost figures will be null. "
            "Set the $/hr for your chosen instance and region (this is a warning, not fatal)."
        )

    return problems
