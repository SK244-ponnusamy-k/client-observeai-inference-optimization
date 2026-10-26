"""
File: compile.py
Purpose: One-shot NxDI compile entrypoint for Qwen3.5 — traces/shards the model to a Neuron artifact.
Owner: genai-platform@shellkode
Created: 2026-10-18 | Deps: torch, neuronx_distributed_inference, src.modeling_qwen35
Jira: client-observeai-inference-optimization (Qwen3.5 nxdi bring-up)

Runs inside the NxDI inference DLC on a Trainium node. Builds the
Qwen35InferenceConfig/NeuronConfig, calls model.compile() to produce the traced
+ sharded artifact, and leaves it on the local path for the compile Job to sync
to S3. server.py imports build_neuron_config()/build_inference_config() from
this module for its first-boot compile, so compile and serve never drift.

Compile-time knobs (all env, all BAKED into the artifact):
  QWEN35_SEQ_LEN              total context (prompt + generation) = KV length
  QWEN35_MAX_CONTEXT_LEN      longest prompt (largest prefill bucket); <= SEQ_LEN
  QWEN35_CONTEXT_BUCKETS      prefill buckets, comma list (largest = MAX_CONTEXT_LEN)
  QWEN35_TKG_BUCKETS          decode buckets, comma list (largest = SEQ_LEN)
  QWEN35_MAX_BATCH_SIZE       decode batch (concurrent sequences per replica)
  QWEN35_CTX_BATCH_SIZE       prefill batch: = MAX_BATCH_SIZE (static) or 1
  QWEN35_CONTINUOUS_BATCHING  auto | 1 | 0 (auto -> on when CTX_BATCH < MAX_BATCH)
  QWEN35_TP_DEGREE / QWEN35_LOGICAL_NC_CONFIG / QWEN35_TORCH_DTYPE / QWEN35_TOP_K_ON_DEVICE
  QWEN35_DISABLE_ARGMAX_KERNEL 1 -> torch argmax instead of the NKI argmax kernel

Defaults (single bucket, CTX_BATCH == MAX_BATCH) reproduce the validated
TP=4 / seq 2048 / batch 4 build exactly.

NOTE on bucketing: the previous version probed inspect.signature(NeuronConfig)
for bucket kwargs. NeuronConfig.__init__ takes **kwargs, so that probe ALWAYS
failed and silently fell back to seq_len=min(seq_len, 4096) with bucketing off
(Troubleshooting #7 was a false negative). Kwargs are now passed directly and
verified on the constructed object; any mismatch fails before tracing starts.

Idempotent: if the compiled path already contains artifacts, exits 0.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qwen35_nxdi_compile")

BUILD_INFO_FILE = "qwen35_build.json"
_CHUNK = 128  # DeltaNet chunk size; buckets should be multiples of it


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int_list(name: str, default: List[int]) -> List[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return sorted({int(x) for x in raw.split(",") if x.strip()})


@dataclass
class BuildParams:
    tp_degree: int
    seq_len: int
    max_context_length: int
    max_batch_size: int
    ctx_batch_size: int
    continuous_batching: bool
    context_buckets: List[int] = field(default_factory=list)
    tkg_buckets: List[int] = field(default_factory=list)
    lnc: int = 2
    dtype: str = "bfloat16"
    top_k: int = 1
    disable_argmax_kernel: bool = False

    @property
    def bucketing(self) -> bool:
        return len(self.context_buckets) > 1 or len(self.tkg_buckets) > 1


def resolve_build_params() -> BuildParams:
    """Read + validate compile parameters from env. Raises ValueError on bad combos."""
    seq_len = int(_env("QWEN35_SEQ_LEN", "2048"))
    max_ctx = int(_env("QWEN35_MAX_CONTEXT_LEN", str(seq_len)))
    max_batch = int(_env("QWEN35_MAX_BATCH_SIZE", "1"))
    ctx_batch = int(_env("QWEN35_CTX_BATCH_SIZE", str(max_batch)))
    cb_raw = _env("QWEN35_CONTINUOUS_BATCHING", "auto").strip().lower()

    if max_ctx > seq_len:
        raise ValueError(f"QWEN35_MAX_CONTEXT_LEN={max_ctx} > QWEN35_SEQ_LEN={seq_len}")
    if ctx_batch not in (1, max_batch):
        raise ValueError(
            f"QWEN35_CTX_BATCH_SIZE must be 1 or MAX_BATCH_SIZE ({max_batch}); got {ctx_batch}"
        )
    cb = (ctx_batch < max_batch) if cb_raw == "auto" else cb_raw in ("1", "true", "yes")
    if ctx_batch < max_batch and not cb:
        # Without CB, NxDI's CTE KV write (fill_prefix) always targets batch row 0.
        raise ValueError("CTX_BATCH_SIZE < MAX_BATCH_SIZE requires QWEN35_CONTINUOUS_BATCHING=1")
    if cb and ctx_batch != 1:
        raise ValueError("continuous batching requires QWEN35_CTX_BATCH_SIZE=1 (NxDI KV manager)")

    cte = [b for b in _int_list("QWEN35_CONTEXT_BUCKETS", [max_ctx]) if b <= max_ctx]
    if not cte or cte[-1] != max_ctx:
        cte = sorted(set(cte + [max_ctx]))
    tkg = [b for b in _int_list("QWEN35_TKG_BUCKETS", [seq_len]) if b <= seq_len]
    if not tkg or tkg[-1] != seq_len:
        tkg = sorted(set(tkg + [seq_len]))
    for b in cte + tkg:
        if b % _CHUNK:
            logger.warning("bucket %d is not a multiple of %d (DeltaNet pads internally)", b, _CHUNK)

    return BuildParams(
        tp_degree=int(_env("QWEN35_TP_DEGREE", "4")),
        seq_len=seq_len,
        max_context_length=max_ctx,
        max_batch_size=max_batch,
        ctx_batch_size=ctx_batch,
        continuous_batching=cb,
        context_buckets=cte,
        tkg_buckets=tkg,
        lnc=int(_env("QWEN35_LOGICAL_NC_CONFIG", "2")),
        dtype=_env("QWEN35_TORCH_DTYPE", "bfloat16").lower(),
        top_k=int(_env("QWEN35_TOP_K_ON_DEVICE", "1")),
        disable_argmax_kernel=_env("QWEN35_DISABLE_ARGMAX_KERNEL", "0").strip() in ("1", "true", "yes"),
    )


def build_neuron_config(params: BuildParams):
    """Construct the NeuronConfig and verify every requested field actually took effect."""
    import torch
    from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig

    dtypes = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16,
              "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}
    if params.dtype not in dtypes:
        raise ValueError(f"Unsupported QWEN35_TORCH_DTYPE={params.dtype}")

    kwargs: Dict[str, Any] = dict(
        tp_degree=params.tp_degree,
        batch_size=params.max_batch_size,       # also sizes the KV cache batch dim
        ctx_batch_size=params.ctx_batch_size,
        tkg_batch_size=params.max_batch_size,
        max_batch_size=params.max_batch_size,
        seq_len=params.seq_len,
        max_context_length=params.max_context_length,
        torch_dtype=dtypes[params.dtype],
        logical_nc_config=params.lnc,
        padding_side="right",                   # NxDI KV/mask bookkeeping assumes right padding
        is_continuous_batching=params.continuous_batching,
        flash_decoding_enabled=False,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=params.top_k),
        save_sharded_checkpoint=True,
        # Greedy on-device sampling uses the NKI argmax kernel by default. On the
        # 10k / batch-8 / CB decode graph it returned float bits as the token id
        # (TROUBLESHOOTING #14). True -> plain torch argmax + TP all-gather.
        disable_argmax_kernel=params.disable_argmax_kernel,
    )
    if params.bucketing:
        kwargs.update(
            enable_bucketing=True,
            context_encoding_buckets=list(params.context_buckets),
            token_generation_buckets=list(params.tkg_buckets),
        )
    else:
        kwargs["enable_bucketing"] = False

    nc = NeuronConfig(**kwargs)

    # Fail fast if this SDK silently ignored anything (NeuronConfig warns, not raises).
    skip = {"on_device_sampling_config", "torch_dtype", "logical_nc_config"}
    bad = {k: (v, getattr(nc, k, "<missing>")) for k, v in kwargs.items()
           if k not in skip and getattr(nc, k, "<missing>") != v}
    if bad:
        raise RuntimeError(f"NeuronConfig did not apply requested fields: {bad}")
    return nc


def build_inference_config(neuron_config, model_path: str):
    from src.modeling_qwen35 import Qwen35InferenceConfig

    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as handle:
        hf_config = json.load(handle)
    text_config = hf_config.get("text_config", hf_config)
    config_dict = dict(text_config)
    config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
    config_dict["tie_word_embeddings"] = hf_config.get(
        "tie_word_embeddings", text_config.get("tie_word_embeddings", False)
    )
    if "rope_parameters" in text_config:
        config_dict["rope_theta"] = text_config["rope_parameters"].get("rope_theta", 10000000)
    return Qwen35InferenceConfig(neuron_config=neuron_config, **config_dict)


def write_build_info(compiled_path: str, params: BuildParams) -> None:
    info = asdict(params)
    info["env"] = {k: os.environ.get(k, "") for k in ("QWEN35_TEXT_ONLY", "USE_NKI_CHUNKED", "USE_NKI_FUSED")}
    with open(os.path.join(compiled_path, BUILD_INFO_FILE), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)


def compile_model(model_path: str, compiled_path: str, params: BuildParams) -> None:
    from src.modeling_qwen35 import NeuronQwen35ForCausalLM

    neuron_config = build_neuron_config(params)
    config = build_inference_config(neuron_config, model_path)
    os.makedirs(compiled_path, exist_ok=True)
    model = NeuronQwen35ForCausalLM(model_path, config)
    model.compile(compiled_path)
    write_build_info(compiled_path, params)


def main() -> int:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    model_path = _env("QWEN35_MODEL_PATH", "/model")
    compiled_path = _env("QWEN35_COMPILED_PATH", "/compiled")

    try:
        params = resolve_build_params()
    except ValueError as exc:
        logger.error("Invalid compile parameters: %s", exc)
        return 2
    logger.info("Compile parameters: %s (bucketing=%s)", json.dumps(asdict(params)), params.bucketing)
    logger.info("Kernel env: USE_NKI_CHUNKED=%s QWEN35_TEXT_ONLY=%s",
                os.environ.get("USE_NKI_CHUNKED", ""), os.environ.get("QWEN35_TEXT_ONLY", ""))

    if not os.path.isfile(os.path.join(model_path, "config.json")):
        logger.error("Weights not found: missing %s/config.json", model_path)
        return 2
    if os.path.isdir(compiled_path) and any(os.scandir(compiled_path)):
        logger.info("Compiled artifact already present at %s — skipping.", compiled_path)
        return 0

    compile_model(model_path, compiled_path, params)
    logger.info("Compile complete -> %s", compiled_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
