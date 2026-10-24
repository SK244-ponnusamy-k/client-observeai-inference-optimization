"""
File: compile.py
Purpose: One-shot NxDI compile entrypoint for Qwen3.5 — traces/shards the model to a Neuron artifact.
Owner: genai-platform@shellkode
Created: 2026-10-18 | Deps: torch, neuronx_distributed_inference, src.modeling_qwen35
Jira: client-observeai-inference-optimization (Qwen3.5 nxdi bring-up)

Runs inside the NxDI inference DLC on a Trainium node. Builds the same
Qwen35InferenceConfig/NeuronConfig the server uses, calls model.compile() to
produce the traced + sharded artifact, and leaves it on the local path for the
compile Job to sync to S3. Compile-time parameters are read from the same
QWEN35_* environment variables the server reads, so compile and serve never
drift.

Idempotent: if the compiled path already contains artifacts, it exits 0 without
recompiling (the Job's S3 pre-check normally prevents even starting this).
"""

from __future__ import annotations

import json
import logging
import os
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qwen35_nxdi_compile")

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main() -> int:
    """Compile the model. Returns a process exit code."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    model_path = _env("QWEN35_MODEL_PATH", "/model")
    compiled_path = _env("QWEN35_COMPILED_PATH", "/compiled")
    tp_degree = int(_env("QWEN35_TP_DEGREE", "4"))
    seq_len = int(_env("QWEN35_SEQ_LEN", "160"))
    max_batch_size = int(_env("QWEN35_MAX_BATCH_SIZE", "1"))
    lnc = int(_env("QWEN35_LOGICAL_NC_CONFIG", "2"))
    dtype_key = _env("QWEN35_TORCH_DTYPE", "bfloat16").lower()
    top_k = int(_env("QWEN35_TOP_K_ON_DEVICE", "1"))

    if dtype_key not in _DTYPES:
        logger.error("Unsupported QWEN35_TORCH_DTYPE=%s", dtype_key)
        return 2
    torch_dtype = _DTYPES[dtype_key]

    config_json = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_json):
        logger.error("Weights not found: missing %s", config_json)
        return 2

    if os.path.isdir(compiled_path) and any(os.scandir(compiled_path)):
        logger.info("Compiled artifact already present at %s — skipping.", compiled_path)
        return 0

    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import Qwen35InferenceConfig, NeuronQwen35ForCausalLM

    logger.info(
        "Compiling Qwen3.5 NxDI: tp=%d seq_len=%d batch=%d lnc=%d dtype=%s",
        tp_degree, seq_len, max_batch_size, lnc, dtype_key,
    )

    # Sequence bucketing: compiling the full seq_len (8192) prefill as ONE
    # monolithic graph makes neuronx-cc re-materialize the fused DeltaNet NKI
    # kernel across all 24 DeltaNet layers at once, which OOMs the host even at
    # 240Gi. Bucketing compiles several smaller context-length graphs instead
    # (each fits in far less RAM); the runtime dispatches each request to the
    # smallest bucket that fits. Buckets are powers of two up to seq_len, and
    # are overridable via QWEN35_CONTEXT_BUCKETS (comma-separated) if needed.
    def _default_buckets(max_len: int) -> list[int]:
        candidates = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        buckets = [b for b in candidates if b < max_len]
        buckets.append(max_len)
        return buckets

    buckets_env = _env("QWEN35_CONTEXT_BUCKETS", "").strip()
    if buckets_env:
        context_buckets = sorted({int(x) for x in buckets_env.split(",") if x.strip()})
        if context_buckets[-1] != seq_len:
            context_buckets.append(seq_len)
            context_buckets = sorted(set(context_buckets))
    else:
        context_buckets = _default_buckets(seq_len)
    logger.info("Sequence bucketing ENABLED. context buckets=%s", context_buckets)

    # Build kwargs, then keep ONLY the ones this installed NeuronConfig accepts.
    # Bucketing kwarg names vary across NxDI versions (context_encoding_buckets,
    # token_generation_buckets, or a generic `buckets`), so probe the signature
    # instead of hardcoding — a wrong kwarg would TypeError at construction.
    base_kwargs = dict(
        tp_degree=tp_degree,
        batch_size=max_batch_size,
        ctx_batch_size=max_batch_size,
        tkg_batch_size=max_batch_size,
        max_batch_size=max_batch_size,
        seq_len=seq_len,
        torch_dtype=torch_dtype,
        logical_nc_config=lnc,
        flash_decoding_enabled=False,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=top_k),
        save_sharded_checkpoint=True,
    )
    bucket_kwargs = dict(
        enable_bucketing=True,
        context_encoding_buckets=context_buckets,
        token_generation_buckets=[seq_len],
        buckets=context_buckets,
    )

    import inspect

    try:
        accepted = set(inspect.signature(NeuronConfig.__init__).parameters)
    except (TypeError, ValueError):
        accepted = set()

    applied_bucket_kwargs = {k: v for k, v in bucket_kwargs.items() if k in accepted}
    if "enable_bucketing" in applied_bucket_kwargs and len(applied_bucket_kwargs) == 1:
        # enable_bucketing accepted but no bucket-list kwarg recognized — the
        # flag alone may not reduce the monolithic prefill, so fall back to a
        # smaller seq_len to keep compiler memory bounded.
        logger.warning(
            "NeuronConfig exposes enable_bucketing but no bucket-list kwarg; "
            "falling back to seq_len=4096 to bound compiler memory."
        )
        base_kwargs["seq_len"] = min(seq_len, 4096)
        applied_bucket_kwargs = {}
    elif not applied_bucket_kwargs:
        logger.warning(
            "This NeuronConfig does not accept bucketing kwargs; falling back to "
            "seq_len=4096 to bound compiler memory (was %d).", seq_len,
        )
        base_kwargs["seq_len"] = min(seq_len, 4096)

    logger.info("NeuronConfig bucket kwargs applied: %s", sorted(applied_bucket_kwargs))
    neuron_config = NeuronConfig(**base_kwargs, **applied_bucket_kwargs)

    with open(config_json, encoding="utf-8") as handle:
        hf_config = json.load(handle)
    text_config = hf_config.get("text_config", hf_config)
    config_dict = dict(text_config)
    config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
    config_dict["tie_word_embeddings"] = hf_config.get(
        "tie_word_embeddings",
        text_config.get("tie_word_embeddings", False),
    )
    if "rope_parameters" in text_config:
        config_dict["rope_theta"] = text_config["rope_parameters"].get("rope_theta", 10000000)

    config = Qwen35InferenceConfig(neuron_config=neuron_config, **config_dict)

    os.makedirs(compiled_path, exist_ok=True)
    model = NeuronQwen35ForCausalLM(model_path, config)
    model.compile(compiled_path)
    logger.info("Compile complete -> %s", compiled_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
