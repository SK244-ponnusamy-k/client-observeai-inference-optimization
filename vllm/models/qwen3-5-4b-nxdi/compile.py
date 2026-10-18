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

    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=max_batch_size,
        ctx_batch_size=max_batch_size,
        tkg_batch_size=max_batch_size,
        max_batch_size=max_batch_size,
        seq_len=seq_len,
        torch_dtype=torch_dtype,
        logical_nc_config=lnc,
        enable_bucketing=False,
        flash_decoding_enabled=False,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=top_k),
        save_sharded_checkpoint=True,
    )

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
