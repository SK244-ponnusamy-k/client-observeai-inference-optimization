# Benchmark Matrix — Models × Hardware

This is the authoritative spec for **what we benchmark and where**, per the Observe.AI
SoW. The framework is model- and hardware-agnostic; this document pins the concrete
matrix, the chosen config per cell, and the tradeoffs behind each choice.

- **Load generator:** `vllm bench serve` (tokenizer-accurate TTFT / TPOT / ITL / E2E +
  throughput), orchestrated by `inference/load-test.py`.
- **Serving:** AWS vLLM DLC (`0.26.0`, digest-pinned) on Amazon EKS, weights streamed
  from S3 via Run:ai. One LLM per GPU node.
- **Profiles:** `realtime_v1` (low concurrency, latency-bound) and `batch_v1` (high
  concurrency, throughput-bound). Frozen — see `configs/workload_profiles/`.
- **Metrics per cell:** TTFT / ITL / TPOT / E2E p50/p95/p99, output tokens/s,
  completed interactions/min, KV-cache %, queue depth, GPU util (DCGM), and
  normalized cost (cost/1M tokens, cost/QA form, tokens/$). Quality (accuracy) is a
  **separate** lm-eval stage — not this harness.

## Models (locked)

| Model | HF ID | Type | Params | Serve precision | Weights |
|---|---|---|---|---|---|
| **gpt-oss-20b** | `openai/gpt-oss-20b` | MoE (3.6B active) | 21B total | MXFP4 (native) | ~13 GB |
| **Qwen3.5-4B** | `Qwen/Qwen3.5-4B` | Dense, multimodal¹ | 4B | bf16 | ~9 GB |
| **Gemma-4-26B-A4B** | `google/gemma-4-26B-A4B-it` | MoE (3.8B active), multimodal¹ | 25.2B total | 4-bit W4A16 | ~14.4 GB |

¹ Qwen3.5-4B and Gemma-4 are multimodal. We benchmark the **text path only** (Observe.AI
transcript QA). Image/video inputs are not exercised.

**Smoke test:** `Qwen/Qwen2.5-0.5B-Instruct` (bf16, ~1 GB) validates the whole
onboard→deploy→benchmark→teardown loop on the cheapest GPU before spending on the matrix.

## Hardware families

| Family | GPU | VRAM/card | NVLink | Notes |
|---|---|---|---|---|
| **g5** | A10G | 24 GB | no | PCIe only |
| **g6** | L4 | 24 GB | no | PCIe only; cheapest |
| **g6e** | L40S | 48 GB | no | PCIe only; primary target |
| **trn** | Trainium2 | — | — | Neuron **Flow B** (compile stage) — see below |

**G7 is excluded**: it needs NVIDIA driver 595+, but stock EKS accelerated AMIs ship 580.
The `gpu-inf` NodePool constrains instance generation to 5/6 (`cluster/gpu-nodepool.yaml`).

## Feasibility grid (single GPU unless noted)

| Model | g5 (A10G 24 GB) | g6 (L4 24 GB) | g6e (L40S 48 GB) | trn (Neuron) |
|---|---|---|---|---|
| **gpt-oss-20b** (MXFP4 ~13 GB) | ✅ 1 GPU | ✅ 1 GPU | ✅ 1 GPU | ⚠ validate (MoE/MXFP4 on Neuron unconfirmed) |
| **Qwen3.5-4B** (bf16 ~9 GB) | ✅ 1 GPU | ✅ 1 GPU | ✅ 1 GPU | ⚠ validate (hybrid-attn VLM on Neuron unconfirmed) |
| **Gemma-4-26B-A4B** (4-bit ~14.4 GB) | ✅ 1 GPU (~7 GB KV) | ✅ 1 GPU (~7 GB KV) | ✅ 1 GPU (roomy) | ⚠ validate (Neuron Gemma-4 support unconfirmed) |

Every GPU cell is **single-GPU** — no tensor parallelism needed for the baseline. That is
the core reason Gemma-4-**26B-A4B** was chosen over the 31B dense: as an MoE with ~3.8B
active params, its 4-bit footprint fits one 24 GB card with real KV headroom.

### Manifests (one per cell)

```
configs/manifests/
  gpt-oss-20b-g5-mxfp4.yaml      gpt-oss-20b-g6-mxfp4.yaml      gpt-oss-20b-g6e-mxfp4.yaml
  qwen3.5-4b-g5-bf16.yaml        qwen3.5-4b-g6-bf16.yaml        qwen3.5-4b-g6e-bf16.yaml
  gemma-4-26b-a4b-g5-w4a16.yaml  gemma-4-26b-a4b-g6-w4a16.yaml  gemma-4-26b-a4b-g6e-w4a16.yaml
  gemma-4-26b-a4b-g6e-fp8.yaml   (optional higher-precision variant, single L40S)
  qwen-2.5-0.5b-baseline.yaml    (smoke)
```

## Precision policy (comparability)

Precision is held **constant per model across hardware** so the *hardware* comparison is
apples-to-apples for a given model:

- gpt-oss-20b → **MXFP4** (the only format it ships in).
- Qwen3.5-4B → **bf16** (fits every card; nothing to gain from quantizing a 4B).
- Gemma-4-26B-A4B → **4-bit W4A16** (the format that fits a single 24 GB card).

Cross-*model* comparisons therefore compare each model **at its deployable precision**, not
at a single shared precision — which is the realistic production question ("what does each
model cost/perform like when actually served on this box").

## Optional higher-precision point (Gemma)

`gemma-4-26b-a4b-g6e-fp8.yaml` runs the MoE at **fp8 (~25 GB) on a single L40S** — a
quality ceiling that still needs **no tensor parallelism**. It does **not** fit 24 GB
(g5/g6). To run it, download an fp8 checkpoint and override the model dir's folder:

```bash
MODEL_FOLDER=Gemma-4-26B-A4B-it-fp8 SERVED_NAME=Gemma-4-26B-A4B-it \
  bash vllm/models/gemma-4-26b-a4b-it/deploy.sh --hw g6e.2xlarge
```

## Why not just run the 31B dense on 12xlarge (4× L40S)?

It's a valid *quality-ceiling* variant but a poor *baseline*:

1. **No NVLink on L40S/A10G/L4.** The 4 GPUs communicate over PCIe. Tensor parallelism
   does an all-reduce every layer, so TP=4 typically yields ~2–2.5× throughput, not 4× →
   **worse $/1M-tokens**, the exact SoW metric.
2. **Breaks cost/latency parity** — the other two models run on 1 GPU; comparing against a
   4-GPU node mixes hardware classes.
3. **Unnecessary for capacity** once the MoE is chosen (fits one card at 4-bit).

If Observe.AI specifically wants a bf16 31B-dense quality point, add a
`gemma-4-31b-*-bf16-tp4` cell with `--hw g6e.12xlarge --tp 4` (the deploy scripts already
support `--tp`). Keep it flagged as a separate class in results.

## trn / Neuron (Flow B) — status

Trainium is a **separate flow** (`vllm/models/neuron/`, scaffolded): it adds a **compile
stage** (Neuron SDK / NEFF) before serve, and for some architectures a weight-conversion
step. Benchmark / observability / teardown are shared with the GPU flow.

**Support is the gating risk.** As of the pinned toolchain, Neuron/vLLM support is solid for
Llama-class; `optimum-neuron` added Gemma 3. For our three:

| Model | Neuron status | Action |
|---|---|---|
| gpt-oss-20b | MoE + MXFP4 on Neuron unconfirmed | validate compile; likely needs conversion or defer |
| Qwen3.5-4B | Hybrid-attention VLM unconfirmed | validate; likely defer until supported |
| Gemma-4-26B-A4B | Gemma-4 on Neuron unconfirmed (Gemma-3 exists) | validate compile first |

Do **not** promise trn numbers until each model compiles and serves under vLLM-Neuron. The
scaffold documents the path and fails loudly rather than silently producing nothing.

## How to run a cell

```bash
# 1. onboard (once per model)
bash model-download/<model>/download.sh
# 2. deploy on a chosen GPU family
bash vllm/models/<model>/deploy.sh --hw g6e.2xlarge      # or g5.2xlarge / g6.2xlarge
# 3. benchmark (renders the in-cluster Job from the matching manifest)
envsubst < inference/benchmark-job.yaml | kubectl apply -f -
# 4. teardown (mandatory — releases the GPU node)
bash vllm/models/<model>/stop.sh
```

Cost figures in the manifests (`cost.instance_hourly_usd`) are **approximate us-east-2
on-demand** and must be verified against current pricing before a run.
