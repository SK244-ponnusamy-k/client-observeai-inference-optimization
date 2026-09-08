# Neuron / Trainium — Flow B (scaffold)

Trainium (`trn`) is a **separate flow** from the GPU path. It adds a **compile stage**
(Neuron SDK → NEFF artifacts) before serving, and for some architectures a
weight-conversion step. The tail of the pipeline — **benchmark → observability → cost →
teardown — is shared unchanged** with the GPU flow (`inference/load-test.py` +
`inference/benchmark-job.yaml`), so results land in the same schema for comparison.

```
HF weights ─▶ [download to S3] ─▶ [COMPILE (Neuron) ─▶ NEFF cache to S3] ─▶ [SERVE vLLM-Neuron] ─▶ [benchmark] ─▶ [teardown]
                                   └── Flow-B-only steps ──┘                 └──────── shared with GPU flow ────────┘
```

## ⚠ Status: PENDING PER-MODEL VALIDATION

This is a **scaffold**. Do not promise trn benchmark numbers until each model is
confirmed to compile **and** serve under vLLM-Neuron on the pinned toolchain.

| Model | Neuron support (as of pinned toolchain) | Action before benchmarking |
|---|---|---|
| gpt-oss-20b | MoE + MXFP4 on Neuron **unconfirmed** | Try compile; MXFP4 likely unsupported → may need bf16/fp8 conversion or **defer** |
| Qwen3.5-4B | Hybrid-attention multimodal VLM **unconfirmed** | Validate architecture support; likely **defer** until added |
| Gemma-4-26B-A4B | Gemma-4 **unconfirmed** (Gemma-3 exists in `optimum-neuron`) | Validate compile first |

Neuron data types are FP16/BF16 (no MXFP4/NVFP4). 4-bit GPU checkpoints are **not**
reusable on Neuron — compile from the bf16 base instead.

## Prerequisites (set in config/config.env)

```bash
# Neuron toolchain image for the COMPILE job (optimum-neuron / neuronx-cc):
export NEURON_COMPILE_IMAGE="<neuron-compile-image>"     # TODO: pin a Neuron SDK image
# vLLM-Neuron serving image (NOT the GPU DLC):
export NEURON_VLLM_IMAGE="<vllm-neuron-image>"           # TODO: pin vLLM-Neuron image
export NEURON_CORES="8"                                   # cores used = tensor-parallel degree
```

## Steps

```bash
# 0. NodePool for Trainium (once)
kubectl apply -f cluster/neuron-nodepool.yaml

# 1. Download bf16 weights to S3 (reuse the GPU download jobs — same weights)
bash model-download/<model>/download.sh

# 2. Compile → NEFF cache in S3   (Flow-B-only)
#    Edit MODEL_HF_ID / instance type in compile-job.yaml, then:
envsubst < vllm/models/neuron/compile-job.yaml | kubectl apply -f -

# 3. Serve on Trainium via vLLM-Neuron
envsubst < vllm/models/neuron/deployment.yaml | kubectl apply -f -

# 4. Benchmark — SHARED GPU-flow harness, just point --endpoint at the neuron service
#    (inference/benchmark-job.yaml, set MODEL_NAME / VLLM_ENDPOINT accordingly)

# 5. Teardown — release the (expensive) Trainium node
kubectl delete -f vllm/models/neuron/deployment.yaml
```

## Key differences vs the GPU flow (why this is not just another --hw)

- **Ahead-of-time compilation**: batch sizes / sequence lengths are compiled into the NEFF.
  Changing `max-num-seqs` or `max-model-len` means recompiling. Cache NEFFs in S3 so a
  benchmark sweep doesn't recompile every run.
- **Fixed buckets**: Neuron serves compiled input/output-length "buckets"; the workload
  profile must map onto them.
- **tensor-parallel = NeuronCores** used, set at compile + serve time (`NEURON_CORES`).
- **Different image + device resource** (`aws.amazon.com/neuron`), different taint/NodePool.
