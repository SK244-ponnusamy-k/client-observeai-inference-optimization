# Qwen3.5-9B on Trainium via NeuronX Distributed Inference (NxDI)

Serves **Qwen/Qwen3.5-9B** — a dense hybrid **DeltaNet + GQA** text decoder — on
AWS Trainium2, behind an OpenAI-compatible HTTP shim on `:8000`.

## Why this is not a `vllm serve` model

Qwen3.5 is a hybrid architecture: 24 of 32 layers are Gated DeltaNet (linear
recurrent attention) and 8 are standard GQA, with a dense SwiGLU MLP. **vLLM-Neuron
does not support this architecture out of the box** — you cannot `vllm serve` it.

Instead we run the reference **NeuronX Distributed Inference (NxDI)** contrib
modeling code (`src/modeling_qwen35.py` plus custom NKI DeltaNet kernels under
`src/nki_kernels/`) through its Python API, wrapped in a thin FastAPI server
(`server.py`) that exposes the OpenAI REST subset the benchmark/quality harness
uses:

- `GET  /health`
- `GET  /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

Because the HTTP contract is identical to the vLLM path, the benchmark,
observability, and teardown stages are unchanged — only onboard/compile/deploy
differ, per the framework's "shared tail" design.

## Provenance

`src/` is vendored verbatim (Apache-2.0) from the reference contrib model
`neuronx-distributed-inference .../contrib/models/Qwen3.5-9B`. `server.py`,
`compile.py`, and the Kubernetes manifests in this directory are ShellKode
additions that wrap it for the EKS framework.

## Layout

| File | Purpose |
| --- | --- |
| `src/` | Vendored NxDI modeling + NKI DeltaNet kernels (the model) |
| `server.py` | OpenAI-compatible FastAPI shim (serve) |
| `compile.py` | `model.compile()` entrypoint (trace/shard to a Neuron artifact) |
| `compile-job.yaml` | K8s Job: sync weights, run `compile.py`, push artifact to S3 |
| `deployment.yaml` / `deployment-managed-ng.yaml` | Serve Deployment (Karpenter / managed NG) |
| `service.yaml` | ClusterIP on `:8000` |
| `model.env` | All tunables (seq_len, TP, dtype, S3 folders) |
| `deploy.sh` / `stop.sh` | Deploy and teardown |

Code (`src/`, `server.py`, `compile.py`) ships to pods as ConfigMaps, matching
this repo's ConfigMap-for-app-code convention.

## Quick start

```bash
# 1. Download weights HF -> S3 (open model, no token)
bash model-download/qwen3-5-9b-nxdi/download.sh

# 2. First deploy: compile (slow, caches the artifact in S3), then serve
bash vllm/models/qwen3-5-9b-nxdi/deploy.sh --compile --managed-ng --validate

# 3. Subsequent deploys reuse the cached artifact (fast)
bash vllm/models/qwen3-5-9b-nxdi/deploy.sh --managed-ng

# Port-forward and smoke-test
kubectl port-forward svc/oai-infopt-vllm-qwen3-5-9b-nxdi 8151:8000 -n oai-infopt &
curl -s localhost:8150/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-9B","prompt":"The capital of France is","max_tokens":16}'

# Teardown (frees the Trainium node)
bash vllm/models/qwen3-5-9b-nxdi/stop.sh --managed-ng
```

## Instance sizing — trn2.3xlarge vs trn2.48xlarge

This path **targets `trn2.3xlarge`** (1 Trainium2 chip; LNC=2 → 4 logical cores
for TP=4) as the cost-optimized bring-up instance.

> **9B is heavier than 4B** — hidden size 4096 (vs 2560) and **no tied
> embeddings**, so weights + dummy DeltaNet KV take more HBM. It is more likely
> than the 4B to need `trn2.48xlarge`, especially at larger `seq_len`. Start at
> `seq_len=160` on 3xlarge to confirm it fits before pushing context.

> **Reference validation was on `trn2.48xlarge`.** The upstream contrib model was
> validated only on `trn2.48xlarge` at TP=4, BF16, `seq_len=160`, `batch=1`
> (9B: unit 44 passed, integration 9 passed, TTFT ~88 ms, ~50 tok/s). Running on
> a single `trn2.3xlarge` chip is **not** covered by that validation. If the model
> does not fit or the graph fails to compile on 3xlarge at your chosen
> `seq_len`, fall back to `trn2.48xlarge`:
>
> ```bash
> # in model.env  (or pass --hw)
> export NEURON_INSTANCE_TYPE="trn2.48xlarge"
> ```
>
> On 48xlarge the deploy requests 16 neuron devices automatically (see
> `deploy.sh` device-count logic).

## Context length — pushing beyond the validated baseline

The reference is validated only at **`seq_len=160`, `batch_size=1`, bucketing
off**. This path parameterizes both via `model.env`
(`QWEN35_SEQ_LEN`, `QWEN35_MAX_BATCH_SIZE`) and defaults `QWEN35_SEQ_LEN=8192`
to match the AutoQA workload profile.

> **`seq_len` > 160 is UNVALIDATED.** It is baked into the compiled artifact, so
> changing it forces a full recompile. Larger context may:
> - exceed HBM on a single `trn2.3xlarge` chip (the dummy-KV for DeltaNet layers
>   wastes HBM in v1 — see the reference's known limitations),
> - surface DeltaNet NKI kernel edge cases, or
> - require moving to `trn2.48xlarge` and/or a higher TP degree.
>
> Recommended bring-up order: first compile+serve at `QWEN35_SEQ_LEN=160` to
> reproduce the known-good baseline, then raise it incrementally (e.g. 512 →
> 2048 → 8192), recompiling and re-validating each step.

## Known limitations (inherited from the reference)

- DeltaNet weights are replicated across TP ranks in v1.
- DeltaNet layers allocate dummy KV cache through NxDI's normal cache manager
  (HBM overhead).
- Quantization, speculative decoding, and MoE are out of scope for bring-up.
- The shim does not implement streaming (`stream: true` returns 400).

## Future work — wiring into the `oai` CLI and pipeline (deferred)

This model is intentionally driven by the `deploy.sh` / `stop.sh` scripts in this
directory, **not** the `oai` CLI, for now. The `oai` template generator
(`tools/oai/templates/neuron.py`) produces the stock `vllm serve` neuron path,
which cannot express this custom NxDI shim (ConfigMap-mounted `src/`, FastAPI
server, `model.compile()` step). Rather than fork the generator before the model
is proven, we run the scripts directly.

**Once this path is validated end to end on hardware** (compile + serve +
benchmark at the reference baseline, then at the target context length), wire it
into the framework so it shows up alongside every other model:

1. **`oai` model list / status** — teach `tools/oai/catalog.py` to recognize
   `serving.engine: nxdi` and list it; have `oai model validate` check the
   nxdi-specific fields (`neuron.compiled_folder`, presence of `src/`,
   `server.py`, `compile.py`).
2. **`oai deploy` / `oai stop`** — route `engine: nxdi` models to this
   directory's `deploy.sh` / `stop.sh` (or port their logic into a new
   `tools/oai/templates/nxdi.py`) so `oai deploy qwen3-5-9b-nxdi` works. Keep the
   ConfigMap-for-code and compile-to-S3 behavior.
3. **`oai download`** — point `engine: nxdi` at `model-download/<id>-nxdi/`.
4. **Benchmark pipeline** — no change expected: the shim already honors the
   `:8000` OpenAI contract, so `inference/` benchmark/quality jobs and the
   results schema work as-is. Add the two manifests to the benchmark matrix
   (`docs/benchmark-matrix.md`) once baselined.

Until then, treat the `deploy.sh` workflow above as the supported entry point.
