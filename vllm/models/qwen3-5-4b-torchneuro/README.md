# Qwen3.5-4B on Trainium via TorchNeuron (native PyTorch backend)

Serves **Qwen/Qwen3.5-4B** on AWS Trainium2 using **TorchNeuron** — the native
PyTorch backend for Trainium announced October 20 2026. This is the simplest
possible Neuron path: one device pointer change and a standard `vllm serve`
command.

## Why this replaces the NxDI path

The previous `qwen3-5-4b-nxdi` path required:
- Vendored reference modeling code (`src/modeling_qwen35.py`, NKI DeltaNet kernels)
- A custom FastAPI shim (`server.py`) to expose the OpenAI contract
- ConfigMap-mounted Python packages in every pod
- A separate `compile.py` entrypoint

TorchNeuron eliminates all of it. It registers Trainium as a standard PyTorch
device via the `PrivateUse1` extension point, so the model loads through the
normal Hugging Face API:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B",
    device_map="neuron"   # ← the only change
)
```

vLLM Neuron is a plugin to vLLM that uses the same TorchNeuron backend. The
serving command is a plain `vllm serve` with `--device neuron`:

```bash
vllm serve /model \
  --served-model-name Qwen3.5-4B \
  --device neuron \
  --tensor-parallel-size 4
```

No custom modeling code. No ConfigMaps. No FastAPI shim. Structurally identical
to the `gpt-oss-20b-neuron` path.

## Layout

| File | Purpose |
|---|---|
| `model.env` | All tunables (seq_len, TP, LNC, S3 folders, port) |
| `compile-job.yaml` | K8s Job: pip install torch-neuronx, run optimum-cli export neuron, push NEFFs to S3 |
| `deployment.yaml` | Serve Deployment — Karpenter (instance-family selector) |
| `deployment-managed-ng.yaml` | Serve Deployment — managed node group selector |
| `service.yaml` | ClusterIP on `:8000` |
| `deploy.sh` | Deploy: scale NG, compile (optional), apply manifests |
| `stop.sh` | Teardown: delete resources, scale NG to 0 |

## Quick start

```bash
# 1. Download weights HF → S3 (open model, no token required)
huggingface-cli download Qwen/Qwen3.5-4B --local-dir /tmp/qwen35-4b
aws s3 sync /tmp/qwen35-4b s3://${MODEL_BUCKET}/Qwen3.5-4B/ --region ${AWS_REGION}

# 2. First deploy: compile NEFF artifacts (cached in S3), then serve
bash vllm/models/qwen3-5-4b-torchneuro/deploy.sh --compile --managed-ng --validate

# 3. Subsequent deploys reuse the S3 NEFF cache — no recompile (fast)
bash vllm/models/qwen3-5-4b-torchneuro/deploy.sh --managed-ng

# Port-forward and smoke-test
kubectl port-forward svc/oai-infopt-vllm-qwen3-5-4b-torchneuro 8152:8000 -n oai-infopt &
curl -s localhost:8152/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-4B","prompt":"The capital of France is","max_tokens":16}'

# Teardown (frees the Trainium node, stops billing)
bash vllm/models/qwen3-5-4b-torchneuro/stop.sh --managed-ng
```

## How TorchNeuron works (from the announcement)

TorchNeuron builds directly on PyTorch's accelerator-agnostic extension points:

- **PrivateUse1 device backend** — registers `"neuron"` as a standard device in
  PyTorch's device enumeration. `model.to('neuron')` and `device_map="neuron"`
  both route through the same path.
- **`torch.accelerator` API** — `device_map="auto"` finds the neuron device
  automatically with no configuration. `device_map="neuron"` names it explicitly.
- **`torch.compile` backend** — `torch.compiler.set_default_backend("neuron")`
  sets it globally once; every subsequent `torch.compile(model)` call uses it.
  vLLM Neuron sets this automatically before serving.
- **Execution modes** — eager for research/debug, `torch.compile` for production
  performance (what `vllm serve` uses). The NEFF artifacts produced by the
  compile job are the `torch.compile` output cached on disk.

Code written against `torch.accelerator`, `torch.nn`, `torch.distributed`, and
`torch.profiler` runs on Trainium unmodified.

## Hardware

| Setting | Value |
|---|---|
| Instance | `trn2.3xlarge` (1 Trainium2 chip) |
| Logical NC config (LNC) | `2` (fuse 2 physical NeuronCores → 1 logical) |
| Tensor-parallel degree | `4` (4 logical cores) |
| Sequence length | `8192` (baked into NEFF at compile time) |
| Batch size | `8` |
| Dtype | `bf16` |

LNC and TP degree must be **identical** at compile time and serve time. Changing
either requires a full recompile (delete `s3://${MODEL_BUCKET}/qwen3-5-4b-torchneuro-neff/`
then re-run `deploy.sh --compile`).

## Recompile from scratch

```bash
# Delete the S3 NEFF cache to force a full recompile on next --compile run
aws s3 rm s3://${MODEL_BUCKET}/qwen3-5-4b-torchneuro-neff/ --recursive --region ${AWS_REGION}
bash vllm/models/qwen3-5-4b-torchneuro/deploy.sh --compile --managed-ng
```

## Migration from qwen3-5-4b-nxdi

| Concern | NxDI path (`qwen3-5-4b-nxdi`) | TorchNeuron path (this) |
|---|---|---|
| Model code | Vendored `src/modeling_qwen35.py` + NKI kernels | Standard HF `Qwen/Qwen3.5-4B` |
| Server | Custom FastAPI shim (`server.py`) | `vllm serve --device neuron` |
| Code distribution | ConfigMaps mounted into every pod | None |
| Compile entrypoint | `compile.py` → `model.compile()` (NxDI) | `optimum-cli export neuron` |
| NEFF cache in S3 | `qwen3-5-4b-nxdi-traced/` | `qwen3-5-4b-torchneuro-neff/` |
| PyTorch device flag | `NEURON_RT_NUM_CORES` + custom env | `--device neuron` |

The HTTP contract is identical (`/health`, `/v1/models`, `/v1/completions`,
`/v1/chat/completions` on `:8000`), so the benchmark and quality-eval pipeline
are unchanged.

## Known considerations

- **Architecture support**: TorchNeuron's broad `aten` operator coverage is
  designed to handle text, multimodal, and SSM architectures. Qwen3.5's hybrid
  DeltaNet+GQA layers should be covered through the standard operator path,
  unlike the NxDI path which required hand-written NKI DeltaNet kernels.
- **Context length**: `seq_len=8192` is the default. Changing it requires a
  recompile. On `trn2.3xlarge` (single chip) larger values may hit HBM limits;
  fall back to `trn2.48xlarge` if needed (`--hw trn2.48xlarge`).
- **First boot**: If the NEFF cache is not present in `/tmp/neuronx_cache`, vLLM
  Neuron will compile on pod start (slow — up to 90 min). Always run
  `--compile` first to pre-populate S3, then the init container syncs it on
  subsequent starts.
