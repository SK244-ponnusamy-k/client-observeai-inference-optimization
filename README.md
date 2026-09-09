# Project: ObserveAI Inference Optimization — EKS + vLLM

| Field         | Value                                          |
|---------------|------------------------------------------------|
| Owner         | genai-platform@shellkode                       |
| Business Unit | AI / Data / Cloud                              |
| Status        | Active                                         |
| Environment   | Sandbox                                        |
| Created On    | 2026-09-01                                     |

## Description

Model- and hardware-agnostic benchmarking framework that deploys open-source LLMs on Amazon EKS, benchmarks them across a **matrix of hardware** (NVIDIA GPU — g5/g6/g6e — and AWS Trainium — trn2/trn3), and records **both** normalized cost/performance **and** model quality on the customer's AutoQA rubric, tearing down accelerator resources after every run.

The framework is built so the **Observe.AI team can run any (model × hardware) cell on demand** — nothing is hard-coded per model; everything comes from a manifest + a `--hw` flag.

Two independent measurement stages:
- **Performance** (`inference/run-benchmark.sh` → `load-test.py`, a `vllm bench serve` orchestrator): TTFT / ITL / TPOT / throughput / cost per (profile × concurrency). Uses a synthetic dataset — content-independent.
- **Quality** (`inference/run-quality.sh` → `quality-eval.py`): accuracy / precision / recall / F1 on the labelled AutoQA dataset. Hardware-independent → run once per (model, quantization).

Serving engine: **AWS vLLM Deep Learning Container (DLC)** on GPU, **vLLM Neuron plugin** on Trainium — OpenAI-compatible API on port 8000 in both cases.  
IaC: **AWS CDK (Python)** for cluster-adjacent resources; Kubernetes manifests versioned in-repo.  
Security: IAM least-privilege via EKS Pod Identity; secrets in AWS Secrets Manager; no public NLB; Checkov-hardened pod specs.

## Team Members

- @shellkode-genai (GitHub handle — update when assigned)

## Repository Layout

```
llm-inference-framework/
├── config/
│   └── config.env                  # Single source of truth — source before running
│
├── cluster/
│   ├── bootstrap.sh                # Full cluster + S3 + S3 Gateway Endpoint setup
│   ├── teardown.sh                 # Deletes cluster; keeps S3 + IAM roles
│   ├── setup-pod-identity.sh       # 3 least-privilege IAM roles + Pod Identity bindings
│   ├── gpu-nodepool.yaml           # Karpenter NodePool: gpu-inf (g6e/g6/g5, gen 5-6)
│   ├── model-storage-sa.yaml       # ServiceAccounts: download / serving / benchmark
│   └── storage-class.yaml          # gp3 EBS StorageClass (encrypted, WaitForFirstConsumer)
│
├── k8s/
│   ├── download/
│   │   └── hf-token-external-secret.yaml  # ESO ExternalSecret — HF token from Secrets Manager
│   ├── serving/
│   │   └── vllm-configmap.yaml     # Non-secret vLLM runtime config (region, logging)
│   └── network-policy.yaml         # Default-deny + allow-list NetworkPolicies
│
├── model-download/
│   ├── qwen-2.5-0.5b/
│   │   ├── download.sh             # HF → S3 (no token required)
│   │   └── job.yaml                # Hardened Kubernetes Job (non-root, readOnlyRootFS)
│   └── gpt-oss-20b/
│       ├── download.sh             # HF → S3 (token via Secrets Manager / ESO)
│       └── job.yaml                # Hardened Kubernetes Job
│
├── vllm/
│   ├── service/
│   │   ├── service-private.yaml    # ClusterIP — in-cluster access + port-forward
│   │   ├── service-ingress.yaml    # HTTPS ALB Ingress (replaces public NLB)
│   │   └── service-public.yaml     # REMOVED — public unauthenticated NLB prohibited
│   ├── models/
│   │   ├── qwen-2.5-0.5b/
│   │   │   ├── deploy.sh / stop.sh
│   │   │   ├── deployment.yaml     # Hardened Deployment (non-root, readOnlyRootFS, drop ALL)
│   │   │   ├── service.yaml        # ClusterIP, namespace oai-infopt
│   │   │   └── pvc.yaml            # Metadata cache (tokenizer only; weights stream from S3)
│   │   └── gpt-oss-20b/
│   │       ├── deploy.sh / stop.sh
│   │       ├── deployment.yaml
│   │       ├── service.yaml
│   │       └── pvc.yaml
│   └── stop.sh                     # Stop all running vLLM deployments
│
├── configs/
│   ├── manifests/
│   │   ├── qwen-2.5-0.5b-baseline.yaml   # Run manifest — drives every benchmark
│   │   └── gpt-oss-20b-baseline.yaml
│   └── workload_profiles/
│       ├── realtime_v1.yaml        # Frozen: low concurrency, TTFT-bound
│       └── batch_v1.yaml           # Frozen: high concurrency, throughput-bound
│
├── inference/
│   ├── load-test.py                # PERF orchestrator — shells out to `vllm bench serve`
│   ├── run-benchmark.sh            # One-command perf run (--model, --hw, --profile)
│   ├── quality-eval.py             # QUALITY evaluator — accuracy / precision / recall / F1
│   ├── run-quality.sh              # One-command quality run (--stage-dataset, --model, --hw, --quant)
│   ├── benchmark-job.yaml          # In-cluster benchmark Job (hardened, runs from vLLM DLC)
│   └── quick-test.sh               # Single-request smoke test
│
├── vllm/models/neuron/            # Trainium (Flow B) — single-instance gpt-oss-20b (non-DI)
│   ├── deployment.yaml / pvc.yaml / service.yaml
│   ├── model.env / deploy.sh / stop.sh
│   └── README.md                   # Trn specifics, caveats, open decisions
│
├── cluster/neuron-nodepool.yaml   # Karpenter NodePool: neuron-inf (trn2/trn3)
│
├── configs/
│   ├── manifests/                  # per-cell run manifests: <model>-<hw>-<quant>.yaml
│   │   ├── gpt-oss-20b-{g5,g6,g6e}-mxfp4.yaml
│   │   ├── gpt-oss-20b-{trn2-bf16,trn3-mxfp4}.yaml
│   │   ├── qwen3.5-4b-{g5,g6,g6e}-bf16.yaml
│   │   ├── gemma-4-26b-a4b-{g5,g6,g6e}-w4a16.yaml (+ g6e-fp8)
│   │   └── qwen-2.5-0.5b-baseline.yaml            # smoke test
│   ├── workload_profiles/          # frozen: realtime_v1.yaml, batch_v1.yaml
│   └── quality/
│       └── autoqa_v1.yaml          # frozen quality-eval config (labels, decoding, fields)
│
└── docs/
    └── benchmark-matrix.md         # the models × hardware grid + methodology
```

> Full model dirs also exist under `vllm/models/` for `gpt-oss-20b`, `qwen3.5-4b`,
> `gemma-4-26b-a4b-it`, and `qwen-2.5-0.5b` (smoke). All GPU deployments are
> hardware-templated — the same file serves g5/g6/g6e via `deploy.sh --hw`.

## How To Run

### Prerequisites

- AWS CLI v2 configured with IAM Roles Anywhere or OIDC (no static keys)
- `eksctl`, `kubectl`, `helm` installed
- Python 3.11+

### Step 1 — Configure

Edit `config/config.env`:

```bash
export AWS_REGION="us-east-2"
export VLLM_IMAGE="public.ecr.aws/deep-learning-containers/vllm:0.26.0-gpu-py312-cu130-ubuntu22.04-ec2-v1.0-soci"
# Bucket names are auto-derived: oai-infopt-models-<account>-<region>
```

### Step 2 — Bootstrap cluster

```bash
cd llm-inference-framework
bash cluster/bootstrap.sh
```

Creates: EKS cluster (`oai-infopt-eks`), S3 buckets (encrypted + versioned), S3 Gateway Endpoint, gp3 StorageClass, GPU NodePool (`gpu-inf`), 3 least-privilege IAM roles, Pod Identity bindings, `oai-infopt` namespace.

Takes ~15 minutes. Idempotent.

### Step 3 — Store HF token in Secrets Manager (gated models only)

```bash
aws secretsmanager create-secret \
  --name 'oai-infopt/hf-token' \
  --description 'HuggingFace API token for gated model download' \
  --secret-string '{"token":"hf_YOUR_TOKEN_HERE"}' \
  --region us-east-2
```

Then install External Secrets Operator and apply the ExternalSecret:

```bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  --namespace external-secrets --create-namespace
kubectl apply -f k8s/download/hf-token-external-secret.yaml
```

### Step 4 — Apply NetworkPolicies and ConfigMap

```bash
kubectl apply -f k8s/network-policy.yaml
kubectl apply -f k8s/serving/vllm-configmap.yaml
```

### Step 5 — Download models to S3

```bash
# Qwen 2.5 0.5B (no token)
bash model-download/qwen-2.5-0.5b/download.sh

# GPT-OSS-20B (requires hf-token ExternalSecret synced)
bash model-download/gpt-oss-20b/download.sh
```

Each script checks S3 first and skips if already present.

### Step 6 — Deploy vLLM (pick the hardware cell)

```bash
bash vllm/models/gpt-oss-20b/deploy.sh --hw g6e.2xlarge     # g5.2xlarge | g6.2xlarge | g6e.2xlarge
# smoke test:
bash vllm/models/qwen-2.5-0.5b/deploy.sh --hw g6e.xlarge
# Trainium (single-instance): see the Trainium section
bash vllm/models/neuron/deploy.sh --hw trn3.48xlarge
```

### Step 7 — Run benchmarks (performance + quality)

Performance (one command per profile; picks the `<model>-<hw>-<quant>` manifest + service):

```bash
bash inference/run-benchmark.sh --model gpt-oss-20b --hw g6e --profile realtime
bash inference/run-benchmark.sh --model gpt-oss-20b --hw g6e --profile batch
```

Quality (accuracy / F1 — once per model+quant; stage the dataset first):

```bash
bash inference/run-quality.sh --stage-dataset "/path/to/synthetic_autoqa_transcripts.csv"
bash inference/run-quality.sh --model gpt-oss-20b --hw g6e --quant mxfp4
```

Results are written as JSONL and uploaded to `s3://<RESULTS_BUCKET>/results/…` (perf)
and `s3://<RESULTS_BUCKET>/quality/…` (quality). See the **Benchmark matrix** and
**Quality evaluation** sections for the full flow.

### Step 8 — Teardown

**Stop model only (keep cluster):**

```bash
bash vllm/models/qwen-2.5-0.5b/stop.sh
```

**Stop all models + cluster:**

```bash
bash vllm/stop.sh
bash cluster/teardown.sh
```

S3 model weights and IAM roles are retained.

## Models Reference

| Model | HF ID | S3 folder | Precision | ~Size |
|---|---|---|---|---|
| gpt-oss-20b | `openai/gpt-oss-20b` | `gpt-oss-20b/` | MXFP4 (BF16 on Trn2) | ~13 GB |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | `Qwen3.5-4B/` | BF16 | ~9 GB |
| Gemma-4-26B-A4B | `google/gemma-4-26B-A4B-it` | `Gemma-4-26B-A4B-it-w4a16/` | 4-bit W4A16 | ~14 GB |
| Qwen2.5-0.5B (smoke) | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen2.5-0.5B-Instruct/` | BF16 | ~1 GB |

Qwen3.5-4B and Gemma-4 are multimodal — we benchmark the **text path** only.

## Benchmark matrix & hardware selection

Every run is a **(model × hardware × quantization)** cell, driven by a manifest and
a `--hw` flag — no per-model code. See `docs/benchmark-matrix.md` for the full grid
and methodology.

| Model | Precision | g5 / g6 (24 GB) | g6e (48 GB) | Trainium |
|---|---|---|---|---|
| gpt-oss-20b | MXFP4 | ✅ 1 GPU | ✅ 1 GPU | Trn3 = MXFP4 (like-for-like) · Trn2 = BF16 |
| Qwen3.5-4B | BF16 | ✅ 1 GPU | ✅ 1 GPU | validate |
| Gemma-4-26B-A4B | 4-bit W4A16 | ✅ 1 GPU | ✅ 1 GPU (+ fp8 variant) | validate |

```bash
# deploy a model on a chosen GPU family, then benchmark that exact cell
bash vllm/models/gpt-oss-20b/deploy.sh --hw g6.2xlarge
bash inference/run-benchmark.sh --model gpt-oss-20b --hw g6 --profile realtime
bash inference/run-benchmark.sh --model gpt-oss-20b --hw g6 --profile batch
```

`--hw` picks the instance type (and, for the manifest, the `<model>-<hw>-<quant>` cell).
Realtime = latency-bound (conc 1/2/4); batch = throughput-bound (conc 16→128).

## Quality evaluation (accuracy / F1)

Separate from performance. Scores the customer AutoQA rubric (binary Yes/No) against
a **deployed** endpoint. Accuracy is hardware-independent → run once per (model,
quantization).

```bash
# one-time: stage the labelled dataset to S3 (kept out of git; SSE-encrypted)
bash inference/run-quality.sh --stage-dataset "/path/to/synthetic_autoqa_transcripts.csv"

# evaluate a deployed model
bash inference/run-quality.sh --model gpt-oss-20b --hw g6e --quant mxfp4
```

Reports accuracy, precision, recall, F1, macro-F1, per-question F1, confusion matrix.
Config is frozen in `configs/quality/autoqa_v1.yaml`. For reasoning models (gpt-oss)
use `mode: free_parse` with an adequate `reasoning_max_tokens` (guided_choice with a
tiny budget truncates them mid-thought → 0 score).

## Trainium (Neuron) — single instance (Flow B)

gpt-oss-20b on Trainium via the vLLM Neuron plugin, single instance (TP8). Same OpenAI
API → the harnesses above run unchanged. **Disaggregated inference (DI) is out of scope**
(phase-2 peak-throughput topology). Full details + caveats: `vllm/models/neuron/README.md`.

```bash
kubectl apply -f cluster/neuron-nodepool.yaml          # + install the Neuron device plugin
bash vllm/models/neuron/deploy.sh --hw trn3.48xlarge   # first launch JIT-compiles NEFFs (20–60 min; cached)
bash inference/run-benchmark.sh --model gpt-oss-20b --hw trn3 --profile realtime
bash vllm/models/neuron/stop.sh
```

Caveats: the recipe compiles `num_seqs_buckets=[4]` (realtime is representative; batch
queues), and a 20B on a whole Trainium node is over-provisioned (report Trn perf/quality
confidently, annotate cost).

## ⚠ Placeholders to fill before running

These are the values only your AWS/EKS environment can provide. They're marked in-code
with `FILL` / `TODO` / `PLACEHOLDER` / `VERIFY`.

| # | Placeholder | Where | How to fill |
|---|---|---|---|
| 1 | `VLLM_IMAGE` | `config/config.env` | Pinned GPU vLLM DLC tag+digest (already set; re-pin on image bumps). |
| 2 | `NEURON_VLLM_IMAGE` | `config/config.env` | **Blank — required for Trainium.** AWS Neuron vLLM-plugin image (SDK 2.31). Deploy fails fast until set. |
| 3 | `NODE_INSTANCE_TYPE` | `vllm/models/neuron/model.env` | Exact bookable Trn3 type: `aws ec2 describe-instance-type-offerings --region us-east-2 --filters Name=instance-type,Values=trn3*` |
| 4 | `NEURON_DEVICE_COUNT` | `vllm/models/neuron/model.env` | TP8 = 8 cores. Trn3: 1 chip = 8 cores → `1`. Verify resource name/count: `kubectl describe node <trn-node> \| grep -i neuron` (chips = `aws.amazon.com/neuron`, cores = `…/neuroncore`). |
| 5 | `cost.instance_hourly_usd` | every `configs/manifests/*.yaml` | Real on-demand $/hr per instance type (GPU cells have approx values marked VERIFY; Trn cells are placeholders). Cost/1M is meaningless until real. |
| 6 | HF access | `hf-token` secret (Step 3) | Token + accepted license for gated models (gpt-oss). |
| 7 | Gemma 4-bit checkpoint | `model-download/gemma-4-26b-a4b-it/download.sh` | `QUANT_HF_ID` = a vLLM-loadable W4A16 checkpoint (or quantize offline). |
| 8 | `Owner` / `CostCenter` tags | `config/config.env`, manifests | Replace `TBD` with the values Observe.AI provides (mandatory tags). |
| 9 | Prefix caching | GPU vs Neuron deployments | Decide on (recipe) vs off (cross-hardware consistency). |

## Security Posture

| Control | Implementation |
|---|---|
| No static AWS keys | IAM Roles Anywhere (local) / OIDC (CI) |
| Secrets | AWS Secrets Manager + External Secrets Operator |
| IAM least privilege | 3 separate roles: download / serving (read-only) / benchmark |
| Pod security | non-root, readOnlyRootFilesystem, drop ALL caps, seccomp RuntimeDefault |
| Network | Default-deny NetworkPolicy; ClusterIP only; ALB Ingress for external |
| S3 | Encrypted at rest, versioned, public access blocked |
| EBS | gp3 encrypted StorageClass |

## Links

- Plane project: TBD
- ECR Public Gallery (vLLM DLC): https://gallery.ecr.aws/deep-learning-containers/vllm
- HF token (gated models): https://huggingface.co/settings/tokens
