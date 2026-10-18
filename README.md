# Project: ObserveAI Inference Optimization — EKS + vLLM

| Field         | Value                                          |
|---------------|------------------------------------------------|
| Owner         | genai-platform@shellkode                       |
| Business Unit | AI / Data / Cloud                              |
| Status        | Active                                         |
| Environment   | Sandbox                                        |
| Created On    | 2026-09-01                                     |

## Description

Model-agnostic benchmarking framework that deploys open-source LLMs on Amazon EKS with GPU auto-scaling, streams weights from S3 via the Run:ai model streamer, benchmarks under two workload profiles (realtime and batch), records normalized cost and performance, and tears down GPU resources after every run.

Serving engine: **AWS vLLM Deep Learning Container (DLC)** — OpenAI-compatible API on port 8000.  
IaC: **AWS CDK (Python)** for cluster-adjacent resources; Kubernetes manifests versioned in-repo.  
Security: IAM least-privilege via EKS Pod Identity; secrets in AWS Secrets Manager; no public NLB; Checkov-hardened pod specs.

## Onboard a model the easy way — the `oai` CLI

You do **not** need to hand-write Kubernetes manifests, download jobs, or deploy
scripts to add a model. Describe the model in one small catalog file and the
`oai` tool generates everything (with the same security hardening) and runs it.

```bash
oai model new              # guided questions -> writes catalog/models/<id>.yaml
oai model generate <id>    # build all deployment files from that one entry
oai download <id>          # HuggingFace weights -> S3
oai deploy <id>            # run it — an available instance is auto-selected
oai deploy <id> --benchmark  # ...or deploy and benchmark in one step
oai status                 # what's running + which nodes are billed
oai stop <id>              # stop and free the accelerator node
```

One-time cluster + monitoring setup is also driven by `oai` (every command
reuses what already exists instead of recreating it):

```bash
oai cluster bootstrap                  # EKS cluster + S3 + IAM + node pools
oai cluster setup-monitoring           # Prometheus + Grafana + DCGM GPU metrics
oai cluster status                     # cluster + monitoring + node-group health
oai cluster teardown                   # delete the cluster (keeps S3 + IAM)
```

For **AWS Trainium** models there is a one-time node-group setup (idempotent —
reuses an existing group, never recreates it), then the normal deploy flow:

```bash
oai cluster setup-neuron               # once: create or reuse the trn2 node group
oai deploy <id> --compile --managed-ng # first deploy (compiles); later: drop --compile
```

Highlights:

- **One file per model** (`catalog/models/<id>.yaml`) — plain fields, fully
  commented. Works for **GPU** and **AWS Trainium/Neuron** models.
- **Dynamic instance selection**: `oai deploy` picks the cheapest available
  instance from your preferred list, checks region availability + quota, and
  falls back automatically. If nothing is available it tells you exactly what to
  do (quota, region, node-group fallback, or wait) — never a silent failure.
- **Customizable, optional auto-benchmarking**: set `benchmark.auto: true` (or
  pass `--benchmark`) to benchmark right after deploy; choose profiles and
  datasets per model.
- **Safe by default**: generated files keep non-root, read-only rootfs, dropped
  capabilities, least-privilege IAM, pinned images, and ClusterIP-only services.
  The generator never overwrites files it didn't create.

Full walkthrough for non-technical users: [`docs/ONBOARDING.md`](docs/ONBOARDING.md).
Catalog field reference: [`catalog/SCHEMA.md`](catalog/SCHEMA.md).

Launcher: `.\oai.ps1 <command>` on Windows, `./oai <command>` on Bash/WSL/macOS.

The sections below document the underlying manual workflow the `oai` CLI
automates — useful for understanding what it generates, or for advanced tuning.

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
└── inference/
    ├── load-test.py                # Profile-driven benchmark harness (async, TTFT/ITL/cost)
    ├── benchmark-job.yaml          # In-cluster benchmark Job (hardened, separate CPU node)
    └── quick-test.sh               # Single-request smoke test
```

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

### Step 6 — Deploy vLLM

```bash
bash vllm/models/qwen-2.5-0.5b/deploy.sh
# or
bash vllm/models/gpt-oss-20b/deploy.sh
```

### Step 7 — Run benchmark

**Local (via port-forward):**

```bash
kubectl port-forward -n oai-infopt svc/oai-infopt-vllm-qwen-0-5b 8080:8000 &

pip install openai pyyaml aiohttp
python inference/load-test.py \
  --manifest configs/manifests/qwen-2.5-0.5b-baseline.yaml \
  --profile  configs/workload_profiles/realtime_v1.yaml \
  --endpoint http://localhost:8080 \
  --output   results/
```

**In-cluster (preferred — no port-forward latency overhead):**

```bash
kubectl apply -f inference/benchmark-job.yaml -n oai-infopt
kubectl logs -n oai-infopt job/oai-infopt-benchmark -f
```

Results are written as JSONL to `results/<run_id>.jsonl` and uploaded to `s3://oai-infopt-results-<account>-<region>/results/`.

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

| Model | S3 Prefix | Size | GPU | VRAM |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | `models/Qwen2.5-0.5B-Instruct/` | ~950 MB | g6e.xlarge | ~2 GB |
| openai/gpt-oss-20b | `models/gpt-oss-20b/` | ~13 GB | g6e.2xlarge | ~16 GB |

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
