# Project: ObserveAI Inference Optimization — EKS + vLLM

| Field         | Value                                          |
|---------------|------------------------------------------------|
| Owner         | genai-platform@shellkode                       |
| Business Unit | AI / Data / Cloud                              |
| Status        | Active                                         |
| Environment   | Sandbox                                        |
| Created On    | 2026-09-01                                     |

## Description

Model-agnostic benchmarking framework that deploys open-source LLMs on Amazon EKS across **GPU (G5/G6/G6e/G7e) and AWS Trainium/Inferentia (Neuron)** accelerators, streams weights from S3, benchmarks under two workload profiles (realtime and batch), records normalized cost and performance, and scales accelerator nodes to zero after every run.

Serving engine: **AWS vLLM Deep Learning Container (DLC)** on GPU and the **vLLM Neuron / NxD Inference DLCs** on Trainium — OpenAI-compatible API on port 8000.
Driven entirely by config — no per-model or per-hardware hard-coding in framework code. One small catalog file per model generates every manifest, download job, and deploy script.
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

For **AWS Trainium / Neuron** models there is a one-time node-group setup
(idempotent — reuses an existing trn2 group, never recreates it), then the
normal deploy flow:

```bash
oai cluster setup-neuron               # once: create or reuse the trn2 node group
oai deploy <id> --compile --managed-ng # first deploy (compiles NEFF); later: drop --compile
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
Hardware feasibility + per-cell run commands: [`docs/benchmark-matrix.md`](docs/benchmark-matrix.md).

Launcher: `.\oai.ps1 <command>` on Windows, `./oai <command>` on Bash/WSL/macOS.

The sections below document the underlying workflow the `oai` CLI
automates — useful for understanding what it generates, or for advanced tuning.

## Team Members

- @shellkode-genai (GitHub handle — update when assigned)

## Repository Layout

```
client-observeai-inference-optimization/
├── oai / oai.ps1                   # CLI entrypoints (Bash/WSL/macOS + Windows)
├── pyproject.toml                  # Python package for the `oai` tool
├── setup-cluster.sh                # One-shot cluster bootstrap (GPU + Neuron)
│
├── config/
│   ├── config.env                  # Single source of truth — source before running
│   └── .env                        # Local overrides + secrets (gitignored)
│
├── catalog/                        # ★ One YAML per model drives everything
│   ├── TEMPLATE.yaml               # Copy-paste starting point
│   ├── SCHEMA.md                   # Field reference
│   └── models/
│       ├── gpt-oss-20b.yaml                # GPU (MXFP4)
│       ├── gpt-oss-20b-neuron.yaml         # Trainium / Neuron (bf16)
│       ├── qwen3-5-4b.yaml                 # GPU
│       ├── qwen3-5-4b-neuron.yaml          # Neuron (vLLM backend)
│       ├── qwen3-5-4b-nxdi.yaml            # Neuron (NxD Inference backend)
│       ├── qwen3-5-9b-tp1.yaml / -tp4.yaml # GPU tensor-parallel variants
│       ├── qwen3-5-9b-nxdi.yaml            # Neuron NxDI
│       ├── qwen3-vl-8b-neuron.yaml         # Neuron multimodal
│       ├── gemma-4-31b.yaml                # GPU baseline
│       ├── gemma-4-31b-opt2.yaml           # KV/batch-tuning A/B candidate
│       ├── gemma-4-31b-fp8-tp1.yaml / -tp4.yaml  # FP8 variants
│       ├── gemma-4-e4b-it.yaml
│       ├── glm-4-1v-9b-thinking-tp1.yaml / -tp4.yaml
│       ├── deepseek-r1-distill-qwen-7b.yaml
│       └── phi-4-mini-instruct.yaml
│
├── cluster/
│   ├── bootstrap.sh                # EKS + S3 + S3 Gateway Endpoint + IAM + node pools
│   ├── teardown.sh                 # Deletes cluster; keeps S3 + IAM roles
│   ├── setup-pod-identity.sh       # 3 least-privilege IAM roles + Pod Identity bindings
│   ├── setup-trn2-nodegroup.sh     # Idempotent trn2 managed node group (reused, not recreated)
│   ├── gpu-nodepool.yaml           # Karpenter NodePool: g5/g6/g6e
│   ├── gpu-g7-nodepool.yaml        # Karpenter NodePool: G7/G7e (Blackwell, custom AMI)
│   ├── neuron-nodepool.yaml        # Karpenter NodePool: Trainium/Inferentia
│   ├── trn2_nodegroup*.yaml        # trn2 managed node group variants (CB / spot / multi-AZ)
│   ├── ami/                        # Packer builds for G7 + Neuron EKS AMIs
│   ├── model-storage-sa.yaml       # ServiceAccounts: download / serving / benchmark
│   └── storage-class.yaml          # gp3 EBS StorageClass (encrypted, WaitForFirstConsumer)
│
├── k8s/
│   ├── download/                   # ESO ExternalSecret — HF token from Secrets Manager
│   ├── serving/                    # Non-secret vLLM runtime config (region, logging)
│   └── network-policy.yaml         # Default-deny + allow-list NetworkPolicies
│
├── model-download/                 # One hardened HF → S3 Job per model (generated)
│
├── vllm/
│   ├── service/                    # service-private (ClusterIP) / service-ingress (ALB)
│   ├── lib/                        # gpu-terminate.sh and shared helpers
│   ├── stop.sh                     # Stop all running vLLM deployments
│   └── models/                     # Generated per-model deploy/stop + manifests
│       ├── gpt-oss-20b/ ... gpt-oss-20b-neuron/
│       ├── qwen3.5-4b/ ... qwen3-5-*-nxdi/ ... qwen3-vl-8b-neuron/
│       ├── gemma-4-31b*/ ... glm-*/
│       └── neuron/                 # Shared Neuron compile-job + serve scaffold
│
├── neuron/
│   └── autoport/                   # ★ AutoPort kit — port new models to vLLM Neuron on Trainium
│       ├── launch-box.sh           # Launch a trn2.48xlarge AutoPort box in a Capacity Block (SSM-only)
│       ├── setup-box.sh            # Prepare the box: vllm-neuron workspace, weights, Claude Code + skills
│       ├── backup-workspace.sh     # Cron-snapshot the port to S3 (box disk dies with the block)
│       └── <model>/PROMPT.md       # AutoPort prompt with architecture facts + acceptance bar
│
├── configs/
│   ├── manifests/                  # Per-cell run manifests: <model>-<hw>[-<quant>].yaml
│   └── workload_profiles/
│       ├── realtime_v1.yaml        # Frozen: low concurrency, TTFT-bound
│       └── batch_v1.yaml           # Frozen: high concurrency, throughput-bound
│
├── inference/
│   ├── load-test.py                # bench-serve orchestrator (TTFT/ITL/cost + scoped telemetry)
│   ├── run-benchmark.sh            # One-command in-cluster benchmark runner
│   ├── benchmark-job.yaml          # In-cluster benchmark Job (hardened, separate CPU node)
│   └── quick-test.sh               # Single-request smoke test
│
├── monitoring/                     # Prometheus + Grafana + DCGM GPU metrics
├── report-gen/                     # Result rollups / report generation
├── tests/                          # unittest regression tests (telemetry, etc.)
├── tools/                          # `oai` CLI implementation (oai/wizard.py, generators)
└── docs/
    ├── ONBOARDING.md               # Non-technical walkthrough
    ├── benchmark-matrix.md         # Feasibility grid + per-cell run commands
    └── gpt-oss-neuron-optimization.md
```

## How To Run

### Prerequisites

- AWS CLI v2 configured with IAM Roles Anywhere or OIDC (no static keys)
- `eksctl`, `kubectl`, `helm` installed
- Python 3.11+

### Step 1 — Configure

Edit `config/config.env` (or set overrides in the gitignored `config/.env`). Key defaults:

```bash
export AWS_REGION="ap-southeast-4"               # EKS region
export S3_BUCKET_REGION="${S3_BUCKET_REGION:-$AWS_REGION}"  # buckets may live in another region
export CLUSTER_NAME="ai-eks-cluster-trn2"
export MODEL_BUCKET="shellkode-ai-models"        # existing weights bucket (reused)
export RESULTS_BUCKET="shellkode-ai-results"     # benchmark output (created fresh)
export VLLM_IMAGE="public.ecr.aws/deep-learning-containers/vllm:0.26.0-gpu-py312-cu130-ubuntu22.04-ec2-v1.0-soci@sha256:..."
# Trainium/Neuron images (NEURON_VLLM_IMAGE / NEURON_NXDI_IMAGE / NEURON_COMPILE_IMAGE) are also pinned here.
```

Accelerator families are toggled via `ENABLE_GPU_STANDARD` (G5/G6/G6e),
`ENABLE_G7_SUPPORT` (G7/G7e Blackwell, custom AMI) and `ENABLE_NEURON_SUPPORT`
(Trn1/Trn1n/Inf2/Trn2).

### Step 2 — Bootstrap cluster

```bash
bash setup-cluster.sh        # or: oai cluster bootstrap
```

Creates/reuses: EKS cluster, S3 buckets (encrypted + versioned), S3 Gateway Endpoint, gp3 StorageClass, GPU + Neuron NodePools, 3 least-privilege IAM roles, Pod Identity bindings, `oai-infopt` namespace. Idempotent (~15 minutes).

For Trainium, run the one-time node-group setup:

```bash
oai cluster setup-neuron     # creates or reuses the trn2-neuron managed node group
```

### Step 3 — Store HF token in Secrets Manager (gated models only)

```bash
aws secretsmanager create-secret \
  --name 'oai-infopt/hf-token' \
  --description 'HuggingFace API token for gated model download' \
  --secret-string '{"token":"hf_YOUR_TOKEN_HERE"}' \
  --region ap-southeast-4
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

### Step 5 — Onboard, download, deploy, benchmark (the `oai` way)

```bash
oai download gpt-oss-20b                 # HF -> S3 (skips if already present)
oai deploy  gpt-oss-20b --benchmark      # GPU: pick instance, deploy, benchmark

# Trainium / Neuron — first deploy compiles the NEFF; later deploys drop --compile:
oai download gpt-oss-20b-neuron
oai deploy  gpt-oss-20b-neuron --compile --managed-ng --benchmark
```

Equivalent manual flow (what the CLI generates):

```bash
bash model-download/<id>/download.sh
bash vllm/models/<id>/deploy.sh --hw <instance-type> [--tp <n>]
```

### Step 6 — Run benchmark manually (optional)

**One command (in-cluster, preferred):**

```bash
bash inference/run-benchmark.sh --manifest configs/manifests/<model>-<hw>.yaml --hw <instance-type>
```

**Local (via port-forward):**

```bash
kubectl port-forward -n oai-infopt svc/<service-name> 8080:8000 &

pip install openai pyyaml aiohttp
python inference/load-test.py \
  --manifest configs/manifests/<model>-<hw>.yaml \
  --profile  configs/workload_profiles/realtime_v1.yaml \
  --endpoint http://localhost:8080 \
  --output   results/
```

Results are written as JSONL to `results/<run_id>.jsonl` and uploaded to `s3://shellkode-ai-results/results/`.

### Step 7 — Teardown / scale to zero

```bash
oai stop <id>                # stop one model, free its accelerator node
# or manually:
bash vllm/models/<id>/stop.sh
bash vllm/stop.sh            # stop all vLLM deployments
oai cluster teardown         # delete the cluster (keeps S3 + IAM)
```

S3 model weights and IAM roles are retained.

## Porting a new model to Trainium — AutoPort kit (`neuron/autoport/`)

Some architectures are not supported by vLLM-Neuron out of the box and need a
real `trn2.48xlarge` to compile, run and validate the port on hardware. The
AutoPort kit stands that box up inside an EC2 Capacity Block and prepares it so
the only manual step is signing in to Claude Code and pasting a prompt.

```bash
# on your laptop — launch the box (private subnet, SSM only, no SSH key):
bash neuron/autoport/launch-box.sh --reservation cr-0123... --name autoport-gemma-4-31b --dry-run
bash neuron/autoport/launch-box.sh --reservation cr-0123... --name autoport-gemma-4-31b

# on the box (Session Manager), as ubuntu:
bash setup-box.sh --model-name gemma4 --weights-folder gemma-4-31B-it
cd ~/autoport/gemma4/vllm-neuron && claude      # sign in once, paste the PROMPT.md prompt

# release the Capacity Block slot back to the EKS neuron-inf NodePool when done:
bash neuron/autoport/launch-box.sh --terminate i-0123...
```

The box's root volume is deleted when the Capacity Block ends, so install
`backup-workspace.sh` as a cron job to snapshot the port (diff + new files) to
S3. Restore anywhere with the base commit in `latest/MANIFEST.json` and
`git apply latest/workspace.patch`. See [`neuron/autoport/README.md`](neuron/autoport/README.md).

> **Capacity note:** a Capacity Block with one instance serves either an AutoPort
> box **or** the EKS `neuron-inf` NodePool, not both. Terminate the box to hand
> the slot back. AutoPort boxes target the **vLLM Neuron** backend, which the
> Neuron SDK now recommends (NxD Inference is in maintenance mode).

## Models Reference

The catalog ships GPU and Trainium/Neuron variants of each model. Current entries:

| Model | Hardware | Notes |
|---|---|---|
| openai/gpt-oss-20b | GPU (MXFP4) / Neuron (bf16) | MoE reasoning model; smoke-tested end-to-end |
| Qwen3.5-4B | GPU / Neuron (vLLM) / Neuron (NxDI) | dense multimodal; text path benchmarked |
| Qwen3.5-9B | GPU TP1/TP4 / Neuron NxDI | tensor-parallel variants |
| Qwen3-VL-8B | Neuron | multimodal |
| Gemma-4-31B | GPU + `opt2` KV/batch A/B + FP8 TP1/TP4 | quality-ceiling variant |
| Gemma-4-e4b-it | GPU | |
| GLM-4.1V-9B-Thinking | GPU TP1/TP4 | |
| DeepSeek-R1-Distill-Qwen-7B | GPU | |
| Phi-4-mini-instruct | GPU | |

Full per-model fields (HF id, quantization, instance candidates, compile config)
live in `catalog/models/<id>.yaml`. Hardware feasibility and per-cell run
commands are in `docs/benchmark-matrix.md`.

## Security Posture

| Control | Implementation |
|---|---|
| No static AWS keys | IAM Roles Anywhere (local) / OIDC (CI) |
| Secrets | AWS Secrets Manager + External Secrets Operator |
| IAM least privilege | 3 separate roles: download / serving (read-only) / benchmark |
| Pod security | non-root, readOnlyRootFilesystem, drop ALL caps, seccomp RuntimeDefault |
| Network | Default-deny NetworkPolicy; ClusterIP only; ALB Ingress for external |
| AutoPort boxes | Private subnet, SSM Session Manager only, no SSH key, IMDSv2 required, encrypted gp3 |
| S3 | Encrypted at rest, versioned, public access blocked |
| EBS | gp3 encrypted StorageClass |

## Links

- Plane project: TBD
- ECR Public Gallery (vLLM GPU DLC): https://gallery.ecr.aws/deep-learning-containers/vllm
- AWS Neuron DLCs: https://github.com/aws-neuron/deep-learning-containers
- vLLM Neuron backend: https://github.com/vllm-project/vllm-neuron
- HF token (gated models): https://huggingface.co/settings/tokens
