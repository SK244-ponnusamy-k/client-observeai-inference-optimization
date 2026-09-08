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

## Team Members

- @shellkode-genai (GitHub handle — update when assigned)

## Repository Layout

```
llm-inference-framework/
├── config/
│   └── config.env                  # Single source of truth — source before running
│
├── models/                         # Model specifications (Single Source of Truth)
│   ├── gpt-oss-20b.yaml
│   ├── qwen-2.5-0.5b.yaml
│   ├── qwen-3.5-4b.yaml
│   └── qwen3-35b-nvfp4.yaml
│
├── configs/
│   ├── manifests/                  # Benchmark manifests (drives benchmark parameters)
│   │   ├── gpt-oss-20b-baseline.yaml
│   │   ├── qwen-2.5-0.5b-baseline.yaml
│   │   ├── qwen-3.5-4b-baseline.yaml
│   │   └── qwen3-35b-nvfp4-baseline.yaml
│   └── workload_profiles/
│       ├── realtime_v1.yaml        # Low concurrency, TTFT-bound
│       └── batch_v1.yaml           # High concurrency, throughput-bound
│
├── scripts/                        # Model-agnostic automation scripts
│   ├── download.sh                 # HF → S3 via K8s Job
│   ├── deploy.sh                   # vLLM EKS Deployment (supports --benchmark flag)
│   ├── expose-public.sh            # Public LoadBalancer toggle script
│   ├── stop.sh                     # Teardown model & release GPU nodes
│   ├── compare_models.py           # Multi-model comparison & Excel report generator
│   └── post-deploy-validate.sh     # Validation pipeline
│
├── inference/
│   ├── load-test.py                # Profile-driven benchmark harness (TTFT/ITL/cost/GPU)
│   ├── benchmark-job.yaml          # In-cluster benchmark Job spec
│   └── run-benchmark.sh            # Benchmark orchestrator & S3 sync
│
├── cluster/                        # EKS IaC & Karpenter NodePool setup
│   ├── bootstrap.sh                # Full cluster + S3 setup
│   ├── teardown.sh                 # Deletes cluster; retains S3 + IAM roles
│   ├── setup-pod-identity.sh       # IAM Roles & EKS Pod Identity bindings
│   ├── gpu-nodepool.yaml           # Karpenter NodePool: gpu-inf (g6e/g6/g5)
│   └── storage-class.yaml          # gp3 EBS StorageClass
│
└── monitoring/                     # AMP / Prometheus / Grafana / Metrics Collector
    ├── metrics-collector/
    │   └── collector.py            # AMP metrics harvester
    ├── setup-monitoring.sh
    └── dashboards/
```

## How To Run

### Step 1 — Download Model Weights to S3

```bash
bash scripts/download.sh --model qwen-3.5-4b
```

### Step 2 — Deploy vLLM Serving Pod

```bash
bash scripts/deploy.sh --model qwen-3.5-4b
# Or Deploy & Benchmark automatically:
bash scripts/deploy.sh --model qwen-3.5-4b --benchmark
```

### Step 3 — Run Benchmark

```bash
# Realtime profile:
bash inference/run-benchmark.sh --model qwen-3.5-4b --profile realtime

# Batch profile:
bash inference/run-benchmark.sh --model qwen-3.5-4b --profile batch
```

### Step 4 — Multi-Model Comparison & Excel Report

```bash
python scripts/compare_models.py --s3-bucket shellkode-ai-results
```

### Step 5 — Teardown Model

```bash
bash scripts/stop.sh --model qwen-3.5-4b
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
| Qwen/Qwen3.5-4B | `models/Qwen3.5-4B/` | ~8 GB | g6e.2xlarge | ~8 GB |
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
