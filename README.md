# LLM Inference Framework on EKS

Run open-source LLMs on AWS EKS with GPU auto-scaling, S3 model storage, and vLLM inference.

---

## Directory Structure

```
llm-inference-framework/
│
├── config/
│   └── config.env                        # ← Single source of truth (edit this first)
│
├── cluster/
│   ├── bootstrap.sh                      # Create cluster + all config in one shot
│   ├── teardown.sh                       # Delete cluster (keeps S3 + IAM)
│   ├── setup-pod-identity.sh             # IAM Role + Pod Identity (called by bootstrap)
│   ├── model-storage-sa.yaml             # Kubernetes ServiceAccount
│   ├── storage-class.yaml                # gp3 EBS StorageClass
│   └── gpu-nodepool.yaml                 # Karpenter GPU NodePool (g6e family)
│
├── model-download/
│   ├── qwen-2.5-0.5b/
│   │   ├── download.sh                   # Run this → model lands in S3, job deleted
│   │   └── job.yaml                      # Kubernetes Job (used by download.sh)
│   └── gpt-oss-20b/
│       ├── download.sh                   # Run this → model lands in S3, job deleted
│       └── job.yaml                      # Kubernetes Job (requires hf-token secret)
│
├── vllm/
│   ├── service/
│   │   ├── service-private.yaml          # ClusterIP (legacy single-model)
│   │   ├── service-public.yaml           # LoadBalancer (legacy single-model)
│   │   └── service-ingress.yaml          # HTTPS ALB Ingress (production)
│   ├── models/
│   │   ├── qwen-2.5-0.5b/
│   │   │   ├── deploy.sh                 # Run this → vLLM serving qwen-0.5b
│   │   │   ├── deployment.yaml           # Deployment name: vllm-qwen-0.5b
│   │   │   ├── service.yaml              # Service name: vllm-qwen-0.5b
│   │   │   └── pvc.yaml                  # PVC: metadata-cache-qwen-0.5b
│   │   └── gpt-oss-20b/
│   │       ├── deploy.sh                 # Run this → vLLM serving gpt-oss-20b
│   │       ├── deployment.yaml           # Deployment name: vllm-gpt-oss-20b
│   │       ├── service.yaml              # Service name: vllm-gpt-oss-20b
│   │       └── pvc.yaml                  # PVC: metadata-cache-gpt-oss-20b
│   └── stop.sh                           # Stop a running vLLM deployment
│
└── inference/
    ├── quick-test.sh                     # Single request smoke test
    ├── load-test.py                      # Local load test (via port-forward)
    └── benchmark-job.yaml                # In-cluster 100-request benchmark
```

---

## Step 1 — Configure

Edit `config/config.env`:

```bash
AWS_REGION="us-east-2"
CLUSTER_NAME="ai-eks-cluster"
MODEL_BUCKET="shellkode-ai-models"
TAG_OWNER="shellkode"
```

---

## Step 2 — Create S3 Bucket

```bash
aws s3 mb s3://shellkode-ai-models --region us-east-2
```

---

## Step 3 — Bootstrap Cluster

```bash
cd llm-inference-framework
bash cluster/bootstrap.sh
```

Creates: EKS cluster, gp3 StorageClass, GPU NodePool (g6e), IAM Role, Pod Identity, ServiceAccount.

Takes ~15 minutes. Idempotent — safe to re-run after cluster recreation.

---

## Step 4 — Download Models to S3

### Qwen 2.5 0.5B (no token required)
```bash
bash model-download/qwen-2.5-0.5b/download.sh
```

### GPT-OSS-20B (HF token required)
```bash
# 1. Accept license: https://huggingface.co/openai/gpt-oss-20b
# 2. Create token secret:
kubectl create secret generic hf-token --from-literal=token=hf_YOURTOKEN
# 3. Download:
bash model-download/gpt-oss-20b/download.sh
```

Each script checks if the model already exists in S3 — skips download if so.

---

## Step 5 — Deploy vLLM

### Deploy one model
```bash
bash vllm/models/qwen-2.5-0.5b/deploy.sh
bash vllm/models/gpt-oss-20b/deploy.sh
```

### Deploy both models simultaneously
Each model has a unique deployment name and service — they run on separate GPU nodes:

```bash
# Terminal 1
bash vllm/models/gpt-oss-20b/deploy.sh

# Terminal 2 (at the same time)
bash vllm/models/qwen-2.5-0.5b/deploy.sh
```

Verify both are running:
```bash
kubectl get deployment -l component=inference-server
kubectl get pods -l component=inference-server
kubectl get svc | grep vllm
```

---

## Step 6 — Test

### Port-forward each model to a different local port

```bash
# gpt-oss-20b → port 8081
kubectl port-forward svc/vllm-gpt-oss-20b 8081:8000 &

# qwen-0.5b → port 8080
kubectl port-forward svc/vllm-qwen-0.5b 8080:8000 &
```

### Quick test
```bash
bash inference/quick-test.sh gpt-oss-20b 8081
bash inference/quick-test.sh qwen-0.5b 8080
```

### Python client
```python
from openai import OpenAI

# gpt-oss-20b
client = OpenAI(base_url="http://localhost:8081/v1", api_key="dummy")
response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "Explain Kubernetes."}],
    max_tokens=256
)
print(response.choices[0].message.content)
print(response.choices[0].message.reasoning)  # chain-of-thought

# qwen-0.5b
client2 = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")
response2 = client2.chat.completions.create(
    model="qwen-0.5b",
    messages=[{"role": "user", "content": "Explain Kubernetes."}],
    max_tokens=256
)
print(response2.choices[0].message.content)
```

### Load test
```bash
pip install openai
python inference/load-test.py --model gpt-oss-20b --port 8081 --requests 10
python inference/load-test.py --model qwen-0.5b   --port 8080 --requests 10
```

### In-cluster benchmark (100 requests, real latency)
```bash
kubectl apply -f inference/benchmark-job.yaml
kubectl logs job/vllm-benchmark -f
kubectl delete job vllm-benchmark
```

---

## Step 7 — Expose Publicly (optional)

Each model can be independently exposed. Apply a LoadBalancer service per model:

```bash
# Expose gpt-oss-20b publicly
kubectl patch svc vllm-gpt-oss-20b -p '{"spec":{"type":"LoadBalancer"}}'
kubectl get svc vllm-gpt-oss-20b   # wait ~2 min for EXTERNAL-IP

# Expose qwen-0.5b publicly
kubectl patch svc vllm-qwen-0.5b -p '{"spec":{"type":"LoadBalancer"}}'
kubectl get svc vllm-qwen-0.5b
```

---

## Stopping vLLM

### Stop a specific model
```bash
kubectl delete deployment vllm-gpt-oss-20b
kubectl delete svc vllm-gpt-oss-20b
kubectl delete pvc metadata-cache-gpt-oss-20b

kubectl delete deployment vllm-qwen-0.5b
kubectl delete svc vllm-qwen-0.5b
kubectl delete pvc metadata-cache-qwen-0.5b
```

### Stop all vLLM deployments
```bash
bash vllm/stop.sh
```

GPU nodes terminate automatically in ~10 minutes via Karpenter.

---

## Tear Down Cluster

```bash
bash cluster/teardown.sh
```

Keeps: S3 model files, IAM Role.
Deletes: EKS cluster, GPU nodes, PVCs, services, VPC.

---

## Resume After Teardown

```bash
bash cluster/bootstrap.sh                       # ~15 min
bash vllm/models/gpt-oss-20b/deploy.sh          # S3 already has model
bash vllm/models/qwen-2.5-0.5b/deploy.sh        # S3 already has model
```

---

## Models Reference

| Model | Folder in S3 | Size | GPU | RAM needed | Port |
|-------|-------------|------|-----|-----------|------|
| Qwen2.5-0.5B-Instruct | `Qwen2.5-0.5B-Instruct/` | ~950MB | g6e.xlarge | 14GB | 8080 |
| openai/gpt-oss-20b | `gpt-oss-20b/` | ~13GB | g6e.2xlarge+ | 30GB | 8081 |

---

## Service Names Reference

| Model | Deployment | Service | Default Local Port |
|-------|-----------|---------|-------------------|
| qwen-0.5b | `vllm-qwen-0.5b` | `vllm-qwen-0.5b` | 8080 |
| gpt-oss-20b | `vllm-gpt-oss-20b` | `vllm-gpt-oss-20b` | 8081 |

---

## Cost Reference

| State | Cost |
|-------|------|
| Cluster deleted | ~$0.32/month (S3 only) |
| Cluster idle (no vLLM) | ~$0.10/hour |
| Qwen only (g6e.xlarge) | ~$1.08/hour total |
| GPT-OSS-20B only (g6e.2xlarge+) | ~$2.17/hour total |
| Both models running | ~$3.25/hour total |

GPU nodes only run when vLLM deployments are active.

---

## Benchmark Results (in-cluster, 100 requests)

### Qwen2.5-0.5B on g6e.4xlarge
| Metric | Value |
|--------|-------|
| TTFT avg | 18.9 ms |
| TTFT p50 | 15.7 ms |
| TTFT p99 | 17.0 ms |
| E2E latency avg | 263 ms |
| Model load time | 1.7 seconds |

### GPT-OSS-20B on g6e.4xlarge
| Metric | Value |
|--------|-------|
| Model load time | 7.3 seconds |
| KV cache | 24.49 GiB |
| Max concurrency | 170 parallel requests |

---

## Adding a New Model

```bash
# 1. Copy model-download folder
cp -r model-download/qwen-2.5-0.5b model-download/your-model

# 2. Edit download.sh — change MODEL_ID, MODEL_FOLDER, JOB_NAME
# 3. Edit job.yaml — change job name, model values

# 4. Copy vllm/models folder
cp -r vllm/models/qwen-2.5-0.5b vllm/models/your-model

# 5. Edit deployment.yaml — change name, app label, --model, --served-model-name
# 6. Edit service.yaml — change name, selector
# 7. Edit pvc.yaml — change PVC name
# 8. Edit deploy.sh — change DEPLOYMENT_NAME, MODEL_NAME, MODEL_FOLDER

# 9. Download and deploy
bash model-download/your-model/download.sh
bash vllm/models/your-model/deploy.sh
```

---

## Quick Reference

```bash
# Status
kubectl get deployment -l component=inference-server
kubectl get pods -l component=inference-server
kubectl get svc | grep vllm

# Logs
kubectl logs deployment/vllm-gpt-oss-20b 2>&1 | tail -20
kubectl logs deployment/vllm-qwen-0.5b 2>&1 | tail -20

# Port-forward
kubectl port-forward svc/vllm-gpt-oss-20b 8081:8000 &
kubectl port-forward svc/vllm-qwen-0.5b 8080:8000 &

# Test
bash inference/quick-test.sh gpt-oss-20b 8081
bash inference/quick-test.sh qwen-0.5b 8080

# S3 models
aws s3 ls s3://shellkode-ai-models/ --recursive --human-readable
```
