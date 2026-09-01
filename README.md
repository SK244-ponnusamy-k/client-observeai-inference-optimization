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
│   │   ├── service-private.yaml          # ClusterIP (default — internal only)
│   │   ├── service-public.yaml           # LoadBalancer (public NLB)
│   │   └── service-ingress.yaml          # HTTPS ALB Ingress (production)
│   └── models/
│       ├── qwen-2.5-0.5b/
│       │   ├── deploy.sh                 # Run this → vLLM serving qwen-0.5b
│       │   ├── deployment.yaml
│       │   └── pvc.yaml
│       └── gpt-oss-20b/
│           ├── deploy.sh                 # Run this → vLLM serving gpt-oss-20b
│           ├── deployment.yaml
│           └── pvc.yaml
│
└── inference/
    ├── quick-test.sh                     # Single request smoke test
    ├── load-test.py                      # Local load test (via port-forward)
    └── benchmark-job.yaml                # In-cluster 100-request benchmark
```

---

## Step 1 — Configure

Edit `config/config.env` and set your values:

```bash
AWS_REGION="us-east-2"
CLUSTER_NAME="ai-eks-cluster"
MODEL_BUCKET="shellkode-ai-models"    # must be lowercase, globally unique
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

This creates:
- EKS Auto Mode cluster with tags
- gp3 StorageClass
- GPU NodePool (g6e family, auto-scales via Karpenter)
- IAM Role + Pod Identity + ServiceAccount

Takes ~15 minutes. Run once. Re-run safely — all steps are idempotent.

---

## Step 4 — Download a Model to S3

### Qwen 2.5 0.5B (no token required)
```bash
bash model-download/qwen-2.5-0.5b/download.sh
```

### GPT-OSS-20B (HF token required)
```bash
# 1. Accept license: https://huggingface.co/openai/gpt-oss-20b
# 2. Create token secret:
kubectl create secret generic hf-token --from-literal=token=hf_YOURTOKEN
# 3. Run download:
bash model-download/gpt-oss-20b/download.sh
```

Each script:
- Checks if model already exists in S3 (skips if so)
- Creates Kubernetes Job
- Streams logs
- Deletes job automatically on completion

---

## Step 5 — Deploy vLLM

### Deploy Qwen 0.5B
```bash
bash vllm/models/qwen-2.5-0.5b/deploy.sh
```

### Deploy GPT-OSS-20B
```bash
bash vllm/models/gpt-oss-20b/deploy.sh
```

Each deploy script:
- Verifies model exists in S3
- Removes any existing vLLM deployment
- Applies PVC + service + deployment
- Waits for pod to be Ready
- Karpenter auto-provisions the GPU node

---

## Step 6 — Test

### Local test (port-forward)
```bash
# Terminal 1
kubectl port-forward svc/vllm-inference 8080:8000 &

# Terminal 2
bash inference/quick-test.sh qwen-0.5b
bash inference/quick-test.sh gpt-oss-20b
```

### Load test (local, 10 requests)
```bash
pip install openai
python inference/load-test.py --model qwen-0.5b --requests 10
```

### In-cluster benchmark (100 requests, real latency)
```bash
kubectl apply -f inference/benchmark-job.yaml
kubectl logs job/vllm-benchmark -f
kubectl delete job vllm-benchmark
```

---

## Step 7 — Expose Publicly (optional)

```bash
kubectl delete -f vllm/service/service-private.yaml
kubectl apply  -f vllm/service/service-public.yaml
kubectl get svc vllm-inference    # wait ~2 min for EXTERNAL-IP
```

Call from anywhere:
```
http://EXTERNAL-IP:8000/v1/chat/completions
```

---

## Switch Between Models

```bash
# Switch to GPT-OSS-20B
bash vllm/models/gpt-oss-20b/deploy.sh

# Switch back to Qwen
bash vllm/models/qwen-2.5-0.5b/deploy.sh
```

The deploy scripts handle teardown of the previous model automatically.

---

## Stopping vLLM (keep cluster, save GPU cost)

```bash
bash vllm/stop.sh
```

Deletes deployment, service, PVC. GPU node terminates in ~10 min. Cluster keeps running.

---

## Tear Down

```bash
bash cluster/teardown.sh
```

Keeps: S3 model files, IAM Role.
Deletes: Cluster, GPU nodes, PVCs, services.

---

## Resume After Teardown

```bash
bash cluster/bootstrap.sh                          # recreate cluster (~15 min)
bash vllm/models/qwen-2.5-0.5b/deploy.sh          # deploy — S3 already has model
```

---

## Cost Reference

| State | Cost |
|-------|------|
| Cluster deleted | ~$0.32/month (S3 only) |
| Cluster idle (no vLLM) | ~$0.10/hour |
| vLLM running (g6e.xlarge) | ~$1.08/hour |
| vLLM running (g6e.4xlarge) | ~$3.71/hour |

GPU node is provisioned only when vLLM deployment is running. Delete the deployment to stop GPU billing.

---

## Benchmark Results (Qwen 0.5B, g6e.4xlarge, 100 requests in-cluster)

| Metric | Value |
|--------|-------|
| TTFT avg | 18.9 ms |
| TTFT p50 | 15.7 ms |
| TTFT p90 | 16.4 ms |
| TTFT p99 | 17.0 ms |
| E2E latency avg | 263 ms |
| Model load time | 1.7 seconds |

---

## Adding a New Model

1. Create folder: `model-download/your-model-name/`
2. Copy `download.sh` from an existing model, update `MODEL_ID` and `MODEL_FOLDER`
3. Copy `job.yaml`, update model values
4. Create folder: `vllm/models/your-model-name/`
5. Copy `deploy.sh`, `deployment.yaml`, `pvc.yaml` from an existing model
6. Update `--model`, `--served-model-name`, resources in `deployment.yaml`
7. Run: `bash model-download/your-model-name/download.sh`
8. Run: `bash vllm/models/your-model-name/deploy.sh`
