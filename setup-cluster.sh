#!/bin/bash
# ==============================================================================
# setup-cluster.sh
#
# SINGLE ENTRY POINT — run this once on a new AWS account to set up everything.
#
# What this does end-to-end:
#   Phase 1 — Cluster & Infrastructure  (cluster/bootstrap.sh)
#     1.  Validate tools + AWS credentials
#     2.  Create S3 buckets (model weights + benchmark results)
#     3.  Create EKS cluster with Auto Mode + Karpenter
#     4.  Add S3 VPC Gateway Endpoint (free private S3 access)
#     5.  Update kubeconfig
#     6.  Apply gp3 StorageClass
#     7.  Apply GPU NodePool (g5/g6/g6e, on-demand, auto-terminates when idle)
#     8.  Set up Pod Identity — 3 least-privilege IAM roles + ServiceAccounts
#     9.  Push HF token to Secrets Manager (reads from config/.env)
#
#   Phase 2 — K8s Prerequisites
#     10. Apply vLLM ConfigMap
#     11. Apply NetworkPolicies (default-deny + allow-list for oai-infopt ns)
#
#   Phase 3 — Monitoring Stack  (monitoring/setup-monitoring.sh)
#     12. Create AMP workspace
#     13. Create IAM policy for AMP + S3 + CloudWatch
#     14. Create monitoring namespace + service accounts + Pod Identity
#     15. Install kube-prometheus-stack (Prometheus → AMP + Grafana)
#     16. Install NVIDIA DCGM Exporter (GPU metrics DaemonSet)
#     17. Apply metrics-collector CronJob (AMP → S3 JSON reports every 30min)
#
# After this script completes:
#   - Cluster is fully running
#   - Grafana is live at the printed URL (dashboards pre-loaded)
#   - Metrics flow: cluster → Prometheus → AMP → Grafana + S3 reports
#   - Models still need to be downloaded and deployed separately:
#       bash vllm/models/qwen-2.5-0.5b/deploy.sh
#       bash vllm/models/gpt-oss-20b/deploy.sh
#
# Prerequisites:
#   - aws CLI configured (aws configure)
#   - eksctl, kubectl, helm installed
#   - config/.env filled in with your HF token
#     (get token from https://huggingface.co/settings/tokens)
#
# Usage:
#   cd llm-inference-framework
#   bash setup-cluster.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }
log_phase() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  $1${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
}

# ==============================================================================
# Pre-flight checks
# ==============================================================================
log_phase "Pre-flight checks"

# Tools
for tool in aws eksctl kubectl helm; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        log_error "Missing required tool: ${tool}"
        log_error "Install it then re-run setup-cluster.sh"
        exit 1
    fi
    log_info "${tool} — OK"
done

# AWS credentials
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    log_error "AWS credentials not configured. Run: aws configure"
    exit 1
fi
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
log_info "AWS Account : ${AWS_ACCOUNT_ID}"
log_info "AWS Region  : ${AWS_REGION}"

# HF token check — warn but don't block (Qwen is public, only gpt-oss-20b needs it)
ENV_FILE="${SCRIPT_DIR}/config/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    log_warn "config/.env not found."
    log_warn "Create it to enable gated model download (openai/gpt-oss-20b):"
    log_warn "  echo 'HF_TOKEN=\"hf_your_token\"' > config/.env"
else
    HF_CHECK=$(grep -E '^HF_TOKEN=' "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"' | tr -d "'" || echo "")
    if [[ -z "${HF_CHECK}" || "${HF_CHECK}" == "hf_REPLACE_WITH_YOUR_TOKEN" ]]; then
        log_warn "HF_TOKEN not set in config/.env — gpt-oss-20b download will be skipped."
        log_warn "Set it at: config/.env  →  HF_TOKEN=\"hf_your_token\""
    else
        log_info "HF token found in config/.env — will push to Secrets Manager."
    fi
fi

# ==============================================================================
# Show full configuration and confirm
# ==============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  FULL CLUSTER SETUP — CONFIGURATION SUMMARY                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-20s : %-37s║\n" "Cluster"        "${CLUSTER_NAME}"
printf "║  %-20s : %-37s║\n" "Region"         "${AWS_REGION}"
printf "║  %-20s : %-37s║\n" "Account"        "${AWS_ACCOUNT_ID}"
printf "║  %-20s : %-37s║\n" "EKS Version"    "${EKS_VERSION}"
printf "║  %-20s : %-37s║\n" "Model Bucket"   "${MODEL_BUCKET}"
printf "║  %-20s : %-37s║\n" "Results Bucket" "${RESULTS_BUCKET}"
printf "║  %-20s : %-37s║\n" "Namespace"      "${BENCHMARK_NAMESPACE}"
printf "║  %-20s : %-37s║\n" "vLLM Image"     "${VLLM_IMAGE:0:37}"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Phases:                                                     ║"
echo "║    Phase 1 — EKS cluster + S3 + IAM + StorageClass + NodePool║"
echo "║    Phase 2 — K8s ConfigMap + NetworkPolicy                   ║"
echo "║    Phase 3 — AMP + Grafana + DCGM + metrics collector        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
read -r -p "Start full setup? [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

# ==============================================================================
# Phase 1 — Cluster & Infrastructure
# ==============================================================================
log_phase "Phase 1 — Cluster & Infrastructure"
bash "${SCRIPT_DIR}/cluster/bootstrap.sh"
log_info "Phase 1 complete."

# ==============================================================================
# Phase 2 — K8s Prerequisites
# ==============================================================================
log_phase "Phase 2 — K8s Prerequisites"

log_info "Applying vLLM ConfigMap..."
kubectl apply -f "${SCRIPT_DIR}/k8s/serving/vllm-configmap.yaml"
log_info "ConfigMap applied."

log_info "Applying NetworkPolicies..."
kubectl apply -f "${SCRIPT_DIR}/k8s/network-policy.yaml" -n "${BENCHMARK_NAMESPACE}"
log_info "NetworkPolicies applied."

log_info "Phase 2 complete."

# ==============================================================================
# Phase 3 — Monitoring Stack
# ==============================================================================
log_phase "Phase 3 — Monitoring Stack"

if command -v helm >/dev/null 2>&1; then
    bash "${SCRIPT_DIR}/monitoring/setup-monitoring.sh"
    log_info "Phase 3 complete."
else
    log_warn "helm not found — skipping monitoring."
    log_warn "Install helm then run:  bash monitoring/setup-monitoring.sh"
fi

# ==============================================================================
# Done — print next steps
# ==============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SETUP COMPLETE                                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  Cluster and monitoring are fully configured.                ║"
echo "║                                                              ║"
echo "║  Verify everything is running:                               ║"
echo "║    kubectl get nodes                                         ║"
echo "║    kubectl get pods -n ${BENCHMARK_NAMESPACE}                        ║"
echo "║    kubectl get pods -n monitoring                            ║"
echo "║                                                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Next — deploy a model:                                      ║"
echo "║                                                              ║"
echo "║  Qwen 2.5-0.5B (public, fast, ~950MB):                       ║"
echo "║    bash vllm/models/qwen-2.5-0.5b/deploy.sh                 ║"
echo "║                                                              ║"
echo "║  GPT-OSS-20B (gated, 13GB, needs HF token):                  ║"
echo "║    bash vllm/models/gpt-oss-20b/deploy.sh                   ║"
echo "║                                                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Grafana — get URL:                                          ║"
echo "║    kubectl get ingress kube-prometheus-stack-grafana \\       ║"
echo "║      -n monitoring                                           ║"
echo "║                                                              ║"
echo "║  Grafana password:                                           ║"
echo "║    kubectl get secret kube-prometheus-stack-grafana \\       ║"
echo "║      -n monitoring \\                                         ║"
echo "║      -o jsonpath=\"{.data.admin-password}\" | base64 -d       ║"
echo "║                                                              ║"
echo "║  Metrics reports:                                            ║"
echo "║    aws s3 ls s3://${RESULTS_BUCKET}/monitoring/reports/      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
