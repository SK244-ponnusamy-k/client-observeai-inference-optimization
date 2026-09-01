#!/bin/bash
# ==============================================================================
# vllm/stop.sh
#
# Stops the running vLLM deployment and releases all resources.
# GPU node terminates automatically within ~10 minutes (Karpenter).
# S3 model files are NOT affected.
# Cluster keeps running (control plane ~$0.10/hr continues).
#
# Usage:
#   cd llm-inference-framework
#   bash vllm/stop.sh
#
# To also delete the cluster:
#   bash cluster/teardown.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

echo ""
echo "══════════════════════════════════════════════════"
echo "  STOP vLLM DEPLOYMENT"
echo "  Cluster : ${CLUSTER_NAME}"
echo "  GPU node terminates in ~10 min after stop"
echo "  S3 models are NOT affected"
echo "══════════════════════════════════════════════════"
echo ""

# ------------------------------------------------------------------------------
# Check what's currently running
# ------------------------------------------------------------------------------
CURRENT_MODEL=""
if kubectl get deployment vllm -n "${NAMESPACE}" >/dev/null 2>&1; then
    CURRENT_MODEL=$(kubectl get deployment vllm -n "${NAMESPACE}" \
        -o jsonpath='{.metadata.labels.model}' 2>/dev/null || echo "unknown")
    log_info "Found running deployment — model: ${CURRENT_MODEL}"
else
    log_warn "No vLLM deployment found — nothing to stop."
    exit 0
fi

read -r -p "Stop deployment '${CURRENT_MODEL}'? [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

# ------------------------------------------------------------------------------
# Delete deployment
# ------------------------------------------------------------------------------
log_info "Deleting vLLM deployment..."
kubectl delete deployment vllm -n "${NAMESPACE}" --ignore-not-found=true

# ------------------------------------------------------------------------------
# Delete service (both private and public if either exists)
# ------------------------------------------------------------------------------
log_info "Deleting service..."
kubectl delete svc vllm-inference -n "${NAMESPACE}" --ignore-not-found=true

# ------------------------------------------------------------------------------
# Delete PVC for current model
# ------------------------------------------------------------------------------
log_info "Deleting PVC..."
kubectl delete pvc "metadata-cache-${CURRENT_MODEL}" \
    -n "${NAMESPACE}" --ignore-not-found=true 2>/dev/null || \
kubectl delete pvc metadata-cache \
    -n "${NAMESPACE}" --ignore-not-found=true 2>/dev/null || true

# ------------------------------------------------------------------------------
# Show remaining resources
# ------------------------------------------------------------------------------
echo ""
log_info "Remaining pods:"
kubectl get pods -n "${NAMESPACE}" 2>/dev/null || true

echo ""
log_info "Remaining nodes (GPU node will terminate in ~10 min):"
kubectl get nodes 2>/dev/null || true

echo ""
echo "══════════════════════════════════════════════════"
echo "  STOPPED"
echo "  Deleted : deployment, service, PVC"
echo "  Kept    : cluster, S3 models, IAM Role"
echo ""
echo "  To redeploy:"
echo "    bash vllm/models/qwen-2.5-0.5b/deploy.sh"
echo "    bash vllm/models/gpt-oss-20b/deploy.sh"
echo ""
echo "  To delete cluster too:"
echo "    bash cluster/teardown.sh"
echo "══════════════════════════════════════════════════"
