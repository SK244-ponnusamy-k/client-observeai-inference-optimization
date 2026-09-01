#!/bin/bash
# ==============================================================================
# vllm/models/gpt-oss-20b/stop.sh
#
# Stops ONLY the gpt-oss-20b deployment and its resources.
# Other running models (e.g. qwen-2.5-0.5b) are NOT affected.
# GPU node terminates automatically within ~10 minutes (Karpenter) if no
# other workloads remain on it.
# S3 model files are NOT affected.
#
# Usage:
#   cd llm-inference-framework
#   bash vllm/models/gpt-oss-20b/stop.sh
#
# To revert public LB back to private (without full stop):
#   kubectl apply -f vllm/models/gpt-oss-20b/service/service-private.yaml
#
# To also delete the cluster:
#   bash cluster/teardown.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

DEPLOYMENT_NAME="vllm-gpt-oss-20b"
SERVICE_NAME="vllm-gpt-oss-20b"
PVC_NAME="metadata-cache-gpt-oss-20b"

echo ""
echo "══════════════════════════════════════════════════"
echo "  STOP — gpt-oss-20b"
echo "  Deployment : ${DEPLOYMENT_NAME}"
echo "  Service    : ${SERVICE_NAME}"
echo "  GPU node terminates in ~10 min if no other workload"
echo "  S3 models are NOT affected"
echo "══════════════════════════════════════════════════"
echo ""

# ------------------------------------------------------------------------------
# Check what's currently running
# ------------------------------------------------------------------------------
if ! kubectl get deployment "${DEPLOYMENT_NAME}" -n "${NAMESPACE}" >/dev/null 2>&1; then
    log_warn "No deployment '${DEPLOYMENT_NAME}' found — nothing to stop."
    exit 0
fi

log_info "Found running deployment: ${DEPLOYMENT_NAME}"

read -r -p "Stop '${DEPLOYMENT_NAME}'? [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

# ------------------------------------------------------------------------------
# Delete deployment
# ------------------------------------------------------------------------------
log_info "Deleting deployment: ${DEPLOYMENT_NAME}..."
kubectl delete deployment "${DEPLOYMENT_NAME}" -n "${NAMESPACE}" --ignore-not-found=true

# ------------------------------------------------------------------------------
# Delete service
# ------------------------------------------------------------------------------
log_info "Deleting service: ${SERVICE_NAME}..."
kubectl delete svc "${SERVICE_NAME}" -n "${NAMESPACE}" --ignore-not-found=true

# ------------------------------------------------------------------------------
# Delete PVC
# ------------------------------------------------------------------------------
log_info "Deleting PVC: ${PVC_NAME}..."
kubectl delete pvc "${PVC_NAME}" -n "${NAMESPACE}" --ignore-not-found=true 2>/dev/null || true

# ------------------------------------------------------------------------------
# Show remaining resources
# ------------------------------------------------------------------------------
echo ""
log_info "Remaining inference deployments:"
kubectl get deployment -l component=inference-server -n "${NAMESPACE}" 2>/dev/null || true

echo ""
log_info "Remaining pods (all namespaces):"
kubectl get pods -n "${NAMESPACE}" 2>/dev/null || true

echo ""
log_info "Remaining nodes (GPU node terminates in ~10 min if idle):"
kubectl get nodes 2>/dev/null || true

echo ""
echo "══════════════════════════════════════════════════"
echo "  STOPPED — gpt-oss-20b"
echo "  Deleted : deployment/${DEPLOYMENT_NAME}, svc/${SERVICE_NAME}, pvc/${PVC_NAME}"
echo "  Kept    : cluster, S3 models, IAM Role, other running models"
echo ""
echo "  To redeploy:"
echo "    bash vllm/models/gpt-oss-20b/deploy.sh"
echo ""
echo "  To stop all models:"
echo "    bash vllm/stop.sh"
echo ""
echo "  To delete cluster too:"
echo "    bash cluster/teardown.sh"
echo "══════════════════════════════════════════════════"
