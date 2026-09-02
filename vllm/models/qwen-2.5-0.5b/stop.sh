#!/bin/bash
# ==============================================================================
# vllm/models/qwen-2.5-0.5b/stop.sh
#
# All model-specific names come from model.env — this script is generic.
#
# Usage:
#   cd llm-inference-framework
#   bash vllm/models/qwen-2.5-0.5b/stop.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"
source "${SCRIPT_DIR}/model.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

echo ""
echo "══════════════════════════════════════════════════"
echo "  STOP — ${MODEL_ID}"
echo "  Deployment : ${DEPLOYMENT_NAME}"
echo "  Namespace  : ${BENCHMARK_NAMESPACE}"
echo "  GPU node terminates in ~10 min if no other workload"
echo "  S3 models are NOT affected"
echo "══════════════════════════════════════════════════"
echo ""

if ! kubectl get deployment "${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" >/dev/null 2>&1; then
    log_warn "No deployment '${DEPLOYMENT_NAME}' found in '${BENCHMARK_NAMESPACE}' — nothing to stop."
    exit 0
fi

log_info "Found running deployment: ${DEPLOYMENT_NAME}"
read -r -p "Stop '${DEPLOYMENT_NAME}'? [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

log_info "Deleting deployment: ${DEPLOYMENT_NAME}..."
kubectl delete deployment "${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true

log_info "Deleting service: ${SERVICE_NAME}..."
kubectl delete svc "${SERVICE_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true

log_info "Deleting PVC: ${PVC_NAME}..."
kubectl delete pvc "${PVC_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true

# Delete GPU nodes so billing stops immediately
log_info "Terminating GPU nodes..."
kubectl delete node -l karpenter.sh/nodepool=gpu-inf 2>/dev/null || true
kubectl delete node -l karpenter.sh/nodepool=gpu-nodepool 2>/dev/null || true
log_info "GPU nodes deleted — billing stops within minutes."

echo ""
log_info "Remaining deployments in ${BENCHMARK_NAMESPACE}:"
kubectl get deployment -n "${BENCHMARK_NAMESPACE}" 2>/dev/null || true

echo ""
echo "══════════════════════════════════════════════════"
echo "  STOPPED — ${MODEL_ID}"
echo ""
echo "  To redeploy:"
echo "    bash vllm/models/qwen-2.5-0.5b/deploy.sh"
echo "══════════════════════════════════════════════════"
