#!/bin/bash
# ==============================================================================
# vllm/models/qwen3.5-4b/stop.sh
#
# Full teardown — deletes deployment + service + PVC, terminates the model's GPU
# node (stops billing), and cleans up this model's benchmark jobs. Leaves S3
# weights and the monitoring stack untouched.
#
# Usage:
#   bash vllm/models/qwen3.5-4b/stop.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"
source "${SCRIPT_DIR}/model.env"
source "${FRAMEWORK_ROOT}/vllm/lib/gpu-terminate.sh"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }

echo ""
echo "══════════════════════════════════════════════════"
echo "  FULL STOP — ${MODEL_ID}"
echo "  Deployment : ${DEPLOYMENT_NAME}"
echo "  Namespace  : ${BENCHMARK_NAMESPACE}"
echo "══════════════════════════════════════════════════"
echo ""
echo "  This will:"
echo "    • Delete vLLM deployment, service, PVC"
echo "    • Terminate the GPU node for this model (stops billing)"
echo "    • Clean up benchmark jobs for this model only"
echo "    • Leave monitoring stack running (~\$0.15/hr)"
echo ""

read -r -p "Proceed? [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

log_info "Looking up node for deployment: ${DEPLOYMENT_NAME}..."
MODEL_NODE=$(kubectl get pod -l "app=${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" \
    -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)
if [[ -z "${MODEL_NODE}" ]]; then
    MODEL_NODE=$(kubectl get pod -l "app.kubernetes.io/name=${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)
fi
[[ -n "${MODEL_NODE}" ]] && log_info "Model is running on node: ${MODEL_NODE}" \
    || log_warn "Pod not found — node located via nodeclaim instance type."

log_info "Deleting deployment: ${DEPLOYMENT_NAME}..."
kubectl delete deployment "${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true
log_info "Deleting service: ${SERVICE_NAME}..."
kubectl delete svc "${SERVICE_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true
log_info "Deleting PVC: ${PVC_NAME}..."
kubectl delete pvc "${PVC_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true

terminate_model_gpu_node "${DEPLOYMENT_NAME}" "${BENCHMARK_NAMESPACE}" "${MODEL_NODE:-}"

log_info "Cleaning up benchmark jobs for model '${MODEL_ID}'..."
kubectl delete jobs -n "${BENCHMARK_NAMESPACE}" \
    -l "app.kubernetes.io/component=benchmark-runner,model=${MODEL_ID}" \
    --ignore-not-found=true 2>/dev/null || true

print_stop_summary "${MODEL_ID}" "${BENCHMARK_NAMESPACE}"
