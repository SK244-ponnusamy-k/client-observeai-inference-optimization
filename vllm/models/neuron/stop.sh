#!/bin/bash
# ==============================================================================
# vllm/models/neuron/stop.sh
#
# Full teardown of the single-instance Neuron gpt-oss-20b deployment — deletes
# deployment + service, terminates the Trainium node (stops billing), cleans up
# this model's benchmark jobs. Leaves the cache PVC and monitoring untouched
# (delete the PVC manually if you want to discard the compiled NEFFs/weights).
#
# Usage:
#   bash vllm/models/neuron/stop.sh
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
echo "  FULL STOP (Neuron) — ${MODEL_ID}"
echo "  Deployment : ${DEPLOYMENT_NAME}"
echo "  Namespace  : ${BENCHMARK_NAMESPACE}"
echo "══════════════════════════════════════════════════"
echo ""
echo "  This will:"
echo "    • Delete vLLM-Neuron deployment + service"
echo "    • Terminate the Trainium node (stops billing)"
echo "    • Clean up benchmark jobs for this model only"
echo "    • KEEP the cache PVC (${PVC_NAME}) — delete manually to drop NEFFs/weights"
echo ""

read -r -p "Proceed? [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

log_info "Looking up node for deployment: ${DEPLOYMENT_NAME}..."
MODEL_NODE=$(kubectl get pod -l "app=${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" \
    -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)
[[ -n "${MODEL_NODE}" ]] && log_info "Model is running on node: ${MODEL_NODE}" \
    || log_warn "Pod not found — node located via nodeclaim instance type."

log_info "Deleting deployment: ${DEPLOYMENT_NAME}..."
kubectl delete deployment "${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true
log_info "Deleting service: ${SERVICE_NAME}..."
kubectl delete svc "${SERVICE_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true

# Reuse the shared GPU-node terminator (works for any Karpenter-managed node).
terminate_model_gpu_node "${DEPLOYMENT_NAME}" "${BENCHMARK_NAMESPACE}" "${MODEL_NODE:-}"

log_info "Cleaning up benchmark jobs for model '${MODEL_ID}'..."
kubectl delete jobs -n "${BENCHMARK_NAMESPACE}" \
    -l "app.kubernetes.io/component=benchmark-runner,model=${MODEL_ID}" \
    --ignore-not-found=true 2>/dev/null || true

print_stop_summary "${MODEL_ID}" "${BENCHMARK_NAMESPACE}"
echo "  NOTE: cache PVC '${PVC_NAME}' kept. Delete with:"
echo "    kubectl delete pvc ${PVC_NAME} -n ${BENCHMARK_NAMESPACE}"
