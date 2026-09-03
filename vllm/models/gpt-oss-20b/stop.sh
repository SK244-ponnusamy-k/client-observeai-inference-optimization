#!/bin/bash
# ==============================================================================
# vllm/models/gpt-oss-20b/stop.sh
#
# Full teardown — run this and close your laptop. Everything stops.
#
# What this does:
#   1. Deletes the vLLM deployment + service + PVC
#   2. Deletes ALL GPU node claims (nodeclaims) — stops billing immediately
#   3. Deletes GPU nodes by any known label/nodepool name
#   4. Cleans up leftover benchmark jobs in oai-infopt namespace
#   5. Confirms nothing expensive is left running
#
# What this does NOT touch:
#   - S3 model weights (safe, no change)
#   - Monitoring stack (Grafana/Prometheus on cheap CPU nodes — ~$0.15/hr)
#   - CPU system nodes (needed for monitoring)
#
# Usage:
#   cd llm-inference-framework
#   bash vllm/models/gpt-oss-20b/stop.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"
source "${SCRIPT_DIR}/model.env"

# Source shared GPU termination helper
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
echo "    • Terminate ALL GPU nodes (stops billing)"
echo "    • Clean up benchmark jobs"
echo "    • Leave monitoring stack running (~\$0.15/hr)"
echo ""

read -r -p "Proceed? [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

# ── Step 1: Capture the GPU node BEFORE deleting the deployment ──────────────
# Pod disappears once deployment is deleted — find the node first.
log_info "Looking up node for deployment: ${DEPLOYMENT_NAME}..."
MODEL_NODE=$(kubectl get pod \
    -l "app=${DEPLOYMENT_NAME}" \
    -n "${BENCHMARK_NAMESPACE}" \
    -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)

if [[ -z "${MODEL_NODE}" ]]; then
    # fallback label
    MODEL_NODE=$(kubectl get pod \
        -l "app.kubernetes.io/name=${DEPLOYMENT_NAME}" \
        -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)
fi

if [[ -n "${MODEL_NODE}" ]]; then
    log_info "Model is running on node: ${MODEL_NODE}"
else
    log_warn "Pod not found — node will be located via nodeclaim instance type."
fi

# ── Step 2: Delete vLLM deployment ──────────────────────────────────────────
log_info "Deleting deployment: ${DEPLOYMENT_NAME}..."
kubectl delete deployment "${DEPLOYMENT_NAME}" \
    -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true

log_info "Deleting service: ${SERVICE_NAME}..."
kubectl delete svc "${SERVICE_NAME}" \
    -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true

log_info "Deleting PVC: ${PVC_NAME}..."
kubectl delete pvc "${PVC_NAME}" \
    -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true

# ── Step 3: Terminate THIS model's GPU node ──────────────────────────────────
# Uses the node name captured before deployment deletion.
# Only deletes that specific node — leaves qwen's node untouched if running.
terminate_model_gpu_node "${DEPLOYMENT_NAME}" "${BENCHMARK_NAMESPACE}" "${MODEL_NODE:-}"

# ── Step 3: Clean up benchmark jobs ─────────────────────────────────────────
log_info "Cleaning up benchmark jobs in ${BENCHMARK_NAMESPACE}..."
kubectl delete jobs -n "${BENCHMARK_NAMESPACE}" \
    -l "app.kubernetes.io/component=benchmark-runner" \
    --ignore-not-found=true 2>/dev/null || true

STALE_JOBS=$(kubectl get jobs -n "${BENCHMARK_NAMESPACE}" \
    --no-headers 2>/dev/null | awk '{print $1}' || true)
if [[ -n "${STALE_JOBS}" ]]; then
    log_info "Deleting remaining jobs in ${BENCHMARK_NAMESPACE}..."
    echo "${STALE_JOBS}" | xargs -r kubectl delete job \
        -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true
fi

# ── Step 4: Final status ─────────────────────────────────────────────────────
print_stop_summary "${MODEL_ID}" "${BENCHMARK_NAMESPACE}"
