#!/bin/bash
# ==============================================================================
# vllm/stop.sh
#
# Stops ALL running vLLM models and releases ALL GPU nodes.
# Run this and close your laptop — everything expensive stops.
#
# What this does:
#   1. Finds and deletes every inference-server deployment in oai-infopt
#   2. Deletes ALL GPU node claims — stops billing immediately
#   3. Cleans up all benchmark jobs
#   4. Confirms nothing expensive remains
#
# What this does NOT touch:
#   - S3 model weights
#   - Monitoring stack (Grafana/Prometheus on CPU nodes — ~$0.15/hr)
#
# Usage:
#   cd llm-inference-framework
#   bash vllm/stop.sh                        # stop ALL models
#
# To stop a single model only:
#   bash vllm/models/qwen-2.5-0.5b/stop.sh
#   bash vllm/models/gpt-oss-20b/stop.sh
#
# To also delete the entire cluster:
#   bash cluster/teardown.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

# Source shared GPU termination helper
source "${SCRIPT_DIR}/lib/gpu-terminate.sh"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }

echo ""
echo "══════════════════════════════════════════════════"
echo "  STOP ALL vLLM MODELS"
echo "  Cluster   : ${CLUSTER_NAME}"
echo "  Namespace : ${BENCHMARK_NAMESPACE}"
echo "══════════════════════════════════════════════════"
echo ""
echo "  This will:"
echo "    • Delete ALL inference-server deployments"
echo "    • Terminate ALL GPU nodes (stops billing)"
echo "    • Clean up all benchmark jobs"
echo "    • Leave monitoring stack running (~\$0.15/hr)"
echo ""

# ── Discover running deployments ────────────────────────────────────────────
RUNNING=$(kubectl get deployment \
    -l app.kubernetes.io/component=inference-server \
    -n "${BENCHMARK_NAMESPACE}" \
    -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)

if [[ -z "${RUNNING}" ]]; then
    log_warn "No inference-server deployments found in ${BENCHMARK_NAMESPACE}."
else
    log_info "Found deployments: ${RUNNING}"
fi

# Show current GPU nodes before confirming
GPU_NODECLAIMS=$(kubectl get nodeclaims --no-headers 2>/dev/null | \
    awk '{print $1, $2}' | grep -E "g4|g5|g6|p3|p4|p5|inf" || true)
if [[ -n "${GPU_NODECLAIMS}" ]]; then
    log_info "GPU nodes currently running (billing):"
    echo "${GPU_NODECLAIMS}" | while read -r line; do echo "    ${line}"; done
fi

echo ""
read -r -p "Stop everything? [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

# ── Step 1: Delete all inference-server deployments + services + PVCs ────────
if [[ -n "${RUNNING}" ]]; then
    log_info "Deleting all inference-server deployments..."
    kubectl delete deployment \
        -l app.kubernetes.io/component=inference-server \
        -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true

    log_info "Deleting all inference-server services..."
    kubectl delete svc \
        -l app.kubernetes.io/component=inference-server \
        -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true

    log_info "Deleting all PVCs in ${BENCHMARK_NAMESPACE}..."
    kubectl delete pvc --all \
        -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
fi

# ── Step 2: Terminate ALL GPU nodes ─────────────────────────────────────────
terminate_gpu_nodes

# ── Step 3: Clean up ALL benchmark jobs ─────────────────────────────────────
log_info "Cleaning up all jobs in ${BENCHMARK_NAMESPACE}..."
ALL_JOBS=$(kubectl get jobs -n "${BENCHMARK_NAMESPACE}" \
    --no-headers 2>/dev/null | awk '{print $1}' || true)
if [[ -n "${ALL_JOBS}" ]]; then
    echo "${ALL_JOBS}" | xargs -r kubectl delete job \
        -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true
fi

# ── Step 4: Final status ─────────────────────────────────────────────────────
print_stop_summary "ALL MODELS" "${BENCHMARK_NAMESPACE}"
