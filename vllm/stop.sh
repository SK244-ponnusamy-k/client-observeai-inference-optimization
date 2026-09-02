#!/bin/bash
# ==============================================================================
# vllm/stop.sh
#
# Stops ALL running vLLM deployments and releases their resources.
# GPU nodes terminate automatically within ~10 minutes (Karpenter).
# S3 model files are NOT affected.
# Cluster keeps running (control plane ~$0.10/hr continues).
#
# Usage:
#   cd llm-inference-framework
#   bash vllm/stop.sh                       # stop ALL models
#
# To stop a single model only:
#   bash vllm/models/qwen-2.5-0.5b/stop.sh
#   bash vllm/models/gpt-oss-20b/stop.sh
#
# To revert a public LB to private (no downtime):
#   kubectl apply -f vllm/models/<model>/service/service-private.yaml
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
echo "  STOP ALL vLLM DEPLOYMENTS"
echo "  Cluster : ${CLUSTER_NAME}"
echo "  GPU nodes terminate in ~10 min after stop"
echo "  S3 models are NOT affected"
echo "══════════════════════════════════════════════════"
echo ""

# ------------------------------------------------------------------------------
# Discover all running inference deployments
# ------------------------------------------------------------------------------
RUNNING=$(kubectl get deployment -l app.kubernetes.io/component=inference-server \
    -n "${BENCHMARK_NAMESPACE}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)

if [[ -z "${RUNNING}" ]]; then
    log_warn "No vLLM deployments found — nothing to stop."
    exit 0
fi

log_info "Found deployments: ${RUNNING}"

read -r -p "Stop ALL inference deployments? [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

# ------------------------------------------------------------------------------
# Stop each model by delegating to its own stop.sh
# (non-interactive: answers 'y' automatically since user already confirmed above)
# ------------------------------------------------------------------------------
MODELS_DIR="${SCRIPT_DIR}/models"

for MODEL_DIR in "${MODELS_DIR}"/*/; do
    MODEL=$(basename "${MODEL_DIR}")
    STOP_SCRIPT="${MODEL_DIR}stop.sh"

    if [[ -f "${STOP_SCRIPT}" ]]; then
        log_info "Stopping model: ${MODEL}..."
        # Pass 'y' automatically — user already confirmed above
        echo "y" | bash "${STOP_SCRIPT}" || log_warn "stop.sh for ${MODEL} returned non-zero (may already be stopped)"
    else
        log_warn "No stop.sh found for ${MODEL} — skipping."
    fi
done

# ------------------------------------------------------------------------------
# Show remaining resources
# ------------------------------------------------------------------------------
echo ""
log_info "Remaining pods:"
kubectl get pods -n "${BENCHMARK_NAMESPACE}" 2>/dev/null || true

echo ""
log_info "Remaining nodes (GPU nodes will terminate in ~10 min):"
kubectl get nodes 2>/dev/null || true

echo ""
echo "══════════════════════════════════════════════════"
echo "  ALL MODELS STOPPED"
echo "  Deleted : deployments, services, PVCs"
echo "  Kept    : cluster, S3 models, IAM Roles"
echo ""
echo "  To redeploy a model:"
echo "    bash vllm/models/qwen-2.5-0.5b/deploy.sh"
echo "    bash vllm/models/gpt-oss-20b/deploy.sh"
echo ""
echo "  To delete cluster too:"
echo "    bash cluster/teardown.sh"
echo "══════════════════════════════════════════════════"
