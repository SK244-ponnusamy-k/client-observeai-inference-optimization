#!/bin/bash
# ==============================================================================
# vllm/models/qwen3-5-4b-neuron/stop.sh
#
# Stops qwen3-5-4b-neuron (Neuron). Deletes deployment/service and frees the Trainium node.
# Friendly wrapper:  oai stop qwen3-5-4b-neuron
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"
source "${SCRIPT_DIR}/model.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }

USE_MANAGED_NG="false"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --managed-ng) USE_MANAGED_NG="true"; shift ;;
        *) shift ;;
    esac
done

echo ""
echo "=============================================================="
echo "  FULL STOP (Neuron) - ${MODEL_ID}"
echo "=============================================================="
echo ""

if [[ "${OAI_ASSUME_YES:-false}" != "true" ]]; then
    read -r -p "Proceed? [y/N]: " CONFIRM
    [[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }
fi

log_info "Deleting deployment: ${DEPLOYMENT_NAME}..."
kubectl delete deployment "${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true
log_info "Deleting service: ${SERVICE_NAME}..."
kubectl delete svc "${SERVICE_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true

# Auto-detect managed node group
if [[ -n "${TRN2_NODEGROUP_NAME:-}" ]]; then
    NG_STATUS=$(aws eks describe-nodegroup --cluster-name "${CLUSTER_NAME}" \
        --nodegroup-name "${TRN2_NODEGROUP_NAME}" --region "${AWS_REGION}" \
        --query 'nodegroup.status' --output text 2>/dev/null || true)
    [[ -n "${NG_STATUS}" && "${NG_STATUS}" != "None" ]] && USE_MANAGED_NG="true"
fi

if [[ "${USE_MANAGED_NG}" == "true" && -n "${TRN2_NODEGROUP_NAME:-}" ]]; then
    log_info "Scaling managed node group ${TRN2_NODEGROUP_NAME} to 0 (stops billing)..."
    aws eks update-nodegroup-config --cluster-name "${CLUSTER_NAME}" \
        --nodegroup-name "${TRN2_NODEGROUP_NAME}" \
        --scaling-config minSize=0,maxSize=1,desiredSize=0 --region "${AWS_REGION}" >/dev/null || \
        log_warn "Could not scale node group - do it manually if needed."
fi

echo ""
log_info "Stopped. S3 weights preserved at s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
