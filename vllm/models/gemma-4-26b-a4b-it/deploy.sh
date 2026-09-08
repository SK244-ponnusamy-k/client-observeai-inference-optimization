#!/bin/bash
# ==============================================================================
# vllm/models/gemma-4-26b-a4b-it/deploy.sh
#
# Generic deploy — model-specific names come from model.env. Hardware chosen per
# run so the model can be benchmarked across the g5/g6/g6e matrix.
#
# Usage:
#   bash vllm/models/gemma-4-26b-a4b-it/deploy.sh                    # defaults (g6e, 4-bit)
#   bash vllm/models/gemma-4-26b-a4b-it/deploy.sh --hw g5.2xlarge    # 4-bit on A10G
#   bash vllm/models/gemma-4-26b-a4b-it/deploy.sh --hw g6.2xlarge    # 4-bit on L4
#   bash vllm/models/gemma-4-26b-a4b-it/deploy.sh --hw g6e.12xlarge --tp 4  # bf16 quality-ceiling
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"
source "${SCRIPT_DIR}/model.env"

: "${VLLM_IMAGE:?ERROR: VLLM_IMAGE is not set. Ensure config/config.env is sourced and exports VLLM_IMAGE with a pinned digest.}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

VALIDATE="false"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hw)       NODE_INSTANCE_TYPE="$2"; shift 2 ;;
        --tp)       TP_SIZE="$2"; shift 2 ;;
        --validate) VALIDATE="true"; shift ;;
        *) log_warn "Unknown arg: $1"; shift ;;
    esac
done
GPU_COUNT="${TP_SIZE}"
export NODE_INSTANCE_TYPE TP_SIZE GPU_COUNT MODEL_FOLDER SERVED_NAME

echo ""
echo "══════════════════════════════════════════════════"
echo "  DEPLOYING — ${MODEL_HF_ID}"
echo "  Deployment    : ${DEPLOYMENT_NAME}"
echo "  Service       : ${SERVICE_NAME}:8000"
echo "  Namespace     : ${BENCHMARK_NAMESPACE}"
echo "  Instance type : ${NODE_INSTANCE_TYPE}   (TP=${TP_SIZE}, GPUs=${GPU_COUNT})"
echo "  Checkpoint    : ${MODEL_FOLDER}"
echo "══════════════════════════════════════════════════"
echo ""

log_info "Checking model in S3..."
if ! aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log_error "Model not found: s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
    log_error "Download it first: bash model-download/gemma-4-26b-a4b-it/download.sh"
    exit 1
fi
log_info "Model found in S3."

log_info "Applying PVC..."
kubectl apply -f "${SCRIPT_DIR}/pvc.yaml"

log_info "Applying service: ${SERVICE_NAME}..."
kubectl apply -f "${SCRIPT_DIR}/service.yaml"

log_info "Applying deployment: ${DEPLOYMENT_NAME} on ${NODE_INSTANCE_TYPE}..."
export MODEL_BUCKET BENCHMARK_NAMESPACE VLLM_IMAGE
envsubst '${MODEL_BUCKET} ${BENCHMARK_NAMESPACE} ${VLLM_IMAGE} ${MODEL_FOLDER} ${SERVED_NAME} ${NODE_INSTANCE_TYPE} ${TP_SIZE} ${GPU_COUNT}' \
    < "${SCRIPT_DIR}/deployment.yaml" | kubectl apply -f -

log_info "Waiting for pod (Karpenter provisioning ${NODE_INSTANCE_TYPE} ~3-5 min)..."
sleep 15
for i in $(seq 1 24); do
    POD=$(kubectl get pods -l app="${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [[ -n "${POD}" ]] && break
    echo -n "."
    sleep 5
done
echo ""
log_info "Pod: ${POD:-not found yet}"

log_info "Waiting for Ready (4-bit MoE loading from S3 — up to 12 min)..."
kubectl wait "deployment/${DEPLOYMENT_NAME}" \
    --for=condition=Available --timeout=720s -n "${BENCHMARK_NAMESPACE}" || {
    log_warn "Deployment not ready yet — check logs:"
    log_warn "  kubectl logs deployment/${DEPLOYMENT_NAME} -n ${BENCHMARK_NAMESPACE} | tail -50"
}

echo ""
echo "══════════════════════════════════════════════════"
echo "  DEPLOYED — ${MODEL_ID} on ${NODE_INSTANCE_TYPE}"
echo "══════════════════════════════════════════════════"
kubectl get pods -l app="${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" -o wide
echo ""
echo "  Port-forward:"
echo "    kubectl port-forward svc/${SERVICE_NAME} ${PORT_FORWARD_PORT}:8000 -n ${BENCHMARK_NAMESPACE} &"
echo "    curl http://localhost:${PORT_FORWARD_PORT}/health"
echo "══════════════════════════════════════════════════"

if [[ "${VALIDATE}" == "true" ]]; then
    echo ""
    log_info "Starting post-deploy validation (--validate flag set)..."
    bash "${SCRIPT_DIR}/../post-deploy-validate.sh" \
        --model "${MODEL_ID}" \
        --manifest "${MANIFEST_PATH}" \
        --endpoint "http://${SERVICE_NAME}:8000"
fi
