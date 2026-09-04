#!/bin/bash
# ==============================================================================
# vllm/models/qwen-2.5-0.5b/deploy.sh
#
# All model-specific names come from model.env — this script is generic.
#
# Usage:
#   cd llm-inference-framework
#   bash vllm/models/qwen-2.5-0.5b/deploy.sh
#   bash vllm/models/qwen-2.5-0.5b/deploy.sh --validate
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"
source "${SCRIPT_DIR}/model.env"

# Fail fast if VLLM_IMAGE is unset or empty — prevents accidental 'latest' pull.
# Set in config/config.env. Value must include a digest for supply-chain safety.
# Added 2026-09-03 — fixes Checkov CKV_K8S_14 / CKV_K8S_43 guard.
: "${VLLM_IMAGE:?ERROR: VLLM_IMAGE is not set. Ensure config/config.env is sourced and exports VLLM_IMAGE with a pinned digest.}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

VALIDATE="false"
for arg in "$@"; do
    [[ "${arg}" == "--validate" ]] && VALIDATE="true"
done

echo ""
echo "══════════════════════════════════════════════════"
echo "  DEPLOYING — ${MODEL_HF_ID}"
echo "  Deployment: ${DEPLOYMENT_NAME}"
echo "  Service   : ${SERVICE_NAME}:8000"
echo "  Namespace : ${BENCHMARK_NAMESPACE}"
echo "══════════════════════════════════════════════════"
echo ""

# Check model in S3
log_info "Checking model in S3..."
if ! aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log_error "Model not found: s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
    log_error "Download it first: bash model-download/qwen-2.5-0.5b/download.sh"
    exit 1
fi
log_info "Model found in S3."

# Apply PVC
log_info "Applying PVC..."
kubectl apply -f "${SCRIPT_DIR}/pvc.yaml"

# Apply service
log_info "Applying service: ${SERVICE_NAME}..."
kubectl apply -f "${SCRIPT_DIR}/service.yaml"

# Apply deployment — envsubst injects MODEL_BUCKET, BENCHMARK_NAMESPACE, VLLM_IMAGE
log_info "Applying deployment: ${DEPLOYMENT_NAME}..."
export MODEL_BUCKET BENCHMARK_NAMESPACE VLLM_IMAGE
envsubst < "${SCRIPT_DIR}/deployment.yaml" | kubectl apply -f -

# Wait for pod
log_info "Waiting for pod (Karpenter provisioning GPU node)..."
sleep 10
for i in $(seq 1 24); do
    POD=$(kubectl get pods -l app="${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [[ -n "${POD}" ]] && break
    echo -n "."
    sleep 5
done
echo ""
log_info "Pod: ${POD:-not found yet}"

# Wait for Ready
log_info "Waiting for pod to be Ready (model loading from S3)..."
kubectl wait "deployment/${DEPLOYMENT_NAME}" \
    --for=condition=Available \
    --timeout=600s \
    -n "${BENCHMARK_NAMESPACE}" || {
    log_warn "Deployment not ready yet — check logs:"
    log_warn "  kubectl logs deployment/${DEPLOYMENT_NAME} -n ${BENCHMARK_NAMESPACE} | tail -30"
}

echo ""
echo "══════════════════════════════════════════════════"
echo "  DEPLOYED — ${MODEL_ID}"
echo "══════════════════════════════════════════════════"
kubectl get pods -l app="${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}"
echo ""
echo "  Port-forward:"
echo "    kubectl port-forward svc/${SERVICE_NAME} ${PORT_FORWARD_PORT}:8000 -n ${BENCHMARK_NAMESPACE} &"
echo "    curl http://localhost:${PORT_FORWARD_PORT}/health"
echo ""
echo "  Running models:"
kubectl get deployment -n "${BENCHMARK_NAMESPACE}" 2>/dev/null || true
echo "══════════════════════════════════════════════════"

# Auto-validate if --validate flag passed
if [[ "${VALIDATE}" == "true" ]]; then
    echo ""
    log_info "Starting post-deploy validation (--validate flag set)..."
    bash "${SCRIPT_DIR}/../post-deploy-validate.sh" \
        --model "${MODEL_ID}" \
        --manifest "${MANIFEST_PATH}" \
        --endpoint "http://${SERVICE_NAME}:8000"
fi
