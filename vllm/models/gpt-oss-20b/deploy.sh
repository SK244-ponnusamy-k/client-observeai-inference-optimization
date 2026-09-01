#!/bin/bash
# ==============================================================================
# vllm/models/gpt-oss-20b/deploy.sh
#
# Deploys openai/gpt-oss-20b on vLLM.
# Can run alongside other models simultaneously (uses unique deployment name).
#
# Usage:
#   cd llm-inference-framework
#   bash vllm/models/gpt-oss-20b/deploy.sh
#
# Port-forward after deploy:
#   kubectl port-forward svc/vllm-gpt-oss-20b 8081:8000 &
#   bash inference/quick-test.sh gpt-oss-20b 8081
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
MODEL_NAME="gpt-oss-20b"
MODEL_FOLDER="gpt-oss-20b"

echo ""
echo "══════════════════════════════════════════════════"
echo "  DEPLOYING — openai/gpt-oss-20b"
echo "  Deployment: ${DEPLOYMENT_NAME}"
echo "  Service   : vllm-gpt-oss-20b:8000"
echo "  Size      : ~13 GB (MXFP4) | VRAM: ~16 GB"
echo "══════════════════════════════════════════════════"
echo ""

# Check model in S3
log_info "Checking model in S3..."
if ! aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log_error "Model not found: s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
    log_error "Download it first: bash model-download/gpt-oss-20b/download.sh"
    exit 1
fi
log_info "Model found in S3."

# Apply PVC
log_info "Applying PVC..."
kubectl apply -f "${SCRIPT_DIR}/pvc.yaml"

# Apply service
log_info "Applying service: vllm-gpt-oss-20b..."
kubectl apply -f "${SCRIPT_DIR}/service.yaml"

# Apply deployment
log_info "Applying deployment: ${DEPLOYMENT_NAME}..."
kubectl apply -f "${SCRIPT_DIR}/deployment.yaml"

# Wait for pod
log_info "Waiting for pod (Karpenter provisioning GPU node ~2-3 min)..."
sleep 10
for i in $(seq 1 24); do
    POD=$(kubectl get pods -l app="${DEPLOYMENT_NAME}" -n "${NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [[ -n "${POD}" ]] && break
    echo -n "."
    sleep 5
done
echo ""
log_info "Pod: ${POD:-not found yet}"

# Wait for Ready
log_info "Waiting for pod to be Ready (model loading from S3 — up to 10 min)..."
kubectl wait "deployment/${DEPLOYMENT_NAME}" \
    --for=condition=Available \
    --timeout=600s \
    -n "${NAMESPACE}" || {
    log_warn "Deployment not ready yet — check logs:"
    log_warn "  kubectl logs deployment/${DEPLOYMENT_NAME} 2>&1 | tail -30"
}

echo ""
echo "══════════════════════════════════════════════════"
echo "  DEPLOYED — ${MODEL_NAME}"
echo "══════════════════════════════════════════════════"
kubectl get pods -l app="${DEPLOYMENT_NAME}"
echo ""
echo "  Port-forward (use port 8081 to avoid conflict with Qwen):"
echo "    kubectl port-forward svc/vllm-gpt-oss-20b 8081:8000 &"
echo "    bash inference/quick-test.sh gpt-oss-20b 8081"
echo ""
echo "  Running models:"
kubectl get deployment -l component=inference-server 2>/dev/null || true
echo "══════════════════════════════════════════════════"
