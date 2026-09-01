#!/bin/bash
# ==============================================================================
# vllm/models/qwen-2.5-0.5b/deploy.sh
#
# Deploys Qwen2.5-0.5B-Instruct on vLLM.
#
# What this does:
#   1. Checks model exists in S3
#   2. Tears down any running vLLM deployment (other model)
#   3. Applies PVC + private service + deployment
#   4. Waits for pod to be Ready
#   5. Prints endpoint info
#
# Usage:
#   cd llm-inference-framework
#   bash vllm/models/qwen-2.5-0.5b/deploy.sh
#
# To expose publicly after deploy:
#   kubectl delete -f vllm/service/service-private.yaml
#   kubectl apply  -f vllm/service/service-public.yaml
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

MODEL_NAME="qwen-0.5b"
MODEL_FOLDER="Qwen2.5-0.5B-Instruct"

echo ""
echo "══════════════════════════════════════════════════"
echo "  DEPLOYING — Qwen2.5-0.5B-Instruct"
echo "  Served as: ${MODEL_NAME}"
echo "══════════════════════════════════════════════════"
echo ""

# ------------------------------------------------------------------------------
# Check model exists in S3
# ------------------------------------------------------------------------------
log_info "Checking model in S3..."
if ! aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log_error "Model not found: s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
    log_error "Download it first: bash model-download/qwen-2.5-0.5b/download.sh"
    exit 1
fi
log_info "Model found in S3."

# ------------------------------------------------------------------------------
# Tear down any existing vLLM deployment
# ------------------------------------------------------------------------------
if kubectl get deployment vllm -n "${NAMESPACE}" >/dev/null 2>&1; then
    CURRENT_MODEL=$(kubectl get deployment vllm -n "${NAMESPACE}" \
        -o jsonpath='{.metadata.labels.model}' 2>/dev/null || echo "unknown")
    if [[ "${CURRENT_MODEL}" != "qwen-2.5-0.5b" ]]; then
        log_warn "Found existing deployment (model: ${CURRENT_MODEL}) — removing..."
        kubectl delete deployment vllm -n "${NAMESPACE}"
        # Wait for pod to terminate and release PVC
        kubectl wait --for=delete pod -l app=vllm -n "${NAMESPACE}" --timeout=120s 2>/dev/null || true
        log_info "Existing deployment removed."
    else
        log_warn "Qwen deployment already running — redeploying..."
        kubectl delete deployment vllm -n "${NAMESPACE}"
        kubectl wait --for=delete pod -l app=vllm -n "${NAMESPACE}" --timeout=120s 2>/dev/null || true
    fi
fi

# ------------------------------------------------------------------------------
# Apply PVC
# ------------------------------------------------------------------------------
log_info "Applying PVC..."
kubectl apply -f "${SCRIPT_DIR}/pvc.yaml"

# ------------------------------------------------------------------------------
# Apply service (private by default)
# ------------------------------------------------------------------------------
log_info "Applying service (private ClusterIP)..."
# Remove public service if exists
kubectl delete -f "${FRAMEWORK_ROOT}/vllm/service/service-public.yaml" 2>/dev/null || true
kubectl apply -f "${FRAMEWORK_ROOT}/vllm/service/service-private.yaml"

# ------------------------------------------------------------------------------
# Apply deployment
# ------------------------------------------------------------------------------
log_info "Applying deployment..."
kubectl apply -f "${SCRIPT_DIR}/deployment.yaml"

# ------------------------------------------------------------------------------
# Wait for pod to be scheduled
# ------------------------------------------------------------------------------
log_info "Waiting for pod to be scheduled (Karpenter provisioning GPU node)..."
sleep 10
for i in $(seq 1 24); do
    POD=$(kubectl get pods -l app=vllm -n "${NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [[ -n "${POD}" ]] && break
    echo -n "."
    sleep 5
done
echo ""
log_info "Pod: ${POD:-not found yet}"

# ------------------------------------------------------------------------------
# Wait for Ready
# ------------------------------------------------------------------------------
log_info "Waiting for pod to be Ready (model loading from S3)..."
kubectl wait deployment/vllm \
    --for=condition=Available \
    --timeout=600s \
    -n "${NAMESPACE}" || {
    log_warn "Deployment not ready yet — check logs:"
    log_warn "  kubectl logs deployment/vllm 2>&1 | tail -30"
}

# ------------------------------------------------------------------------------
# Done
# ------------------------------------------------------------------------------
echo ""
echo "══════════════════════════════════════════════════"
echo "  DEPLOYED — ${MODEL_NAME}"
echo "══════════════════════════════════════════════════"
kubectl get pods -l app=vllm
echo ""
echo "  Port-forward for local test:"
echo "    kubectl port-forward svc/vllm-inference 8080:8000 &"
echo "    bash inference/quick-test.sh qwen-0.5b"
echo ""
echo "  Switch to public:"
echo "    kubectl delete -f vllm/service/service-private.yaml"
echo "    kubectl apply  -f vllm/service/service-public.yaml"
echo "══════════════════════════════════════════════════"
