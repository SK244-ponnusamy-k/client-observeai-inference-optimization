#!/bin/bash
# ==============================================================================
# scripts/post-deploy-validate.sh
#
# Automatic post-deploy validation pipeline.
# Called by deploy.sh --validate after the model is Ready.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }
log_step()  { echo ""; echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; \
              echo -e "${GREEN}  $1${NC}"; \
              echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# Parse arguments
MODEL_NAME=""
MANIFEST_PATH=""
ENDPOINT=""
SKIP_BATCH="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)     MODEL_NAME="$2";    shift 2 ;;
        --manifest)  MANIFEST_PATH="$2"; shift 2 ;;
        --endpoint)  ENDPOINT="$2";      shift 2 ;;
        --skip-batch) SKIP_BATCH="true"; shift ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL_NAME}" || -z "${MANIFEST_PATH}" || -z "${ENDPOINT}" ]]; then
    log_error "Usage: $0 --model <name> --manifest <path> --endpoint <url>"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
REALTIME_JOB="oai-infopt-validate-realtime-${TIMESTAMP}"
BATCH_JOB="oai-infopt-validate-batch-${TIMESTAMP}"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  POST-DEPLOY VALIDATION"
echo "══════════════════════════════════════════════════════════════"
echo "  Model    : ${MODEL_NAME}"
echo "  Manifest : ${MANIFEST_PATH}"
echo "  Endpoint : ${ENDPOINT}"
echo "  Namespace: ${BENCHMARK_NAMESPACE}"
echo "══════════════════════════════════════════════════════════════"

# Step 1 — Wait for /health
log_step "Step 1 — Waiting for model health endpoint"

LOCAL_PORT=18080
SVC_NAME="oai-infopt-vllm-${MODEL_NAME}"

log_info "Port-forwarding ${SVC_NAME}:8000 → localhost:${LOCAL_PORT}..."
kubectl port-forward "svc/${SVC_NAME}" "${LOCAL_PORT}:8000" \
    -n "${BENCHMARK_NAMESPACE}" &
PF_PID=$!
trap "kill ${PF_PID} 2>/dev/null || true" EXIT

sleep 3

HEALTH_TIMEOUT=300
HEALTH_START=$(date +%s)
while true; do
    if curl -sf "http://localhost:${LOCAL_PORT}/health" >/dev/null 2>&1; then
        log_info "Model is healthy and serving."
        break
    fi
    NOW=$(date +%s)
    ELAPSED=$((NOW - HEALTH_START))
    if [[ ${ELAPSED} -gt ${HEALTH_TIMEOUT} ]]; then
        log_error "Model health check timed out after ${HEALTH_TIMEOUT}s"
        exit 1
    fi
    echo -n "."
    sleep 5
done
echo ""

MODELS_RESPONSE=$(curl -sf "http://localhost:${LOCAL_PORT}/v1/models" 2>/dev/null || echo "{}")
log_info "Models response: ${MODELS_RESPONSE}"

# Step 2 — Create benchmark ConfigMaps
log_step "Step 2 — Creating benchmark ConfigMaps"

log_info "Creating benchmark script ConfigMap..."
kubectl create configmap oai-infopt-benchmark-script \
    --from-file=load-test.py="${FRAMEWORK_ROOT}/inference/load-test.py" \
    --namespace "${BENCHMARK_NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -

log_info "Creating manifest ConfigMap..."
kubectl create configmap oai-infopt-benchmark-manifest \
    --from-file=manifest.yaml="${FRAMEWORK_ROOT}/${MANIFEST_PATH}" \
    --namespace "${BENCHMARK_NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -

log_info "Creating realtime profile ConfigMap..."
kubectl create configmap oai-infopt-benchmark-profile-realtime \
    --from-file=profile.yaml="${FRAMEWORK_ROOT}/configs/workload_profiles/realtime_v1.yaml" \
    --namespace "${BENCHMARK_NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -

log_info "Creating batch profile ConfigMap..."
kubectl create configmap oai-infopt-benchmark-profile-batch \
    --from-file=profile.yaml="${FRAMEWORK_ROOT}/configs/workload_profiles/batch_v1.yaml" \
    --namespace "${BENCHMARK_NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -

log_info "ConfigMaps created."

# Step 3 — Run Realtime benchmark
log_step "Step 3 — Submitting realtime benchmark Job"
bash "${FRAMEWORK_ROOT}/inference/run-benchmark.sh" --model "${MODEL_NAME}" --profile realtime || true

# Step 4 — Run Batch benchmark (unless --skip-batch)
if [[ "${SKIP_BATCH}" == "false" ]]; then
    log_step "Step 4 — Submitting batch benchmark Job"
    bash "${FRAMEWORK_ROOT}/inference/run-benchmark.sh" --model "${MODEL_NAME}" --profile batch || true
fi

log_info "Validation pipeline finished."
