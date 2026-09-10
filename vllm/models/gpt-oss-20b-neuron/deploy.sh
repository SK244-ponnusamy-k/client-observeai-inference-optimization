#!/bin/bash
# ==============================================================================
# vllm/models/gpt-oss-20b-neuron/deploy.sh
#
# Deploys gpt-oss-20b on AWS Trainium (trn1) via vLLM-Neuron (Flow B).
#
# Usage:
#   bash vllm/models/gpt-oss-20b-neuron/deploy.sh
#   bash vllm/models/gpt-oss-20b-neuron/deploy.sh --cores 16          # override NEURON_CORES
#   bash vllm/models/gpt-oss-20b-neuron/deploy.sh --compile           # run compile job first
#   bash vllm/models/gpt-oss-20b-neuron/deploy.sh --compile --validate
#
# Flags:
#   --compile   Run compile-job.yaml before deploying (idempotent — skips if
#               NEFF cache already exists in S3). Use on first deploy or after
#               changing SEQ_LEN / BATCH / NEURON_CORES.
#   --cores N   Override NEURON_CORES from model.env (must match compile-time value).
#   --validate  Run a smoke-test after the deployment is ready.
#
# Prerequisites:
#   1. config/config.env is populated, NEURON_VLLM_IMAGE and NEURON_COMPILE_IMAGE set.
#   2. cluster/neuron-nodepool.yaml has been applied (one-time cluster setup).
#   3. Model bf16 weights exist at s3://${MODEL_BUCKET}/gpt-oss-20b/
#      (download first if not: bash model-download/gpt-oss-20b/download.sh)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"
source "${SCRIPT_DIR}/model.env"

: "${NEURON_VLLM_IMAGE:?ERROR: NEURON_VLLM_IMAGE is not set. Set it in config/config.env.}"
: "${NEURON_COMPILE_IMAGE:?ERROR: NEURON_COMPILE_IMAGE is not set. Set it in config/config.env.}"
: "${MODEL_BUCKET:?ERROR: MODEL_BUCKET is not set. Set it in config/config.env.}"
: "${BENCHMARK_NAMESPACE:?ERROR: BENCHMARK_NAMESPACE is not set. Set it in config/config.env.}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

RUN_COMPILE="false"
VALIDATE="false"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --compile)  RUN_COMPILE="true"; shift ;;
        --cores)    NEURON_CORES="$2"; shift 2 ;;
        --validate) VALIDATE="true"; shift ;;
        *) log_warn "Unknown argument: $1"; shift ;;
    esac
done
export NEURON_CORES MODEL_FOLDER SERVED_NAME NEURON_VLLM_IMAGE NEURON_COMPILE_IMAGE

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  DEPLOYING (Neuron / trn1) — ${MODEL_HF_ID}"
echo "  Deployment    : ${DEPLOYMENT_NAME}"
echo "  Service       : ${SERVICE_NAME}:8000"
echo "  Namespace     : ${BENCHMARK_NAMESPACE}"
echo "  Instance type : ${NEURON_INSTANCE_TYPE}   (NeuronCores=${NEURON_CORES})"
echo "  NEFF cache    : s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
echo "  Compile first : ${RUN_COMPILE}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Verify bf16 source weights exist in S3 ───────────────────────────
# Always check SOURCE_FOLDER (bf16 weights). MODEL_FOLDER (NEFF cache) won't
# exist yet before the first compile — that's expected, not an error.
log_info "Checking bf16 source weights in S3: s3://${MODEL_BUCKET}/${SOURCE_FOLDER}/"
if ! aws s3 ls "s3://${MODEL_BUCKET}/${SOURCE_FOLDER}/" \
        --region "${AWS_REGION}" >/dev/null 2>&1; then
    log_error "Source weights not found: s3://${MODEL_BUCKET}/${SOURCE_FOLDER}/"
    log_error "Download them first: bash model-download/gpt-oss-20b/download.sh"
    exit 1
fi
log_info "Source weights found in S3."

# ── Step 2: Optionally run compile job ───────────────────────────────────────
# Use --compile on the FIRST deploy. After that, the NEFF cache exists in S3
# and every subsequent deploy skips this step entirely (fast pod startup).
# Without --compile: vLLM-Neuron auto-compiles inside the pod on first start
# (60-90 min) but the result is lost on pod restart — recompiles every time.
if [[ "${RUN_COMPILE}" == "true" ]]; then
    log_info "Submitting compile job (idempotent — skips if NEFF cache exists)..."
    envsubst '${BENCHMARK_NAMESPACE} ${DOWNLOAD_SERVICE_ACCOUNT} ${MODEL_BUCKET} ${AWS_REGION} ${NEURON_COMPILE_IMAGE}' \
        < "${SCRIPT_DIR}/compile-job.yaml" | kubectl apply -f -

    log_info "Waiting for compile job to complete (can take up to 12 hours for MoE)..."
    kubectl wait job/oai-infopt-neuron-compile-gpt-oss-20b \
        --for=condition=Complete \
        --timeout=43200s \
        -n "${BENCHMARK_NAMESPACE}" || {
        log_error "Compile job did not complete successfully."
        log_error "Check logs: kubectl logs -f job/oai-infopt-neuron-compile-gpt-oss-20b -n ${BENCHMARK_NAMESPACE}"
        exit 1
    }
    log_info "Compile job complete."
fi

# ── Step 3: Apply service ─────────────────────────────────────────────────────
log_info "Applying service: ${SERVICE_NAME}..."
kubectl apply -f "${SCRIPT_DIR}/service.yaml"

# ── Step 4: Apply deployment ──────────────────────────────────────────────────
log_info "Applying deployment: ${DEPLOYMENT_NAME} (NeuronCores=${NEURON_CORES})..."
export MODEL_BUCKET BENCHMARK_NAMESPACE NEURON_VLLM_IMAGE MODEL_FOLDER MODEL_HF_ID SERVED_NAME NEURON_CORES
envsubst '${MODEL_BUCKET} ${BENCHMARK_NAMESPACE} ${NEURON_VLLM_IMAGE} ${MODEL_FOLDER} ${MODEL_HF_ID} ${SERVED_NAME} ${NEURON_CORES}' \
    < "${SCRIPT_DIR}/deployment.yaml" | kubectl apply -f -

# ── Step 5: Wait for pod to appear (Karpenter provisions trn1 node) ───────────
log_info "Waiting for pod (Karpenter provisioning ${NEURON_INSTANCE_TYPE} — allow 3-5 min)..."
sleep 15
POD=""
for i in $(seq 1 36); do
    POD=$(kubectl get pods \
        -l "app=${DEPLOYMENT_NAME}" \
        -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [[ -n "${POD}" ]] && break
    echo -n "."
    sleep 10
done
echo ""
log_info "Pod: ${POD:-not found yet — check: kubectl get pods -n ${BENCHMARK_NAMESPACE}}"

# ── Step 6: Wait for deployment ready ────────────────────────────────────────
# With --compile: NEFF already in S3, pod loads and starts in ~10-15 min.
# Without --compile: vLLM-Neuron compiles inside the pod — allow up to 2 hours.
log_info "Waiting for Ready (allow up to 2 hours if compiling inside pod)..."
kubectl wait "deployment/${DEPLOYMENT_NAME}" \
    --for=condition=Available \
    --timeout=7200s \
    -n "${BENCHMARK_NAMESPACE}" || {
    log_warn "Deployment not yet ready — check pod logs:"
    log_warn "  kubectl logs deployment/${DEPLOYMENT_NAME} -n ${BENCHMARK_NAMESPACE} --tail=50"
    log_warn "  kubectl describe pod -l app=${DEPLOYMENT_NAME} -n ${BENCHMARK_NAMESPACE}"
}

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  DEPLOYED — ${MODEL_ID} on ${NEURON_INSTANCE_TYPE}"
echo "══════════════════════════════════════════════════════════════"
kubectl get pods -l "app=${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" -o wide
echo ""
echo "  Port-forward:"
echo "    kubectl port-forward svc/${SERVICE_NAME} ${PORT_FORWARD_PORT}:8000 -n ${BENCHMARK_NAMESPACE} &"
echo "    curl http://localhost:${PORT_FORWARD_PORT}/health"
echo "    curl http://localhost:${PORT_FORWARD_PORT}/v1/models"
echo ""
echo "  Benchmark (shared harness):"
echo "    export VLLM_ENDPOINT=http://${SERVICE_NAME}:8000"
echo "    export MODEL_NAME=${SERVED_NAME}"
echo "    bash inference/run-benchmark.sh"
echo "══════════════════════════════════════════════════════════════"

if [[ "${VALIDATE}" == "true" ]]; then
    echo ""
    log_info "Running post-deploy validation (--validate flag set)..."
    bash "${SCRIPT_DIR}/../post-deploy-validate.sh" \
        --model "${MODEL_ID}" \
        --manifest "${MANIFEST_PATH}" \
        --endpoint "http://${SERVICE_NAME}:8000"
fi
