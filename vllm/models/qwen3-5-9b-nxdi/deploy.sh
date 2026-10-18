#!/bin/bash
# ==============================================================================
# vllm/models/qwen3-5-9b-nxdi/deploy.sh
#
# Deploys Qwen/Qwen3.5-9B (hybrid DeltaNet + GQA) on AWS Trainium/Neuron via the
# NxDI custom modeling code behind an OpenAI-compatible FastAPI shim.
#
#   bash deploy.sh [--compile] [--cores N] [--managed-ng] [--validate]
#
#   --compile     Run the NxDI compile job first (idempotent; first deploy only)
#   --cores N     Override NEURON_CORES (must match the compiled artifact)
#   --managed-ng  Use the managed node group instead of Karpenter
#   --validate    Smoke-test the endpoint after it is ready
#
# vLLM-Neuron does NOT support this architecture; that is why this path exists.
# Code (src/, server.py, compile.py) ships to the pods via ConfigMaps, matching
# this repo's ConfigMap-for-app-code convention (see inference/benchmark-job.yaml).
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"
source "${SCRIPT_DIR}/model.env"

: "${NEURON_NXDI_IMAGE:?ERROR: NEURON_NXDI_IMAGE is not set in config/config.env.}"
: "${MODEL_BUCKET:?ERROR: MODEL_BUCKET is not set in config/config.env.}"
: "${BENCHMARK_NAMESPACE:?ERROR: BENCHMARK_NAMESPACE is not set in config/config.env.}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

RUN_COMPILE="false"; VALIDATE="false"; USE_MANAGED_NG="false"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --compile)    RUN_COMPILE="true"; shift ;;
        --cores)      NEURON_CORES="$2"; shift 2 ;;
        --hw)         NEURON_INSTANCE_TYPE="$2"; export OAI_INSTANCE_TYPE="$2"; shift 2 ;;
        --validate)   VALIDATE="true"; shift ;;
        --managed-ng) USE_MANAGED_NG="true"; shift ;;
        *) log_warn "Unknown argument: $1"; shift ;;
    esac
done

# Device count per instance: trn2.3xlarge = 1 chip; trn2.48xlarge/trn1.32xlarge = 16.
case "${NEURON_INSTANCE_TYPE}" in
    trn2.3xlarge) NEURON_DEVICE_COUNT=1 ;;
    trn2.*)       NEURON_DEVICE_COUNT=16 ;;
    trn1.*)       NEURON_DEVICE_COUNT=16 ;;
    *)            NEURON_DEVICE_COUNT=1 ;;
esac
INSTANCE_FAMILY="${NEURON_INSTANCE_TYPE%%.*}"

# S3 buckets may live in a different region than the cluster.
export S3_BUCKET_REGION="${S3_BUCKET_REGION:-$AWS_REGION}"

export MODEL_BUCKET BENCHMARK_NAMESPACE NEURON_NXDI_IMAGE SOURCE_FOLDER COMPILED_FOLDER \
    SERVED_NAME NEURON_CORES NEURON_DEVICE_COUNT AWS_REGION S3_BUCKET_REGION INSTANCE_FAMILY \
    TRN2_NODEGROUP_NAME DOWNLOAD_SERVICE_ACCOUNT \
    QWEN35_SEQ_LEN QWEN35_MAX_BATCH_SIZE QWEN35_TP_DEGREE QWEN35_LOGICAL_NC_CONFIG \
    QWEN35_TORCH_DTYPE QWEN35_TOP_K_ON_DEVICE

echo ""
echo "=============================================================="
echo "  DEPLOYING (NxDI) - Qwen/Qwen3.5-9B  (DeltaNet + GQA)"
echo "  Deployment    : ${DEPLOYMENT_NAME}"
echo "  Instance type : ${NEURON_INSTANCE_TYPE}  (NeuronCores=${NEURON_CORES}, devices=${NEURON_DEVICE_COUNT})"
echo "  seq_len/batch : ${QWEN35_SEQ_LEN} / ${QWEN35_MAX_BATCH_SIZE}"
echo "  Compile first : ${RUN_COMPILE}   Managed-NG: ${USE_MANAGED_NG}"
echo "=============================================================="
echo ""

# ── Ship code to the cluster as ConfigMaps (src/, server.py, compile.py) ─────
create_code_configmaps() {
    log_info "Creating/updating code ConfigMaps from ${SCRIPT_DIR} ..."
    kubectl create configmap "oai-infopt-nxdi-code-${MODEL_ID}" \
        --from-file=server.py="${SCRIPT_DIR}/server.py" \
        --from-file=compile.py="${SCRIPT_DIR}/compile.py" \
        -n "${BENCHMARK_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    kubectl create configmap "oai-infopt-nxdi-src-${MODEL_ID}" \
        --from-file="${SCRIPT_DIR}/src/__init__.py" \
        --from-file="${SCRIPT_DIR}/src/modeling_qwen35.py" \
        -n "${BENCHMARK_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    kubectl create configmap "oai-infopt-nxdi-nki-${MODEL_ID}" \
        --from-file="${SCRIPT_DIR}/src/nki_kernels/__init__.py" \
        --from-file="${SCRIPT_DIR}/src/nki_kernels/nki_deltanet.py" \
        --from-file="${SCRIPT_DIR}/src/nki_kernels/nki_deltanet_chunked.py" \
        --from-file="${SCRIPT_DIR}/src/nki_kernels/nki_deltanet_fused.py" \
        -n "${BENCHMARK_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
}

scale_up_managed_ng() {
    log_info "Scaling up managed node group: ${TRN2_NODEGROUP_NAME} (desiredSize=1)..."
    aws eks update-nodegroup-config --cluster-name "${CLUSTER_NAME}" \
        --nodegroup-name "${TRN2_NODEGROUP_NAME}" \
        --scaling-config minSize=0,maxSize=1,desiredSize=1 --region "${AWS_REGION}" >/dev/null
    log_info "Waiting for node to join (~3-5 min)..."
    for i in $(seq 1 30); do
        TRN2_NODE=$(kubectl get nodes -l "eks.amazonaws.com/nodegroup=${TRN2_NODEGROUP_NAME}" \
            -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
        [[ -n "${TRN2_NODE}" ]] && { log_info "Node: ${TRN2_NODE}"; break; }
        echo -n "."; sleep 10
    done
    echo ""
    [[ -n "${TRN2_NODE:-}" ]] && kubectl wait node "${TRN2_NODE}" --for=condition=Ready --timeout=300s || \
        log_warn "Node not Ready yet - pod will wait."
}

# Compile-job nodeSelector mirrors the serving selector.
if [[ "${USE_MANAGED_NG}" == "true" ]]; then
    COMPILE_NODE_SELECTOR_KEY="eks.amazonaws.com/nodegroup"
    COMPILE_NODE_SELECTOR_VALUE="${TRN2_NODEGROUP_NAME}"
else
    COMPILE_NODE_SELECTOR_KEY="eks.amazonaws.com/instance-family"
    COMPILE_NODE_SELECTOR_VALUE="${INSTANCE_FAMILY}"
fi
export COMPILE_NODE_SELECTOR_KEY COMPILE_NODE_SELECTOR_VALUE

if [[ "${USE_MANAGED_NG}" == "true" && -n "${TRN2_NODEGROUP_NAME:-}" ]]; then
    scale_up_managed_ng
fi

log_info "Checking bf16 source weights: s3://${MODEL_BUCKET}/${SOURCE_FOLDER}/"
if ! aws s3 ls "s3://${MODEL_BUCKET}/${SOURCE_FOLDER}/" --region "${S3_BUCKET_REGION}" >/dev/null 2>&1; then
    log_error "Source weights not found. Download first (HF -> S3): see model-download/qwen3-5-9b-nxdi/"
    exit 1
fi
log_info "Source weights found."

create_code_configmaps

if [[ "${RUN_COMPILE}" == "true" ]]; then
    log_info "Submitting NxDI compile job (idempotent)..."
    envsubst '${BENCHMARK_NAMESPACE} ${DOWNLOAD_SERVICE_ACCOUNT} ${MODEL_BUCKET} ${AWS_REGION} ${S3_BUCKET_REGION} ${NEURON_NXDI_IMAGE} ${SOURCE_FOLDER} ${COMPILED_FOLDER} ${QWEN35_SEQ_LEN} ${QWEN35_MAX_BATCH_SIZE} ${QWEN35_TP_DEGREE} ${QWEN35_LOGICAL_NC_CONFIG} ${QWEN35_TORCH_DTYPE} ${QWEN35_TOP_K_ON_DEVICE} ${NEURON_CORES} ${NEURON_DEVICE_COUNT} ${COMPILE_NODE_SELECTOR_KEY} ${COMPILE_NODE_SELECTOR_VALUE}' \
        < "${SCRIPT_DIR}/compile-job.yaml" | kubectl apply -f -
    log_info "Waiting for compile job (up to 12h)..."
    kubectl wait job/oai-infopt-nxdi-compile-qwen3-5-9b-nxdi --for=condition=Complete --timeout=43200s -n "${BENCHMARK_NAMESPACE}" || {
        log_error "Compile job failed. Logs: kubectl logs -f job/oai-infopt-nxdi-compile-qwen3-5-9b-nxdi -n ${BENCHMARK_NAMESPACE}"
        exit 1
    }
    log_info "Compile complete."
fi

log_info "Applying service: ${SERVICE_NAME}..."
kubectl apply -f "${SCRIPT_DIR}/service.yaml"

if [[ "${USE_MANAGED_NG}" == "true" ]]; then
    DEPLOY_FILE="${SCRIPT_DIR}/deployment-managed-ng.yaml"
    log_info "Using managed node group deployment..."
else
    DEPLOY_FILE="${SCRIPT_DIR}/deployment.yaml"
fi
log_info "Applying deployment: ${DEPLOYMENT_NAME} (S3 region=${S3_BUCKET_REGION})..."
envsubst '${BENCHMARK_NAMESPACE} ${NEURON_NXDI_IMAGE} ${MODEL_BUCKET} ${SOURCE_FOLDER} ${COMPILED_FOLDER} ${SERVED_NAME} ${NEURON_CORES} ${NEURON_DEVICE_COUNT} ${AWS_REGION} ${S3_BUCKET_REGION} ${INSTANCE_FAMILY} ${TRN2_NODEGROUP_NAME} ${QWEN35_SEQ_LEN} ${QWEN35_MAX_BATCH_SIZE} ${QWEN35_TP_DEGREE} ${QWEN35_LOGICAL_NC_CONFIG} ${QWEN35_TORCH_DTYPE} ${QWEN35_TOP_K_ON_DEVICE}' \
    < "${DEPLOY_FILE}" | kubectl apply -f -

log_info "Waiting for pod..."
sleep 15
for i in $(seq 1 36); do
    POD=$(kubectl get pods -l "app=${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [[ -n "${POD}" ]] && break
    echo -n "."; sleep 10
done
echo ""
log_info "Pod: ${POD:-not found yet}"

log_info "Waiting for Ready (up to 2h if compiling inside the pod)..."
kubectl wait "deployment/${DEPLOYMENT_NAME}" --for=condition=Available --timeout=7200s -n "${BENCHMARK_NAMESPACE}" || {
    log_warn "Not ready yet - check: kubectl logs deployment/${DEPLOYMENT_NAME} -n ${BENCHMARK_NAMESPACE} --tail=50"
}

echo ""
echo "=============================================================="
echo "  DEPLOYED - qwen3-5-9b-nxdi on ${NEURON_INSTANCE_TYPE}"
echo "=============================================================="
kubectl get pods -l "app=${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" -o wide
echo ""
echo "  Port-forward:"
echo "    kubectl port-forward svc/${SERVICE_NAME} ${PORT_FORWARD_PORT}:8000 -n ${BENCHMARK_NAMESPACE} &"
echo "=============================================================="

if [[ "${VALIDATE}" == "true" ]]; then
    log_info "Running post-deploy validation..."
    bash "${SCRIPT_DIR}/../post-deploy-validate.sh" \
        --model "${MODEL_ID}" --manifest "${MANIFEST_PATH}" --endpoint "http://${SERVICE_NAME}:8000" || \
        log_warn "Validation script reported issues - inspect manually."
fi
