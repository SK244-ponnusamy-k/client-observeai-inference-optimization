#!/bin/bash
# ==============================================================================
# vllm/models/qwen3-5-4b-torchneuro/deploy.sh
#
# Deploys Qwen/Qwen3.5-4B on AWS Trainium via TorchNeuron + vLLM Neuron.
#
# TorchNeuron (public beta Oct 20 2026) registers Trainium as a native PyTorch
# device so the model loads with a single change:
#
#   model = AutoModelForCausalLM.from_pretrained(..., device_map="neuron")
#
# vLLM Neuron then serves it with `vllm serve --device neuron` — no custom
# modeling code, no ConfigMaps, no FastAPI shim. This is structurally identical
# to the gpt-oss-20b-neuron path.
#
#   bash deploy.sh [--compile] [--cores N] [--managed-ng] [--validate]
#
#   --compile     Run the compile job first (idempotent; caches NEFFs in S3)
#   --cores N     Override NEURON_CORES (must match compile-time value)
#   --hw TYPE     Override NEURON_INSTANCE_TYPE
#   --managed-ng  Use the managed node group instead of Karpenter
#   --validate    Smoke-test the endpoint after it is ready
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"
source "${SCRIPT_DIR}/model.env"

: "${NEURON_VLLM_IMAGE:?ERROR: NEURON_VLLM_IMAGE is not set in config/config.env.}"
: "${NEURON_COMPILE_IMAGE:?ERROR: NEURON_COMPILE_IMAGE is not set in config/config.env.}"
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

# S3 buckets may live in a different region than the EKS cluster.
export S3_BUCKET_REGION="${S3_BUCKET_REGION:-$AWS_REGION}"

export NEURON_CORES NEURON_INSTANCE_TYPE OAI_INSTANCE_TYPE \
    MODEL_FOLDER SERVED_NAME NEURON_VLLM_IMAGE NEURON_COMPILE_IMAGE

echo ""
echo "=============================================================="
echo "  DEPLOYING (TorchNeuron) - Qwen/Qwen3.5-4B"
echo "  Deployment    : ${DEPLOYMENT_NAME}"
echo "  Instance type : ${NEURON_INSTANCE_TYPE}  (NeuronCores=${NEURON_CORES}, LNC=${NEURON_LOGICAL_NC_CONFIG})"
echo "  Compile first : ${RUN_COMPILE}   Managed-NG: ${USE_MANAGED_NG}"
echo ""
echo "  TorchNeuron approach:"
echo "    model = AutoModelForCausalLM.from_pretrained(..., device_map='neuron')"
echo "    vllm serve /model --device neuron   # the only Neuron-specific arg"
echo "=============================================================="
echo ""

# ── Scale up managed node group if requested ─────────────────────────────────
if [[ "${USE_MANAGED_NG}" == "true" && -n "${TRN2_NODEGROUP_NAME:-}" ]]; then
    log_info "Scaling up managed node group: ${TRN2_NODEGROUP_NAME} (desiredSize=1)..."
    aws eks update-nodegroup-config \
        --cluster-name "${CLUSTER_NAME}" \
        --nodegroup-name "${TRN2_NODEGROUP_NAME}" \
        --scaling-config minSize=0,maxSize=1,desiredSize=1 \
        --region "${AWS_REGION}" >/dev/null
    log_info "Waiting for node to join (~3-5 min)..."
    for i in $(seq 1 30); do
        TRN2_NODE=$(kubectl get nodes \
            -l "eks.amazonaws.com/nodegroup=${TRN2_NODEGROUP_NAME}" \
            -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
        [[ -n "${TRN2_NODE}" ]] && { log_info "Node: ${TRN2_NODE}"; break; }
        echo -n "."; sleep 10
    done
    echo ""
    [[ -n "${TRN2_NODE:-}" ]] && \
        kubectl wait node "${TRN2_NODE}" --for=condition=Ready --timeout=300s || \
        log_warn "Node not Ready yet — pod will wait."
fi

# ── Verify bf16 source weights exist in S3 ───────────────────────────────────
log_info "Checking bf16 source weights: s3://${MODEL_BUCKET}/${SOURCE_FOLDER}/"
if ! aws s3 ls "s3://${MODEL_BUCKET}/${SOURCE_FOLDER}/" \
        --region "${S3_BUCKET_REGION}" >/dev/null 2>&1; then
    log_error "Source weights not found. Download first:"
    log_error "  huggingface-cli download Qwen/Qwen3.5-4B --local-dir /tmp/qwen"
    log_error "  aws s3 sync /tmp/qwen s3://${MODEL_BUCKET}/${SOURCE_FOLDER}/ --region ${S3_BUCKET_REGION}"
    exit 1
fi
log_info "Source weights found."

# ── Compile (idempotent — skips if NEFF cache already in S3) ─────────────────
if [[ "${RUN_COMPILE}" == "true" ]]; then
    log_info "Submitting TorchNeuron compile job (idempotent)..."
    export MODEL_BUCKET AWS_REGION S3_BUCKET_REGION DOWNLOAD_SERVICE_ACCOUNT \
        BENCHMARK_NAMESPACE NEURON_COMPILE_IMAGE
    envsubst '${BENCHMARK_NAMESPACE} ${DOWNLOAD_SERVICE_ACCOUNT} ${MODEL_BUCKET} ${AWS_REGION} ${S3_BUCKET_REGION} ${NEURON_COMPILE_IMAGE}' \
        < "${SCRIPT_DIR}/compile-job.yaml" | kubectl apply -f -
    log_info "Waiting for compile job (up to 4h)..."
    kubectl wait job/oai-infopt-neuron-compile-qwen3-5-4b-torchneuro \
        --for=condition=Complete --timeout=14400s -n "${BENCHMARK_NAMESPACE}" || {
        log_error "Compile job failed. Check logs:"
        log_error "  kubectl logs -f job/oai-infopt-neuron-compile-qwen3-5-4b-torchneuro -n ${BENCHMARK_NAMESPACE}"
        exit 1
    }
    log_info "Compile complete — NEFFs cached at s3://${MODEL_BUCKET}/${NEFF_FOLDER}/"
fi

# ── Apply service ─────────────────────────────────────────────────────────────
log_info "Applying service: ${SERVICE_NAME}..."
kubectl apply -f "${SCRIPT_DIR}/service.yaml"

# ── Apply deployment ──────────────────────────────────────────────────────────
if [[ "${USE_MANAGED_NG}" == "true" ]]; then
    DEPLOY_FILE="${SCRIPT_DIR}/deployment-managed-ng.yaml"
    log_info "Using managed node-group deployment..."
else
    DEPLOY_FILE="${SCRIPT_DIR}/deployment.yaml"
fi

log_info "Applying deployment: ${DEPLOYMENT_NAME} (NeuronCores=${NEURON_CORES}, S3 region=${S3_BUCKET_REGION})..."
envsubst '${MODEL_BUCKET} ${BENCHMARK_NAMESPACE} ${NEURON_VLLM_IMAGE} ${MODEL_FOLDER} ${MODEL_HF_ID} ${SERVED_NAME} ${NEURON_CORES} ${AWS_REGION} ${S3_BUCKET_REGION} ${TRN2_NODEGROUP_NAME}' \
    < "${DEPLOY_FILE}" | kubectl apply -f -

# ── Wait for pod ──────────────────────────────────────────────────────────────
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

log_info "Waiting for Ready (up to 2h; model loads NEFF cache on first start)..."
kubectl wait "deployment/${DEPLOYMENT_NAME}" \
    --for=condition=Available --timeout=7200s -n "${BENCHMARK_NAMESPACE}" || {
    log_warn "Not ready yet — check:"
    log_warn "  kubectl logs deployment/${DEPLOYMENT_NAME} -n ${BENCHMARK_NAMESPACE} --tail=50"
}

echo ""
echo "=============================================================="
echo "  DEPLOYED - qwen3-5-4b-torchneuro on ${NEURON_INSTANCE_TYPE}"
echo "=============================================================="
kubectl get pods -l "app=${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" -o wide
echo ""
echo "  Port-forward:"
echo "    kubectl port-forward svc/${SERVICE_NAME} ${PORT_FORWARD_PORT}:8000 -n ${BENCHMARK_NAMESPACE} &"
echo ""
echo "  Smoke-test:"
echo "    curl -s localhost:${PORT_FORWARD_PORT}/v1/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\":\"${SERVED_NAME}\",\"prompt\":\"The capital of France is\",\"max_tokens\":16}'"
echo "=============================================================="

if [[ "${VALIDATE}" == "true" ]]; then
    log_info "Running post-deploy validation..."
    bash "${SCRIPT_DIR}/../post-deploy-validate.sh" \
        --model "${MODEL_ID}" \
        --manifest "${MANIFEST_PATH}" \
        --endpoint "http://${SERVICE_NAME}:8000" || \
        log_warn "Validation script reported issues — inspect manually."
fi
