#!/bin/bash
# ==============================================================================
# vllm/models/qwen3-5-4b-neuron/deploy.sh
#
# Deploys Qwen/Qwen3.5-4B on AWS Trainium/Neuron via vLLM-Neuron.
# Friendly wrapper:  oai deploy qwen3-5-4b-neuron [--managed-ng]
#
#   --cores N     Override NEURON_CORES (must match compile-time value)
#   --managed-ng  Use the managed node group instead of Karpenter
#   --validate    Smoke-test after ready
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"
source "${SCRIPT_DIR}/model.env"

: "${NEURON_VLLM_IMAGE:?ERROR: NEURON_VLLM_IMAGE is not set in config/config.env.}"
: "${MODEL_BUCKET:?ERROR: MODEL_BUCKET is not set in config/config.env.}"
: "${BENCHMARK_NAMESPACE:?ERROR: BENCHMARK_NAMESPACE is not set in config/config.env.}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

VALIDATE="false"; USE_MANAGED_NG="false"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cores)      NEURON_CORES="$2"; shift 2 ;;
        --hw)         NEURON_INSTANCE_TYPE="$2"; export OAI_INSTANCE_TYPE="$2"; shift 2 ;;
        --validate)   VALIDATE="true"; shift ;;
        --managed-ng) USE_MANAGED_NG="true"; shift ;;
        *) log_warn "Unknown argument: $1"; shift ;;
    esac
done
export NEURON_CORES NEURON_INSTANCE_TYPE OAI_INSTANCE_TYPE MODEL_FOLDER SERVED_NAME NEURON_VLLM_IMAGE

echo ""
echo "=============================================================="
echo "  DEPLOYING (Neuron) - Qwen/Qwen3.5-4B"
echo "  Deployment    : ${DEPLOYMENT_NAME}"
echo "  Instance type : ${NEURON_INSTANCE_TYPE}   (NeuronCores=${NEURON_CORES})"
echo "  Compile first : false   Managed-NG: ${USE_MANAGED_NG}"
echo "=============================================================="
echo ""

if [[ "${USE_MANAGED_NG}" == "true" && -n "${TRN2_NODEGROUP_NAME:-}" ]]; then
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
fi

log_info "Checking bf16 source weights: s3://${MODEL_BUCKET}/${SOURCE_FOLDER}/"
if ! aws s3 ls "s3://${MODEL_BUCKET}/${SOURCE_FOLDER}/" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log_error "Source weights not found. Download first: oai download qwen3-5-4b-neuron"
    exit 1
fi
log_info "Source weights found."

log_info "Applying service: ${SERVICE_NAME}..."
kubectl apply -f "${SCRIPT_DIR}/service.yaml"

export MODEL_BUCKET BENCHMARK_NAMESPACE NEURON_VLLM_IMAGE MODEL_FOLDER MODEL_HF_ID SERVED_NAME NEURON_CORES AWS_REGION
if [[ "${USE_MANAGED_NG}" == "true" ]]; then
    DEPLOY_FILE="${SCRIPT_DIR}/deployment-managed-ng.yaml"
    log_info "Using managed node group deployment..."
else
    DEPLOY_FILE="${SCRIPT_DIR}/deployment.yaml"
fi
log_info "Applying deployment: ${DEPLOYMENT_NAME} (NeuronCores=${NEURON_CORES})..."
envsubst '${MODEL_BUCKET} ${BENCHMARK_NAMESPACE} ${NEURON_VLLM_IMAGE} ${MODEL_FOLDER} ${MODEL_HF_ID} ${SERVED_NAME} ${NEURON_CORES} ${AWS_REGION}' \
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

log_info "Waiting for Ready (up to 2h if compiling inside pod)..."
kubectl wait "deployment/${DEPLOYMENT_NAME}" --for=condition=Available --timeout=7200s -n "${BENCHMARK_NAMESPACE}" || {
    log_warn "Not ready yet - check: kubectl logs deployment/${DEPLOYMENT_NAME} -n ${BENCHMARK_NAMESPACE} --tail=50"
}

echo ""
echo "=============================================================="
echo "  DEPLOYED - qwen3-5-4b-neuron on ${NEURON_INSTANCE_TYPE}"
echo "=============================================================="
kubectl get pods -l "app=${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" -o wide
echo ""
echo "  Port-forward:"
echo "    kubectl port-forward svc/${SERVICE_NAME} ${PORT_FORWARD_PORT}:8000 -n ${BENCHMARK_NAMESPACE} &"
echo "=============================================================="

if [[ "${VALIDATE}" == "true" ]]; then
    log_info "Running post-deploy validation..."
    bash "${SCRIPT_DIR}/../post-deploy-validate.sh" \
        --model "${MODEL_ID}" --manifest "${MANIFEST_PATH}" --endpoint "http://${SERVICE_NAME}:8000"
fi
