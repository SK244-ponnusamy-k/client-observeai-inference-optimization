#!/bin/bash
# ==============================================================================
# vllm/models/neuron/deploy.sh
#
# Deploy SINGLE-INSTANCE (non-DI) gpt-oss-20b on Trainium via vLLM Neuron.
# Names come from model.env; hardware chosen per run.
#
# Prereqs: Neuron NodePool (cluster/neuron-nodepool.yaml) + Neuron device plugin;
# NEURON_VLLM_IMAGE set in config/config.env.
#
# Usage:
#   bash vllm/models/neuron/deploy.sh                      # defaults (trn2.48xlarge)
#   bash vllm/models/neuron/deploy.sh --hw trn3.<size>     # MXFP4 on Trn3
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"
source "${SCRIPT_DIR}/model.env"

: "${NEURON_VLLM_IMAGE:?ERROR: NEURON_VLLM_IMAGE is not set. Add it to config/config.env (AWS Neuron vLLM plugin image, SDK 2.31).}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

VALIDATE="false"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hw)       NODE_INSTANCE_TYPE="$2"; shift 2 ;;
        --validate) VALIDATE="true"; shift ;;
        *) log_warn "Unknown arg: $1"; shift ;;
    esac
done
export NODE_INSTANCE_TYPE NEURON_DEVICE_COUNT MODEL_REF SERVED_NAME

echo ""
echo "══════════════════════════════════════════════════"
echo "  DEPLOYING (Neuron, single-instance) — ${MODEL_HF_ID}"
echo "  Deployment    : ${DEPLOYMENT_NAME}"
echo "  Service       : ${SERVICE_NAME}:8000"
echo "  Instance type : ${NODE_INSTANCE_TYPE}   (TP8, neuron=${NEURON_DEVICE_COUNT})"
echo "  Model ref     : ${MODEL_REF}"
echo "══════════════════════════════════════════════════"
echo ""

log_info "Applying cache PVC..."
kubectl apply -f "${SCRIPT_DIR}/pvc.yaml"

log_info "Applying service: ${SERVICE_NAME}..."
kubectl apply -f "${SCRIPT_DIR}/service.yaml"

log_info "Applying deployment: ${DEPLOYMENT_NAME} on ${NODE_INSTANCE_TYPE}..."
export BENCHMARK_NAMESPACE NEURON_VLLM_IMAGE
envsubst '${BENCHMARK_NAMESPACE} ${NEURON_VLLM_IMAGE} ${MODEL_REF} ${SERVED_NAME} ${NODE_INSTANCE_TYPE} ${NEURON_DEVICE_COUNT}' \
    < "${SCRIPT_DIR}/deployment.yaml" | kubectl apply -f -

log_info "Waiting for pod (Karpenter provisioning ${NODE_INSTANCE_TYPE})..."
sleep 15
for i in $(seq 1 36); do
    POD=$(kubectl get pods -l app="${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [[ -n "${POD}" ]] && break
    echo -n "."
    sleep 5
done
echo ""
log_info "Pod: ${POD:-not found yet}"

# First launch JIT-compiles the model graphs — this can take many minutes.
log_info "Waiting for Ready (FIRST launch compiles NEFFs — can take 20-60 min; cached after)..."
kubectl wait "deployment/${DEPLOYMENT_NAME}" \
    --for=condition=Available --timeout=3900s -n "${BENCHMARK_NAMESPACE}" || {
    log_warn "Not Available yet — check compile progress:"
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
