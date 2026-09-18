#!/bin/bash
# ==============================================================================
# vllm/models/qwen3.5-4b/deploy.sh
#
# Generic deploy — model-specific names come from model.env. The HARDWARE is
# chosen per run so one model can be benchmarked across the g5/g6/g6e matrix
# without editing YAML.
#
# Usage:
#   bash vllm/models/qwen3.5-4b/deploy.sh                       # defaults (model.env)
#   bash vllm/models/qwen3.5-4b/deploy.sh --hw g5.2xlarge       # pin GPU family/size
#   bash vllm/models/qwen3.5-4b/deploy.sh --hw g6e.12xlarge --tp 4   # tensor-parallel
#   bash vllm/models/qwen3.5-4b/deploy.sh --hw g6.2xlarge --validate
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

# ── Parse flags: --hw <type> --tp <n> --validate --benchmark [--profile --dataset] ─
VALIDATE="false"
RUN_BENCHMARK="false"
BENCH_PROFILE="both"          # both | realtime | batch
BENCH_DATASET=""              # optional custom dataset (S3 key / path)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hw)        NODE_INSTANCE_TYPE="$2"; shift 2 ;;
        --tp)        TP_SIZE="$2"; shift 2 ;;
        --validate)  VALIDATE="true"; shift ;;
        --benchmark) RUN_BENCHMARK="true"; shift ;;
        --profile)   BENCH_PROFILE="$2"; shift 2 ;;
        --dataset)   BENCH_DATASET="$2"; shift 2 ;;
        *) log_warn "Unknown arg: $1"; shift ;;
    esac
done
# TP implies one GPU per shard on the node.
GPU_COUNT="${TP_SIZE}"
export NODE_INSTANCE_TYPE TP_SIZE GPU_COUNT MODEL_FOLDER SERVED_NAME

echo ""
echo "══════════════════════════════════════════════════"
echo "  DEPLOYING — ${MODEL_HF_ID}"
echo "  Deployment    : ${DEPLOYMENT_NAME}"
echo "  Service       : ${SERVICE_NAME}:8000"
echo "  Namespace     : ${BENCHMARK_NAMESPACE}"
echo "  Instance type : ${NODE_INSTANCE_TYPE}   (TP=${TP_SIZE}, GPUs=${GPU_COUNT})"
echo "══════════════════════════════════════════════════"
echo ""

# Check model in S3
log_info "Checking model in S3..."
if ! aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log_error "Model not found: s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
    log_error "Download it first: bash model-download/qwen3.5-4b/download.sh"
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

log_info "Waiting for pod (Karpenter provisioning ${NODE_INSTANCE_TYPE} ~2-4 min)..."
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

log_info "Waiting for Ready (model loading from S3 — up to 10 min)..."
DEPLOY_READY="false"
if kubectl wait "deployment/${DEPLOYMENT_NAME}" \
    --for=condition=Available --timeout=600s -n "${BENCHMARK_NAMESPACE}"; then
    DEPLOY_READY="true"
else
    log_warn "Deployment not ready yet — check logs:"
    log_warn "  kubectl logs deployment/${DEPLOYMENT_NAME} -n ${BENCHMARK_NAMESPACE} | tail -30"
fi

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

# ==============================================================================
# Auto-benchmark — runs after a successful deploy when --benchmark is passed.
#   --benchmark                     → runs BOTH realtime and batch (default)
#   --benchmark --profile realtime  → realtime only
#   --benchmark --profile batch     → batch only
#
# 'both' submits realtime and batch as TWO SEPARATE k8s Jobs that run
# SEQUENTIALLY: batch's init container waits (via an S3 marker) for realtime to
# finish before it loads the GPU. Both Jobs are submitted up front, so a batch
# failure never discards the realtime results, and closing this terminal / Ctrl+C
# does not stop either Job — they run to completion in the cluster.
#
# NOTE: run-benchmark.sh resolves manifests as <model>-<hw>-bf16.yaml. The qwen
# manifests use the dotted name (qwen3.5-4b-*), so pass that, not MODEL_ID.
# ==============================================================================
if [[ "${RUN_BENCHMARK}" == "true" ]]; then
    HW_SUFFIX="${NODE_INSTANCE_TYPE%%.*}"     # g6e.2xlarge -> g6e
    if [[ "${DEPLOY_READY}" != "true" ]]; then
        log_warn "Skipping auto-benchmark: deployment never became Ready."
        log_warn "Once the pod is 1/1, run manually:"
        log_warn "  bash inference/run-benchmark.sh --model qwen3.5-4b --hw ${HW_SUFFIX} --profile ${BENCH_PROFILE}"
        exit 0
    fi

    BENCH_DATASET_ARGS=()
    if [[ -n "${BENCH_DATASET}" ]]; then
        BENCH_DATASET_ARGS=(--dataset "${BENCH_DATASET}")
    fi

    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  AUTO-BENCHMARK — qwen3.5-4b / ${HW_SUFFIX} / ${BENCH_PROFILE}"
    echo "  (realtime + batch = two separate sequential Jobs)"
    echo "══════════════════════════════════════════════════"
    bash "${FRAMEWORK_ROOT}/inference/run-benchmark.sh" \
        --model "qwen3.5-4b" \
        --hw "${HW_SUFFIX}" \
        --profile "${BENCH_PROFILE}" \
        "${BENCH_DATASET_ARGS[@]}" || \
        log_warn "Benchmark submit reported non-zero — check output/S3 results above."

    echo ""
    log_info "Auto-benchmark submitted (profile: ${BENCH_PROFILE}). Jobs run in-cluster."
fi
