#!/bin/bash
# ==============================================================================
# model-download/qwen3.5-4b/download.sh
#
# Downloads Qwen/Qwen3.5-4B from HuggingFace → S3. Public model (token optional).
#
# Model info:
#   Repo : Qwen/Qwen3.5-4B   (dense multimodal VLM, ~4B params, bf16 ~8-9 GB)
#   Note : we benchmark the TEXT path only. The vision tower weights are still
#          downloaded (full checkpoint) but not exercised at serve time.
#
# Usage:
#   bash model-download/qwen3.5-4b/download.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

MODEL_FOLDER="Qwen3.5-4B"
JOB_NAME="oai-infopt-download-qwen3-5-4b"

echo ""
echo "══════════════════════════════════════════════════"
echo "  MODEL DOWNLOAD — Qwen/Qwen3.5-4B"
echo "  Bucket : s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
echo "  Size   : ~8-9 GB (bf16, multimodal)"
echo "══════════════════════════════════════════════════"
echo ""

log_info "Checking S3 for existing model files..."
EXISTING=$(aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/" --region "${AWS_REGION}" 2>/dev/null | head -1 || true)
if [[ -n "${EXISTING}" ]]; then
    log_warn "Model already exists in S3 — skipping download."
    aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/" --human-readable --region "${AWS_REGION}"
    exit 0
fi

if ! kubectl get serviceaccount "${DOWNLOAD_SERVICE_ACCOUNT}" -n "${BENCHMARK_NAMESPACE}" >/dev/null 2>&1; then
    log_error "ServiceAccount '${DOWNLOAD_SERVICE_ACCOUNT}' not found. Run: bash cluster/bootstrap.sh"
    exit 1
fi
log_info "ServiceAccount '${DOWNLOAD_SERVICE_ACCOUNT}' found."
kubectl get secret hf-token -n "${BENCHMARK_NAMESPACE}" >/dev/null 2>&1 \
    || log_warn "HF token secret not found — proceeding (Qwen3.5-4B is public)."

if kubectl get job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" >/dev/null 2>&1; then
    log_warn "Existing job '${JOB_NAME}' found — deleting before rerun..."
    kubectl delete job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}"; sleep 3
fi

log_info "Applying download job..."
export MODEL_BUCKET AWS_REGION DOWNLOAD_SERVICE_ACCOUNT BENCHMARK_NAMESPACE
envsubst '${MODEL_BUCKET} ${AWS_REGION} ${DOWNLOAD_SERVICE_ACCOUNT} ${BENCHMARK_NAMESPACE}' \
    < "${SCRIPT_DIR}/job.yaml" | kubectl apply -f -
log_info "Job '${JOB_NAME}' created. (~9 GB — expect 10-30 minutes)"

log_info "Waiting for pod to start..."
for i in $(seq 1 36); do
    POD=$(kubectl get pods -l job-name="${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [[ -n "${POD}" ]] && break; echo -n "."; sleep 5
done
echo ""
[[ -z "${POD:-}" ]] && { log_error "Pod not found after 3 minutes."; exit 1; }
log_info "Pod: ${POD}"

log_info "Streaming logs (Ctrl+C to detach — job continues)..."
echo ""
kubectl logs "${POD}" -n "${BENCHMARK_NAMESPACE}" -f 2>/dev/null || \
    kubectl logs "${POD}" -n "${BENCHMARK_NAMESPACE}" || true

log_info "Waiting for job to complete (timeout: 6h)..."
kubectl wait job "${JOB_NAME}" --for=condition=complete --timeout=21600s -n "${BENCHMARK_NAMESPACE}" || {
    log_error "Job did not complete. Check: kubectl logs ${POD} -n ${BENCHMARK_NAMESPACE}"; exit 1;
}
log_info "Job completed successfully."

echo ""
log_info "Verifying S3 upload..."
aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/" --recursive --human-readable --region "${AWS_REGION}"

log_info "Cleaning up job..."
kubectl delete job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}"

echo ""
echo "══════════════════════════════════════════════════"
echo "  DOWNLOAD COMPLETE — s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
echo "  Next : bash vllm/models/qwen3.5-4b/deploy.sh --hw g6e.2xlarge"
echo "══════════════════════════════════════════════════"
