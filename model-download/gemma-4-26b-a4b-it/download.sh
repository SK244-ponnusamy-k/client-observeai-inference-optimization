#!/bin/bash
# ==============================================================================
# model-download/gemma-4-26b-a4b-it/download.sh
#
# Downloads a 4-bit (W4A16) Gemma-4-26B-A4B-it checkpoint from HuggingFace → S3.
#
# ⚠ CHECKPOINT SOURCE MUST BE CONFIRMED.
#   vLLM needs a 4-bit checkpoint it can load directly. Preferred format is
#   compressed-tensors W4A16 (auto-detected, no --quantization flag) or AWQ.
#   Set QUANT_HF_ID to that repo. If no ready 4-bit MoE checkpoint exists for
#   this model yet, quantize google/gemma-4-26B-A4B-it offline with llm-compressor
#   (W4A16) and push the result to s3://${MODEL_BUCKET}/${MODEL_FOLDER}/ directly,
#   then skip this script.
#
#   Example:
#     QUANT_HF_ID=<org>/gemma-4-26B-A4B-it-W4A16 bash model-download/gemma-4-26b-a4b-it/download.sh
#
# Uploads to: s3://${MODEL_BUCKET}/Gemma-4-26B-A4B-it-w4a16/
#
# Model info:
#   Base : google/gemma-4-26B-A4B-it  (MoE, 25.2B total / ~3.8B active, multimodal)
#   Quant: 4-bit W4A16 → ~14-15 GB. Fits a single GPU on g5 / g6 / g6e.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

# The target checkpoint repo — defaults to google/gemma-4-26B-A4B-it
QUANT_HF_ID="${QUANT_HF_ID:-google/gemma-4-26B-A4B-it}"
MODEL_FOLDER="Gemma-4-26B-A4B-it-w4a16"
JOB_NAME="oai-infopt-download-gemma-4-26b-a4b"

echo ""
echo "══════════════════════════════════════════════════"
echo "  MODEL DOWNLOAD — ${QUANT_HF_ID} (4-bit)"
echo "  Bucket : s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
echo "  Size   : ~14-15 GB (W4A16 MoE)"
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
    || log_warn "HF token secret not found — proceeding (add it if the checkpoint repo is gated)."

if kubectl get job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" >/dev/null 2>&1; then
    log_warn "Existing job '${JOB_NAME}' found — deleting before rerun..."
    kubectl delete job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}"; sleep 3
fi

log_info "Applying download job (source: ${QUANT_HF_ID})..."
export MODEL_BUCKET AWS_REGION DOWNLOAD_SERVICE_ACCOUNT BENCHMARK_NAMESPACE QUANT_HF_ID MODEL_FOLDER
envsubst '${MODEL_BUCKET} ${AWS_REGION} ${DOWNLOAD_SERVICE_ACCOUNT} ${BENCHMARK_NAMESPACE} ${QUANT_HF_ID} ${MODEL_FOLDER}' \
    < "${SCRIPT_DIR}/job.yaml" | kubectl apply -f -
log_info "Job '${JOB_NAME}' created. (~15 GB — expect 20-60 minutes)"

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

log_info "Waiting for job to complete (timeout: 12h)..."
kubectl wait job "${JOB_NAME}" --for=condition=complete --timeout=43200s -n "${BENCHMARK_NAMESPACE}" || {
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
echo "  Next : bash vllm/models/gemma-4-26b-a4b-it/deploy.sh --hw g6e.2xlarge"
echo "══════════════════════════════════════════════════"
