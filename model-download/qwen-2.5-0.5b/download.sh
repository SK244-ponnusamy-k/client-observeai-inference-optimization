#!/bin/bash
# ==============================================================================
# model-download/qwen-2.5-0.5b/download.sh
#
# Downloads Qwen/Qwen2.5-0.5B-Instruct from HuggingFace → S3.
#
# What this does:
#   1. Reads config/config.env
#   2. Checks S3 — skips download if model already exists
#   3. Applies the Kubernetes Job (job.yaml)
#   4. Streams logs until job completes
#   5. Verifies files in S3
#   6. Deletes the job automatically
#
# Usage:
#   cd llm-inference-framework
#   bash model-download/qwen-2.5-0.5b/download.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_FOLDER="Qwen2.5-0.5B-Instruct"
JOB_NAME="download-qwen-0.5b"

echo ""
echo "══════════════════════════════════════════════════"
echo "  MODEL DOWNLOAD — Qwen2.5-0.5B-Instruct"
echo "  Bucket : s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
echo "  HF ID  : ${MODEL_ID}"
echo "  Token  : Not required"
echo "══════════════════════════════════════════════════"
echo ""

# ------------------------------------------------------------------------------
# Check if model already exists in S3
# ------------------------------------------------------------------------------
log_info "Checking S3 for existing model files..."
EXISTING=$(aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/model.safetensors" \
    --region "${AWS_REGION}" 2>/dev/null || true)

if [[ -n "${EXISTING}" ]]; then
    log_warn "Model already exists in S3 — skipping download."
    log_info "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
    aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/" --human-readable --region "${AWS_REGION}"
    exit 0
fi

# ------------------------------------------------------------------------------
# Check prerequisites
# ------------------------------------------------------------------------------
log_info "Checking prerequisites..."
if ! kubectl get serviceaccount "${SERVICE_ACCOUNT}" -n "${NAMESPACE}" >/dev/null 2>&1; then
    log_error "ServiceAccount '${SERVICE_ACCOUNT}' not found."
    log_error "Run: bash cluster/bootstrap.sh"
    exit 1
fi

# ------------------------------------------------------------------------------
# Clean up any existing job with same name
# ------------------------------------------------------------------------------
if kubectl get job "${JOB_NAME}" -n "${NAMESPACE}" >/dev/null 2>&1; then
    log_warn "Existing job '${JOB_NAME}' found — deleting before rerun..."
    kubectl delete job "${JOB_NAME}" -n "${NAMESPACE}"
    sleep 3
fi

# ------------------------------------------------------------------------------
# Apply the job
# ------------------------------------------------------------------------------
log_info "Applying download job..."

# Inject config values into job.yaml via envsubst
export MODEL_BUCKET AWS_REGION SERVICE_ACCOUNT NAMESPACE
envsubst < "${SCRIPT_DIR}/job.yaml" | kubectl apply -f -

log_info "Job '${JOB_NAME}' created."

# ------------------------------------------------------------------------------
# Wait for pod to start
# ------------------------------------------------------------------------------
log_info "Waiting for pod to start..."
for i in $(seq 1 30); do
    POD=$(kubectl get pods -l job-name="${JOB_NAME}" -n "${NAMESPACE}" \
        --field-selector=status.phase!=Pending \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [[ -n "${POD}" ]] && break
    echo -n "."
    sleep 5
done
echo ""

if [[ -z "${POD:-}" ]]; then
    # Pod still pending — get it anyway
    POD=$(kubectl get pods -l job-name="${JOB_NAME}" -n "${NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
fi

if [[ -z "${POD:-}" ]]; then
    log_error "Pod not found. Check: kubectl get pods -l job-name=${JOB_NAME}"
    exit 1
fi

log_info "Pod: ${POD}"

# ------------------------------------------------------------------------------
# Stream logs
# ------------------------------------------------------------------------------
log_info "Streaming logs (Ctrl+C to detach — job continues running)..."
echo ""
kubectl logs "${POD}" -n "${NAMESPACE}" -f 2>/dev/null || \
    kubectl logs "${POD}" -n "${NAMESPACE}" || true

# ------------------------------------------------------------------------------
# Wait for job completion
# ------------------------------------------------------------------------------
log_info "Waiting for job to complete..."
kubectl wait job "${JOB_NAME}" \
    --for=condition=complete \
    --timeout=7200s \
    -n "${NAMESPACE}" || {
    log_error "Job did not complete within 2 hours."
    log_error "Check: kubectl logs ${POD}"
    exit 1
}

log_info "Job completed successfully."

# ------------------------------------------------------------------------------
# Verify S3 upload
# ------------------------------------------------------------------------------
echo ""
log_info "Verifying S3 upload..."
aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/" \
    --recursive --human-readable --region "${AWS_REGION}"

# ------------------------------------------------------------------------------
# Delete the job
# ------------------------------------------------------------------------------
log_info "Cleaning up job..."
kubectl delete job "${JOB_NAME}" -n "${NAMESPACE}"
log_info "Job deleted."

echo ""
echo "══════════════════════════════════════════════════"
echo "  DOWNLOAD COMPLETE"
echo "  Model: s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
echo "  Next : bash vllm/models/qwen-2.5-0.5b/deploy.sh"
echo "══════════════════════════════════════════════════"
