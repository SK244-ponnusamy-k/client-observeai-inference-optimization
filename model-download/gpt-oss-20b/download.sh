#!/bin/bash
# ==============================================================================
# model-download/gpt-oss-20b/download.sh
#
# Downloads openai/gpt-oss-20b from HuggingFace → S3.
#
# What this does:
#   1. Reads config/config.env
#   2. Checks S3 — skips download if model already exists
#   3. Checks HF token secret exists in cluster
#   4. Applies the Kubernetes Job (job.yaml)
#   5. Streams logs until job completes
#   6. Verifies files in S3
#   7. Deletes the job automatically
#
# Prerequisites:
#   1. Accept license at: https://huggingface.co/openai/gpt-oss-20b
#   2. Create HF token secret:
#        kubectl create secret generic hf-token \
#          --from-literal=token=hf_YOURTOKEN
#
# Usage:
#   cd llm-inference-framework
#   bash model-download/gpt-oss-20b/download.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

MODEL_ID="openai/gpt-oss-20b"
MODEL_FOLDER="gpt-oss-20b"
JOB_NAME="download-gpt-oss-20b"

echo ""
echo "══════════════════════════════════════════════════"
echo "  MODEL DOWNLOAD — openai/gpt-oss-20b"
echo "  Bucket : s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
echo "  HF ID  : ${MODEL_ID}"
echo "  Token  : Required (gated model)"
echo "  Size   : ~13 GB (MXFP4 quantized)"
echo "══════════════════════════════════════════════════"
echo ""

# ------------------------------------------------------------------------------
# Check if model already exists in S3
# ------------------------------------------------------------------------------
log_info "Checking S3 for existing model files..."
EXISTING=$(aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/model-00000-of-00002.safetensors" \
    --region "${AWS_REGION}" 2>/dev/null || true)

if [[ -n "${EXISTING}" ]]; then
    log_warn "Model already exists in S3 — skipping download."
    aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/" --human-readable --region "${AWS_REGION}"
    exit 0
fi

# ------------------------------------------------------------------------------
# Check HF token secret
# ------------------------------------------------------------------------------
log_info "Checking HuggingFace token secret..."
if ! kubectl get secret hf-token -n "${NAMESPACE}" >/dev/null 2>&1; then
    log_error "HuggingFace token secret not found."
    log_error "Create it first:"
    log_error "  kubectl create secret generic hf-token --from-literal=token=hf_YOURTOKEN"
    log_error ""
    log_error "Also accept the license at:"
    log_error "  https://huggingface.co/openai/gpt-oss-20b"
    exit 1
fi
log_info "HF token secret found."

# ------------------------------------------------------------------------------
# Check prerequisites
# ------------------------------------------------------------------------------
if ! kubectl get serviceaccount "${SERVICE_ACCOUNT}" -n "${NAMESPACE}" >/dev/null 2>&1; then
    log_error "ServiceAccount '${SERVICE_ACCOUNT}' not found."
    log_error "Run: bash cluster/bootstrap.sh"
    exit 1
fi

# ------------------------------------------------------------------------------
# Clean up existing job
# ------------------------------------------------------------------------------
if kubectl get job "${JOB_NAME}" -n "${NAMESPACE}" >/dev/null 2>&1; then
    log_warn "Existing job '${JOB_NAME}' found — deleting..."
    kubectl delete job "${JOB_NAME}" -n "${NAMESPACE}"
    sleep 3
fi

# ------------------------------------------------------------------------------
# Apply job
# ------------------------------------------------------------------------------
log_info "Applying download job..."
export MODEL_BUCKET AWS_REGION SERVICE_ACCOUNT NAMESPACE
envsubst < "${SCRIPT_DIR}/job.yaml" | kubectl apply -f -
log_info "Job '${JOB_NAME}' created. (~13GB — expect 20-60 minutes)"

# ------------------------------------------------------------------------------
# Wait for pod to start
# ------------------------------------------------------------------------------
log_info "Waiting for pod to start..."
for i in $(seq 1 30); do
    POD=$(kubectl get pods -l job-name="${JOB_NAME}" -n "${NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [[ -n "${POD}" ]] && break
    echo -n "."
    sleep 5
done
echo ""

if [[ -z "${POD:-}" ]]; then
    log_error "Pod not found. Check: kubectl get pods -l job-name=${JOB_NAME}"
    exit 1
fi
log_info "Pod: ${POD}"

# ------------------------------------------------------------------------------
# Stream logs
# ------------------------------------------------------------------------------
log_info "Streaming logs (Ctrl+C to detach — job continues)..."
echo ""
kubectl logs "${POD}" -n "${NAMESPACE}" -f 2>/dev/null || \
    kubectl logs "${POD}" -n "${NAMESPACE}" || true

# ------------------------------------------------------------------------------
# Wait for job completion
# ------------------------------------------------------------------------------
log_info "Waiting for job to complete (timeout: 24h)..."
kubectl wait job "${JOB_NAME}" \
    --for=condition=complete \
    --timeout=86400s \
    -n "${NAMESPACE}" || {
    log_error "Job did not complete. Check: kubectl logs ${POD}"
    exit 1
}

log_info "Job completed."

# ------------------------------------------------------------------------------
# Verify S3
# ------------------------------------------------------------------------------
echo ""
log_info "Verifying S3 upload..."
aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/" \
    --recursive --human-readable --region "${AWS_REGION}"

# ------------------------------------------------------------------------------
# Delete job
# ------------------------------------------------------------------------------
log_info "Cleaning up job..."
kubectl delete job "${JOB_NAME}" -n "${NAMESPACE}"
log_info "Job deleted."

echo ""
echo "══════════════════════════════════════════════════"
echo "  DOWNLOAD COMPLETE"
echo "  Model: s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
echo "  Next : bash vllm/models/gpt-oss-20b/deploy.sh"
echo "══════════════════════════════════════════════════"
