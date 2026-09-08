#!/bin/bash
# ==============================================================================
# model-download/qwen3-35b-nvfp4/download.sh
#
# Downloads nvidia/Qwen3.6-35B-A3B-NVFP4 from HuggingFace → S3.
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
#   1. Accept license at: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
#   2. Ensure HF token secret is synced (ExternalSecret from Secrets Manager):
#        kubectl get secret hf-token -n oai-infopt
#      Or create manually:
#        kubectl create secret generic hf-token \
#          --from-literal=token=hf_YOURTOKEN \
#          -n oai-infopt
#
# Model info:
#   Repo     : nvidia/Qwen3.6-35B-A3B-NVFP4
#   Size     : ~20 GB (NVFP4 quantized)
#   Type     : MoE — 35.7B total params, ~3.6B active per token
#   Requires : NVIDIA NIM / vLLM 0.8.5+ with NVFP4 kernel support
#
# Usage:
#   cd llm-inference-framework
#   bash model-download/qwen3-35b-nvfp4/download.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

MODEL_ID="nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_FOLDER="Qwen3.6-35B-A3B-NVFP4"
JOB_NAME="oai-infopt-download-qwen3-35b-nvfp4"

echo ""
echo "══════════════════════════════════════════════════"
echo "  MODEL DOWNLOAD — nvidia/Qwen3.6-35B-A3B-NVFP4"
echo "  Bucket : s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
echo "  HF ID  : ${MODEL_ID}"
echo "  Token  : Required (gated model — NVIDIA org)"
echo "  Size   : ~20 GB (NVFP4 quantized MoE)"
echo "══════════════════════════════════════════════════"
echo ""

# ------------------------------------------------------------------------------
# Check if model already exists in S3
# ------------------------------------------------------------------------------
log_info "Checking S3 for existing model files..."
EXISTING=$(aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/" \
    --region "${AWS_REGION}" 2>/dev/null | head -1 || true)

if [[ -n "${EXISTING}" ]]; then
    log_warn "Model already exists in S3 — skipping download."
    aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_FOLDER}/" --human-readable --region "${AWS_REGION}"
    exit 0
fi

# ------------------------------------------------------------------------------
# Check HF token secret
# ------------------------------------------------------------------------------
log_info "Checking HuggingFace token secret in namespace ${BENCHMARK_NAMESPACE}..."
if ! kubectl get secret hf-token -n "${BENCHMARK_NAMESPACE}" >/dev/null 2>&1; then
    log_error "HuggingFace token secret not found in namespace '${BENCHMARK_NAMESPACE}'."
    log_error "Create it first:"
    log_error "  kubectl create secret generic hf-token \\"
    log_error "    --from-literal=token=hf_YOURTOKEN \\"
    log_error "    -n ${BENCHMARK_NAMESPACE}"
    log_error ""
    log_error "Also accept the license at:"
    log_error "  https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4"
    exit 1
fi
log_info "HF token secret found."

# ------------------------------------------------------------------------------
# Check prerequisites
# ------------------------------------------------------------------------------
if ! kubectl get serviceaccount "${DOWNLOAD_SERVICE_ACCOUNT}" -n "${BENCHMARK_NAMESPACE}" >/dev/null 2>&1; then
    log_error "ServiceAccount '${DOWNLOAD_SERVICE_ACCOUNT}' not found in namespace '${BENCHMARK_NAMESPACE}'."
    log_error "Run: bash cluster/bootstrap.sh"
    exit 1
fi
log_info "ServiceAccount '${DOWNLOAD_SERVICE_ACCOUNT}' found."

# ------------------------------------------------------------------------------
# Clean up existing job
# ------------------------------------------------------------------------------
if kubectl get job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" >/dev/null 2>&1; then
    log_warn "Existing job '${JOB_NAME}' found — deleting before rerun..."
    kubectl delete job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}"
    sleep 3
fi

# ------------------------------------------------------------------------------
# Apply job
# ------------------------------------------------------------------------------
log_info "Applying download job..."
export MODEL_BUCKET AWS_REGION DOWNLOAD_SERVICE_ACCOUNT BENCHMARK_NAMESPACE
# Explicit variable list — only substitute the 4 runtime vars.
# All other ${VAR} references inside the pod script (MODEL_ID, MODEL_FOLDER,
# LOCAL_DIR) are intentionally left for bash to resolve inside the container.
envsubst '${MODEL_BUCKET} ${AWS_REGION} ${DOWNLOAD_SERVICE_ACCOUNT} ${BENCHMARK_NAMESPACE}' \
    < "${SCRIPT_DIR}/job.yaml" | kubectl apply -f -
log_info "Job '${JOB_NAME}' created. (~20 GB — expect 30-90 minutes)"

# ------------------------------------------------------------------------------
# Wait for pod to start
# ------------------------------------------------------------------------------
log_info "Waiting for pod to start..."
for i in $(seq 1 36); do
    POD=$(kubectl get pods -l job-name="${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [[ -n "${POD}" ]] && break
    echo -n "."
    sleep 5
done
echo ""

if [[ -z "${POD:-}" ]]; then
    log_error "Pod not found after 3 minutes."
    log_error "Check: kubectl get pods -l job-name=${JOB_NAME} -n ${BENCHMARK_NAMESPACE}"
    exit 1
fi
log_info "Pod: ${POD}"

# ------------------------------------------------------------------------------
# Stream logs
# ------------------------------------------------------------------------------
log_info "Streaming logs (Ctrl+C to detach — job continues)..."
echo ""
kubectl logs "${POD}" -n "${BENCHMARK_NAMESPACE}" -f 2>/dev/null || \
    kubectl logs "${POD}" -n "${BENCHMARK_NAMESPACE}" || true

# ------------------------------------------------------------------------------
# Wait for job completion
# ------------------------------------------------------------------------------
log_info "Waiting for job to complete (timeout: 24h)..."
kubectl wait job "${JOB_NAME}" \
    --for=condition=complete \
    --timeout=86400s \
    -n "${BENCHMARK_NAMESPACE}" || {
    log_error "Job did not complete. Check:"
    log_error "  kubectl logs ${POD} -n ${BENCHMARK_NAMESPACE}"
    exit 1
}
log_info "Job completed successfully."

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
kubectl delete job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}"
log_info "Job deleted."

echo ""
echo "══════════════════════════════════════════════════"
echo "  DOWNLOAD COMPLETE"
echo "  Model: s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
echo "  Next : bash vllm/models/qwen3-35b-nvfp4/deploy.sh"
echo "══════════════════════════════════════════════════"
