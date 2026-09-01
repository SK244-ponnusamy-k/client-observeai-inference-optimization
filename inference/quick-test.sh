#!/bin/bash
# ==============================================================================
# inference/quick-test.sh
#
# Sends a single chat completion request to the running vLLM service.
#
# Usage:
#   cd llm-inference-framework
#   bash inference/quick-test.sh <model-name> [port]
#
# Examples:
#   bash inference/quick-test.sh qwen-0.5b
#   bash inference/quick-test.sh gpt-oss-20b
#   bash inference/quick-test.sh qwen-0.5b 8080
#
# Prerequisites:
#   kubectl port-forward svc/vllm-inference 8080:8000 &
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

MODEL_NAME="${1:-qwen-0.5b}"
PORT="${2:-8080}"
BASE_URL="http://localhost:${PORT}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

echo ""
echo "══════════════════════════════════════════════════"
echo "  QUICK TEST"
echo "  Model   : ${MODEL_NAME}"
echo "  Endpoint: ${BASE_URL}"
echo "══════════════════════════════════════════════════"
echo ""

# Check vLLM is reachable
log_info "Checking vLLM health..."
if ! curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
    log_error "vLLM not reachable at ${BASE_URL}"
    log_error "Start port-forward first:"
    log_error "  kubectl port-forward svc/vllm-inference 8080:8000 &"
    exit 1
fi
log_info "vLLM is healthy."

# List models
echo ""
log_info "Available models:"
curl -s "${BASE_URL}/v1/models"
echo ""
echo ""

# Chat completion
log_info "Sending chat completion..."
echo "──────────────────────────────────────────────────"
curl -s -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_NAME}\",
        \"messages\": [
            {\"role\": \"system\", \"content\": \"You are a concise assistant.\"},
            {\"role\": \"user\", \"content\": \"Explain Kubernetes in two sentences.\"}
        ],
        \"max_tokens\": 256,
        \"temperature\": 0.2
    }"
echo ""
echo "──────────────────────────────────────────────────"
log_info "Test complete."
