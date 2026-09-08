#!/bin/bash
# ==============================================================================
# scripts/stop.sh
#
# Generic model stop — reads everything from models/<model-id>.yaml.
# Deletes deployment, service, PVC, terminates the GPU node, cleans jobs.
# No per-model scripts needed. No Python dependency.
#
# Usage:
#   bash scripts/stop.sh --model gpt-oss-20b
#   bash scripts/stop.sh --model qwen3-35b-nvfp4
#   bash scripts/stop.sh --all                  # stop ALL models + GPU nodes
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"
source "${FRAMEWORK_ROOT}/vllm/lib/gpu-terminate.sh"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

# ── Pure-bash YAML field reader — no Python needed ────────────────────────────
# Usage: yaml_field <file> <key>
# Reads a top-level scalar value from a YAML file using grep + sed.
yaml_field() {
    local file="$1" key="$2"
    grep -m1 "^${key}:" "${file}" | sed "s/^${key}:[[:space:]]*//" | sed 's/[[:space:]]*#.*//' | tr -d "'\"" | xargs
}

# ── Parse args ────────────────────────────────────────────────────────────────
MODEL=""
STOP_ALL="false"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --all)   STOP_ALL="true"; shift ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Stop ALL models ───────────────────────────────────────────────────────────
if [[ "${STOP_ALL}" == "true" ]]; then
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  FULL STOP — ALL MODELS"
    echo "══════════════════════════════════════════════════"
    echo "  This will stop every model and terminate ALL GPU nodes."
    echo ""
    read -r -p "Proceed? [y/N]: " CONFIRM
    [[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

    for f in "${FRAMEWORK_ROOT}/models/"*.yaml; do
        MID=$(yaml_field "${f}" "model_id")
        [[ -z "${MID}" ]] && continue
        DNAME="oai-infopt-vllm-${MID}"
        log_info "Stopping model: ${MID}"
        kubectl delete deployment "${DNAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
        kubectl delete svc "${DNAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
        kubectl delete pvc "oai-infopt-metadata-${MID}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
        kubectl delete jobs -n "${BENCHMARK_NAMESPACE}" \
            -l "app.kubernetes.io/component=benchmark-runner,model=${MID}" \
            --ignore-not-found=true 2>/dev/null || true
    done

    terminate_gpu_nodes
    print_stop_summary "all-models" "${BENCHMARK_NAMESPACE}"
    exit 0
fi

# ── Single model stop ─────────────────────────────────────────────────────────
if [[ -z "${MODEL}" ]]; then
    log_error "Usage: bash scripts/stop.sh --model <model-id>"
    log_error "       bash scripts/stop.sh --all"
    log_error "Available models:"
    ls "${FRAMEWORK_ROOT}/models/"*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | sed 's/^/  /'
    exit 1
fi

# ── Resolve model file ────────────────────────────────────────────────────────
MODEL_FILE="${FRAMEWORK_ROOT}/models/${MODEL}.yaml"
if [[ ! -f "${MODEL_FILE}" ]]; then
    # Try alias match
    FOUND=""
    for f in "${FRAMEWORK_ROOT}/models/"*.yaml; do
        ALIAS=$(yaml_field "${f}" "model_alias")
        if [[ "${ALIAS}" == "${MODEL}" ]]; then FOUND="${f}"; break; fi
    done
    [[ -n "${FOUND}" ]] && MODEL_FILE="${FOUND}" || {
        log_error "No model file found for '${MODEL}'"
        log_error "Available: $(ls "${FRAMEWORK_ROOT}/models/"*.yaml | xargs -I{} basename {} .yaml | tr '\n' ' ')"
        exit 1
    }
fi

# ── Load model fields ─────────────────────────────────────────────────────────
MODEL_ID=$(yaml_field "${MODEL_FILE}" "model_id")
MODEL_HF_ID=$(yaml_field "${MODEL_FILE}" "hf_id")

if [[ -z "${MODEL_ID}" ]]; then
    log_error "Could not read model_id from ${MODEL_FILE}"
    exit 1
fi

DEPLOYMENT_NAME="oai-infopt-vllm-${MODEL_ID}"
SERVICE_NAME="oai-infopt-vllm-${MODEL_ID}"
PVC_NAME="oai-infopt-metadata-${MODEL_ID}"

echo ""
echo "══════════════════════════════════════════════════"
echo "  STOPPING — ${MODEL_ID}"
echo "  HF ID      : ${MODEL_HF_ID}"
echo "  Deployment : ${DEPLOYMENT_NAME}"
echo "  Namespace  : ${BENCHMARK_NAMESPACE}"
echo "══════════════════════════════════════════════════"
echo ""
echo "  This will:"
echo "    • Delete vLLM deployment, service, PVC"
echo "    • Terminate the GPU node for this model"
echo "    • Clean up benchmark jobs for this model"
echo "    • Leave monitoring stack running (~\$0.15/hr)"
echo ""
read -r -p "Proceed? [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

# ── Capture node before deleting deployment ───────────────────────────────────
log_info "Looking up node for ${DEPLOYMENT_NAME}..."
MODEL_NODE=$(kubectl get pod \
    -l "app=${DEPLOYMENT_NAME}" \
    -n "${BENCHMARK_NAMESPACE}" \
    -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)
[[ -n "${MODEL_NODE}" ]] && log_info "Model running on node: ${MODEL_NODE}" || \
    log_warn "Pod not found — node will be located via nodeclaim."

# ── Delete k8s resources ──────────────────────────────────────────────────────
log_info "Deleting deployment: ${DEPLOYMENT_NAME}..."
kubectl delete deployment "${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true

log_info "Deleting service: ${SERVICE_NAME}..."
kubectl delete svc "${SERVICE_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true

log_info "Deleting PVC: ${PVC_NAME}..."
kubectl delete pvc "${PVC_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true

# ── Terminate GPU node ────────────────────────────────────────────────────────
terminate_model_gpu_node "${DEPLOYMENT_NAME}" "${BENCHMARK_NAMESPACE}" "${MODEL_NODE:-}"

# ── Clean benchmark jobs ──────────────────────────────────────────────────────
log_info "Cleaning up benchmark jobs for model '${MODEL_ID}'..."
kubectl delete jobs -n "${BENCHMARK_NAMESPACE}" \
    -l "app.kubernetes.io/component=benchmark-runner,model=${MODEL_ID}" \
    --ignore-not-found=true 2>/dev/null || true

# ── Summary ───────────────────────────────────────────────────────────────────
print_stop_summary "${MODEL_ID}" "${BENCHMARK_NAMESPACE}"
