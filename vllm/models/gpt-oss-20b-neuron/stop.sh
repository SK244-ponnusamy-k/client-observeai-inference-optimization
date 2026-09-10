#!/bin/bash
# ==============================================================================
# vllm/models/gpt-oss-20b-neuron/stop.sh
#
# Full teardown of the gpt-oss-20b Neuron / trn1 serving deployment.
#
# What this does:
#   1. Deletes the vLLM-Neuron deployment and service
#   2. Deletes the Trainium node this model was running on (stops billing —
#      trn1.32xlarge costs ~$21.50/hr on-demand)
#   3. Cleans up the compile job if still present
#   4. Confirms nothing expensive is left running
#
# What this does NOT touch:
#   - S3 bf16 weights  (s3://.../gpt-oss-20b/)        — safe, no change
#   - S3 NEFF cache    (s3://.../gpt-oss-20b-neuron/)  — safe, kept for next run
#   - Monitoring stack (Grafana/Prometheus on cheap CPU nodes)
#   - GPU deployments or other model deployments
#
# Usage:
#   cd llm-inference-framework
#   bash vllm/models/gpt-oss-20b-neuron/stop.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"
source "${SCRIPT_DIR}/model.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  FULL STOP (Neuron / trn1) — ${MODEL_ID}"
echo "  Deployment : ${DEPLOYMENT_NAME}"
echo "  Namespace  : ${BENCHMARK_NAMESPACE}"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  This will:"
echo "    • Delete vLLM-Neuron deployment and service"
echo "    • Terminate the trn1 node (~\$21.50/hr — stops billing)"
echo "    • Delete the compile job (if present)"
echo "    • Leave S3 NEFF cache intact (reusable for next deploy)"
echo "    • Leave monitoring stack running"
echo ""

read -r -p "Proceed? [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

# ── Step 1: Capture the trn1 node BEFORE deleting the deployment ──────────────
# Pod disappears once the deployment is deleted — find the node first.
log_info "Looking up trn1 node for deployment: ${DEPLOYMENT_NAME}..."
MODEL_NODE=$(kubectl get pod \
    -l "app=${DEPLOYMENT_NAME}" \
    -n "${BENCHMARK_NAMESPACE}" \
    -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)

if [[ -z "${MODEL_NODE}" ]]; then
    MODEL_NODE=$(kubectl get pod \
        -l "app.kubernetes.io/name=${DEPLOYMENT_NAME}" \
        -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)
fi

if [[ -n "${MODEL_NODE}" ]]; then
    log_info "Model is running on node: ${MODEL_NODE}"
else
    log_warn "Pod not found — will attempt to locate trn1 node via NodeClaim label."
fi

# ── Step 2: Delete vLLM-Neuron deployment ────────────────────────────────────
log_info "Deleting deployment: ${DEPLOYMENT_NAME}..."
kubectl delete deployment "${DEPLOYMENT_NAME}" \
    -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true

log_info "Deleting service: ${SERVICE_NAME}..."
kubectl delete svc "${SERVICE_NAME}" \
    -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true

# ── Step 3: Delete compile job (if still present) ────────────────────────────
log_info "Deleting compile job (if present)..."
kubectl delete job oai-infopt-neuron-compile-gpt-oss-20b \
    -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true

# ── Step 4: Terminate the trn1 node ──────────────────────────────────────────
# Trainium nodes are expensive. Delete the specific node rather than draining
# the entire nodepool — other trn1 nodes (if any) are left untouched.
if [[ -n "${MODEL_NODE}" ]]; then
    log_info "Deleting trn1 node: ${MODEL_NODE}..."
    kubectl delete node "${MODEL_NODE}" --ignore-not-found=true

    # Wait a moment then check for the associated NodeClaim and delete it so
    # Karpenter does not immediately re-provision.
    sleep 5
    NODECLAIM=$(kubectl get nodeclaim \
        -o jsonpath="{.items[?(@.status.nodeName=='${MODEL_NODE}')].metadata.name}" \
        2>/dev/null || true)
    if [[ -n "${NODECLAIM}" ]]; then
        log_info "Deleting NodeClaim: ${NODECLAIM}..."
        kubectl delete nodeclaim "${NODECLAIM}" --ignore-not-found=true
    fi
else
    log_warn "Node name unknown — attempting to locate via instance-family label..."
    # Fallback: find any trn1 node with no remaining Neuron workloads
    TRN1_NODES=$(kubectl get nodes \
        -l "eks.amazonaws.com/instance-family=trn1" \
        -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
    if [[ -n "${TRN1_NODES}" ]]; then
        log_warn "Found trn1 nodes: ${TRN1_NODES}"
        log_warn "Verify no other Neuron workloads are running before deleting manually:"
        log_warn "  kubectl delete node <node-name>"
    else
        log_warn "No trn1 nodes found in cluster — may have already been cleaned up."
    fi
fi

# ── Step 5: Final status ──────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  STOP COMPLETE — ${MODEL_ID}"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  Remaining pods in ${BENCHMARK_NAMESPACE}:"
kubectl get pods -n "${BENCHMARK_NAMESPACE}" -o wide 2>/dev/null || true
echo ""
echo "  Remaining trn1 nodes:"
kubectl get nodes -l "eks.amazonaws.com/instance-family=trn1" 2>/dev/null \
    || echo "  (none)"
echo ""
echo "  S3 NEFF cache preserved at:"
echo "    s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
echo "  To redeploy without recompiling:"
echo "    bash vllm/models/gpt-oss-20b-neuron/deploy.sh"
echo "══════════════════════════════════════════════════════════════"
