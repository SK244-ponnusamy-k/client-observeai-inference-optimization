#!/bin/bash
# ==============================================================================
# vllm/models/gpt-oss-20b-neuron/stop.sh
#
# Full teardown of the gpt-oss-20b Neuron / trn2 serving deployment.
#
# What this does:
#   1. Deletes the vLLM-Neuron deployment and service
#   2. Scales the trn2 managed node group to 0 (stops billing — ~$98/hr)
#      OR deletes the Karpenter node if using Auto Mode (trn1)
#   3. Cleans up the compile job if still present
#   4. Confirms nothing expensive is left running
#
# What this does NOT touch:
#   - S3 bf16 weights  (s3://.../gpt-oss-20b/)       — safe, no change
#   - S3 NEFF cache    (s3://.../gpt-oss-20b-neuron/) — safe, kept for next run
#   - Monitoring stack (Grafana/Prometheus on cheap CPU nodes)
#   - GPU deployments or other model deployments
#
# Usage:
#   cd llm-inference-framework
#   bash vllm/models/gpt-oss-20b-neuron/stop.sh
#   bash vllm/models/gpt-oss-20b-neuron/stop.sh --managed-ng   # scale down node group
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"
source "${SCRIPT_DIR}/model.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }

USE_MANAGED_NG="false"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --managed-ng) USE_MANAGED_NG="true"; shift ;;
        *) log_warn "Unknown argument: $1"; shift ;;
    esac
done

# Auto-detect managed-ng if TRN2_NODEGROUP_NAME is set
if [[ -n "${TRN2_NODEGROUP_NAME:-}" && "${USE_MANAGED_NG}" == "false" ]]; then
    NG_STATUS=$(aws eks describe-nodegroup \
        --cluster-name "${CLUSTER_NAME}" \
        --nodegroup-name "${TRN2_NODEGROUP_NAME}" \
        --region "${AWS_REGION}" \
        --query 'nodegroup.status' \
        --output text 2>/dev/null || echo "NOT_FOUND")
    if [[ "${NG_STATUS}" != "NOT_FOUND" ]]; then
        USE_MANAGED_NG="true"
        log_info "Auto-detected managed node group: ${TRN2_NODEGROUP_NAME}"
    fi
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  FULL STOP (Neuron / trn2) — ${MODEL_ID}"
echo "  Deployment : ${DEPLOYMENT_NAME}"
echo "  Namespace  : ${BENCHMARK_NAMESPACE}"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  This will:"
echo "    • Delete vLLM-Neuron deployment and service"
if [[ "${USE_MANAGED_NG}" == "true" ]]; then
echo "    • Scale trn2 node group to 0 (~\$98/hr — stops billing)"
else
echo "    • Terminate the Trainium node (stops billing)"
fi
echo "    • Delete the compile job (if present)"
echo "    • Leave S3 weights intact (reusable for next deploy)"
echo "    • Leave monitoring stack running"
echo ""

read -r -p "Proceed? [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

# ── Step 1: Delete vLLM-Neuron deployment ────────────────────────────────────
log_info "Deleting deployment: ${DEPLOYMENT_NAME}..."
kubectl delete deployment "${DEPLOYMENT_NAME}" \
    -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true

log_info "Deleting service: ${SERVICE_NAME}..."
kubectl delete svc "${SERVICE_NAME}" \
    -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true

# ── Step 2: Delete compile job (if still present) ────────────────────────────
log_info "Deleting compile job (if present)..."
kubectl delete job oai-infopt-neuron-compile-gpt-oss-20b \
    -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true

# ── Step 3: Stop billing ──────────────────────────────────────────────────────
if [[ "${USE_MANAGED_NG}" == "true" && -n "${TRN2_NODEGROUP_NAME:-}" ]]; then
    # Managed node group — scale to 0 (instance terminates, billing stops)
    log_info "Scaling managed node group to 0: ${TRN2_NODEGROUP_NAME}..."
    aws eks update-nodegroup-config \
        --cluster-name "${CLUSTER_NAME}" \
        --nodegroup-name "${TRN2_NODEGROUP_NAME}" \
        --scaling-config minSize=0,maxSize=1,desiredSize=0 \
        --region "${AWS_REGION}" >/dev/null
    log_info "Node group scaled to 0. trn2 instance will terminate shortly."
    log_info "Billing stops when instance terminates (~2-3 min)."
else
    # Karpenter / Auto Mode — delete the node directly
    log_info "Looking up Trainium node for deployment: ${DEPLOYMENT_NAME}..."
    MODEL_NODE=$(kubectl get pod \
        -l "app=${DEPLOYMENT_NAME}" \
        -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)

    if [[ -n "${MODEL_NODE}" ]]; then
        log_info "Deleting node: ${MODEL_NODE}..."
        kubectl delete node "${MODEL_NODE}" --ignore-not-found=true
        sleep 5
        NODECLAIM=$(kubectl get nodeclaim \
            -o jsonpath="{.items[?(@.status.nodeName=='${MODEL_NODE}')].metadata.name}" \
            2>/dev/null || true)
        if [[ -n "${NODECLAIM}" ]]; then
            log_info "Deleting NodeClaim: ${NODECLAIM}..."
            kubectl delete nodeclaim "${NODECLAIM}" --ignore-not-found=true
        fi
    else
        log_warn "No pod found — node may already be gone."
        TRN_NODES=$(kubectl get nodes \
            -l "eks.amazonaws.com/instance-family=trn2" \
            -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
        if [[ -n "${TRN_NODES}" ]]; then
            log_warn "Found trn2 nodes still running: ${TRN_NODES}"
            log_warn "Delete manually if no other workloads: kubectl delete node <name>"
        fi
    fi
fi

# ── Step 4: Final status ──────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  STOP COMPLETE — ${MODEL_ID}"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  Remaining pods in ${BENCHMARK_NAMESPACE}:"
kubectl get pods -n "${BENCHMARK_NAMESPACE}" -o wide 2>/dev/null || true
echo ""
if [[ "${USE_MANAGED_NG}" == "true" ]]; then
    echo "  Node group status (should show desiredSize=0):"
    aws eks describe-nodegroup \
        --cluster-name "${CLUSTER_NAME}" \
        --nodegroup-name "${TRN2_NODEGROUP_NAME}" \
        --region "${AWS_REGION}" \
        --query 'nodegroup.{Status:status,Desired:scalingConfig.desiredSize}' \
        --output json 2>/dev/null || true
fi
echo ""
echo "  S3 weights preserved — next deploy will reuse them:"
echo "    s3://${MODEL_BUCKET}/${MODEL_FOLDER}/"
echo ""
echo "  To redeploy:"
echo "    bash vllm/models/gpt-oss-20b-neuron/deploy.sh --managed-ng"
echo "══════════════════════════════════════════════════════════════"
