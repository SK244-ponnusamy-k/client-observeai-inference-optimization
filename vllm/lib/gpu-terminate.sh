#!/bin/bash
# ==============================================================================
# vllm/lib/gpu-terminate.sh
#
# Shared library — source this file, then call the appropriate function.
#
# Functions:
#   terminate_model_gpu_node <deployment_name> <namespace>
#     Finds the specific node running this model's pod and deletes it —
#     ONLY if no other inference-server pods are sharing that node.
#     Safe to call when multiple models are running simultaneously.
#
#   terminate_gpu_nodes
#     Kills ALL GPU nodes in the cluster regardless of what is running.
#     Used by vllm/stop.sh (stop everything).
#
#   print_stop_summary <model_id> <namespace>
#     Prints final status and safe-to-close message.
# ==============================================================================

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
_log_info() { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
_log_warn() { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }

# ------------------------------------------------------------------------------
# terminate_model_gpu_node <deployment_name> <namespace> [node_name]
#
# Per-model teardown:
#   - If node_name is passed (captured before deployment deletion), use it
#     directly — no pod lookup needed.
#   - If node_name is empty, falls back to nodeclaim instance-type lookup.
#   - Only deletes the node if no other inference-server pods share it.
# ------------------------------------------------------------------------------
terminate_model_gpu_node() {
    local deployment="${1}"
    local namespace="${2}"
    local node="${3:-}"   # pre-captured before deployment was deleted

    # If node was not pre-captured, try to look it up now (pod may be terminating)
    if [[ -z "${node}" ]]; then
        node=$(kubectl get pod \
            -l "app=${deployment}" \
            -n "${namespace}" \
            -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)
    fi

    if [[ -z "${node}" ]]; then
        _log_warn "Node name unknown — falling back to nodeclaim instance-type lookup..."
        # Find any GPU nodeclaim and delete it (last resort)
        local gpu_nc
        gpu_nc=$(kubectl get nodeclaims -o json 2>/dev/null | \
            python3 -c "
import sys, json
data = json.load(sys.stdin)
GPU_PREFIXES = ('g4', 'g5', 'g6', 'p3', 'p4', 'p5', 'inf')
for item in data.get('items', []):
    it = item.get('status', {}).get('instanceType', '')
    if any(it.startswith(p) for p in GPU_PREFIXES):
        print(item['metadata']['name'])
        break
" 2>/dev/null || true)
        if [[ -n "${gpu_nc}" ]]; then
            _log_info "Deleting GPU nodeclaim: ${gpu_nc}"
            kubectl delete nodeclaim "${gpu_nc}" --ignore-not-found=true
            _log_info "GPU node deleted — billing stops within minutes."
        else
            _log_warn "No GPU nodeclaims found. Verify manually: kubectl get nodeclaims"
        fi
        return
    fi

    _log_info "Targeting node: ${node}"

    # Check if any OTHER inference-server pods are on the same node
    local other_pods
    other_pods=$(kubectl get pod \
        -l "app.kubernetes.io/component=inference-server" \
        --all-namespaces \
        -o json 2>/dev/null | \
        python3 -c "
import sys, json
data = json.load(sys.stdin)
node = '${node}'
this_deploy = '${deployment}'
others = []
for i in data.get('items', []):
    if i.get('spec', {}).get('nodeName') == node:
        labels = i.get('metadata', {}).get('labels', {})
        # Exclude pods belonging to this deployment
        if labels.get('app') != this_deploy and \
           labels.get('app.kubernetes.io/name') != this_deploy:
            others.append(i['metadata']['name'])
print('\n'.join(others))
" 2>/dev/null || true)

    if [[ -n "${other_pods}" ]]; then
        _log_warn "Node '${node}' is shared with other model pods:"
        echo "${other_pods}" | while read -r p; do _log_warn "  - ${p}"; done
        _log_warn "Node NOT deleted — other models still using it."
        _log_warn "To stop all models: bash vllm/stop.sh"
        return
    fi

    # Node is exclusively used by this model — find and delete its NodeClaim
    local nodeclaim
    nodeclaim=$(kubectl get nodeclaims -o json 2>/dev/null | \
        python3 -c "
import sys, json
data = json.load(sys.stdin)
for item in data.get('items', []):
    if item.get('status', {}).get('nodeName') == '${node}':
        print(item['metadata']['name'])
        break
" 2>/dev/null || true)

    if [[ -n "${nodeclaim}" ]]; then
        _log_info "Deleting nodeclaim: ${nodeclaim} (node: ${node})"
        kubectl delete nodeclaim "${nodeclaim}" --ignore-not-found=true
    else
        _log_warn "NodeClaim not found for node '${node}' — deleting node directly."
        kubectl delete node "${node}" --ignore-not-found=true
    fi

    _log_info "GPU node deleted — billing stops within minutes."
}

# ------------------------------------------------------------------------------
# terminate_gpu_nodes
#
# Kills ALL GPU nodes — used by vllm/stop.sh only.
# Uses 3 independent methods so nothing is missed regardless of nodepool name.
# ------------------------------------------------------------------------------
terminate_gpu_nodes() {
    _log_info "Terminating ALL GPU nodes..."

    # Method A: NodeClaims by GPU instance type prefix
    local gpu_nodeclaims
    gpu_nodeclaims=$(kubectl get nodeclaims -o json 2>/dev/null | \
        python3 -c "
import sys, json
data = json.load(sys.stdin)
GPU_PREFIXES = ('g4', 'g5', 'g6', 'p3', 'p4', 'p5', 'inf')
for item in data.get('items', []):
    it = item.get('status', {}).get('instanceType', '')
    if any(it.startswith(p) for p in GPU_PREFIXES):
        print(item['metadata']['name'])
" 2>/dev/null || true)

    if [[ -n "${gpu_nodeclaims}" ]]; then
        echo "${gpu_nodeclaims}" | while read -r nc; do
            _log_info "  Deleting nodeclaim: ${nc}"
            kubectl delete nodeclaim "${nc}" --ignore-not-found=true 2>/dev/null || true
        done
    else
        _log_warn "  No GPU nodeclaims found by instance type — trying labels..."
    fi

    # Method B: NodeClaims by all known GPU nodepool names
    for pool in gpu-nodepool gpu-inf gpu gpu-nodes; do
        local count
        count=$(kubectl get nodeclaims -l "karpenter.sh/nodepool=${pool}" \
            --no-headers 2>/dev/null | wc -l | tr -d ' ')
        if [[ "${count}" -gt 0 ]]; then
            _log_info "  Deleting ${count} nodeclaim(s) in nodepool: ${pool}"
            kubectl delete nodeclaims -l "karpenter.sh/nodepool=${pool}" \
                --ignore-not-found=true 2>/dev/null || true
        fi
    done

    # Method C: Kubernetes nodes by GPU labels
    for selector in \
        "karpenter.sh/nodepool=gpu-nodepool" \
        "karpenter.sh/nodepool=gpu-inf" \
        "karpenter.sh/nodepool=gpu" \
        "eks.amazonaws.com/instance-gpu-manufacturer=nvidia" \
        "nvidia.com/gpu=true"; do
        local count
        count=$(kubectl get nodes -l "${selector}" \
            --no-headers 2>/dev/null | wc -l | tr -d ' ')
        if [[ "${count}" -gt 0 ]]; then
            _log_info "  Deleting ${count} node(s) with label: ${selector}"
            kubectl delete nodes -l "${selector}" --ignore-not-found=true 2>/dev/null || true
        fi
    done

    _log_info "GPU node termination sent — billing stops within minutes."
}

# Internal helper used as fallback in terminate_model_gpu_node
_terminate_node_by_labels() {
    for pool in gpu-nodepool gpu-inf gpu; do
        local count
        count=$(kubectl get nodeclaims -l "karpenter.sh/nodepool=${pool}" \
            --no-headers 2>/dev/null | wc -l | tr -d ' ')
        if [[ "${count}" -gt 0 ]]; then
            _log_info "  Deleting ${count} nodeclaim(s) in nodepool: ${pool}"
            kubectl delete nodeclaims -l "karpenter.sh/nodepool=${pool}" \
                --ignore-not-found=true 2>/dev/null || true
        fi
    done
}

# ------------------------------------------------------------------------------
# print_stop_summary <model_id> <namespace>
# ------------------------------------------------------------------------------
print_stop_summary() {
    local model_id="${1:-unknown}"
    local namespace="${2:-oai-infopt}"

    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  FINAL STATUS"
    echo "══════════════════════════════════════════════════"

    echo ""
    _log_info "GPU nodes remaining:"
    local gpu_left
    gpu_left=$(kubectl get nodeclaims -o json 2>/dev/null | \
        python3 -c "
import sys, json
data = json.load(sys.stdin)
GPU_PREFIXES = ('g4', 'g5', 'g6', 'p3', 'p4', 'p5', 'inf')
gpu = [
    '{} ({})'.format(i['metadata']['name'], i.get('status',{}).get('instanceType','?'))
    for i in data.get('items', [])
    if any(i.get('status',{}).get('instanceType','').startswith(p) for p in GPU_PREFIXES)
]
print('\n'.join('  ' + n for n in gpu) if gpu else '  none — GPU billing stopped ✓')
" 2>/dev/null || echo "  (could not query nodeclaims)")
    echo "${gpu_left}"

    echo ""
    _log_info "Remaining workloads in ${namespace}:"
    local remaining
    remaining=$(kubectl get all -n "${namespace}" --no-headers 2>/dev/null | \
        grep -v "^$" | wc -l | tr -d ' ')
    if [[ "${remaining}" -eq 0 ]]; then
        echo "  none — namespace is clean ✓"
    else
        kubectl get all -n "${namespace}" 2>/dev/null || true
    fi

    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  STOPPED — ${model_id}"
    echo "  Safe to close your laptop ✓"
    echo ""
    echo "  Still running (expected, cheap CPU nodes ~\$0.15/hr):"
    echo "    • Grafana / Prometheus — monitoring stack"
    echo "    • metrics-collector cronjob"
    echo ""
    echo "  To redeploy:"
    echo "    bash vllm/models/gpt-oss-20b/deploy.sh"
    echo "    bash vllm/models/qwen-2.5-0.5b/deploy.sh"
    echo "══════════════════════════════════════════════════"
}
