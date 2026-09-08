#!/bin/bash
# ==============================================================================
# scripts/expose-public.sh
#
# Single-script automation to publicly expose or privatize any model.
# Automatically manages AWS Security Group firewall rules and K8s LoadBalancer.
#
# Usage:
#   bash scripts/expose-public.sh --model qwen-3.5-4b           # OPEN public access
#   bash scripts/expose-public.sh --model gpt-oss-20b          # OPEN public access
#   bash scripts/expose-public.sh --model qwen-3.5-4b --private # CLOSE public access
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

yaml_field() {
    local file="$1" key="$2"
    grep -m1 "^${key}:" "${file}" | sed "s/^${key}:[[:space:]]*//" | sed 's/[[:space:]]*#.*//' | tr -d "'\"" | xargs
}

MODEL=""
TYPE="LoadBalancer"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2"; shift 2 ;;
        --private) TYPE="ClusterIP"; shift ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    log_error "Usage: bash scripts/expose-public.sh --model <model-name> [--private]"
    exit 1
fi

MODEL_FILE="${FRAMEWORK_ROOT}/models/${MODEL}.yaml"
if [[ ! -f "${MODEL_FILE}" ]]; then
    FOUND=""
    for f in "${FRAMEWORK_ROOT}/models/"*.yaml; do
        ALIAS=$(yaml_field "${f}" "model_alias")
        if [[ "${ALIAS}" == "${MODEL}" ]]; then FOUND="${f}"; break; fi
    done
    [[ -n "${FOUND}" ]] && MODEL_FILE="${FOUND}" || {
        log_error "Model spec file not found for '${MODEL}'"
        exit 1
    }
fi

MODEL_ID=$(yaml_field "${MODEL_FILE}" "model_id")
SERVED_NAME=$(yaml_field "${MODEL_FILE}" "served_name")
SVC_NAME="oai-infopt-vllm-${MODEL_ID}"

# Fetch EKS Cluster Security Group ID
log_info "Detecting EKS Cluster Security Group for '${CLUSTER_NAME}' in region '${AWS_REGION}'..."
CLUSTER_SG=$(aws eks describe-cluster \
    --name "${CLUSTER_NAME}" \
    --region "${AWS_REGION}" \
    --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" \
    --output text 2>/dev/null || echo "")

if [[ "${TYPE}" == "LoadBalancer" ]]; then
    if [[ -n "${CLUSTER_SG}" ]]; then
        log_info "Opening AWS Security Group firewall rule on SG '${CLUSTER_SG}' (ports 30000-32767)..."
        aws ec2 authorize-security-group-ingress \
            --group-id "${CLUSTER_SG}" \
            --protocol tcp \
            --port 30000-32767 \
            --cidr 0.0.0.0/0 \
            --region "${AWS_REGION}" 2>/dev/null || true
    fi

    log_info "Configuring Service '${SVC_NAME}' as public LoadBalancer..."
    if ! kubectl get svc "${SVC_NAME}" -n "${BENCHMARK_NAMESPACE}" >/dev/null 2>&1; then
        kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: ${SVC_NAME}
  namespace: ${BENCHMARK_NAMESPACE}
  labels:
    app.kubernetes.io/name: ${SVC_NAME}
    app.kubernetes.io/component: inference-server
    project: observeai-inference-optimization
    model: ${MODEL_ID}
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-scheme: "internet-facing"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
  selector:
    app: ${SVC_NAME}
  ports:
    - name: http
      port: 8000
      targetPort: 8000
      protocol: TCP
EOF
    else
        kubectl annotate svc "${SVC_NAME}" -n "${BENCHMARK_NAMESPACE}" \
            service.beta.kubernetes.io/aws-load-balancer-scheme="internet-facing" \
            service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled="true" --overwrite >/dev/null 2>&1 || true
        kubectl patch svc "${SVC_NAME}" -n "${BENCHMARK_NAMESPACE}" -p '{"spec": {"type": "LoadBalancer"}}' >/dev/null 2>&1 || true
    fi
    log_info "Waiting for AWS LoadBalancer allocation..."
    ELB_HOST=""
    for i in $(seq 1 12); do
        ELB_HOST=$(kubectl get svc "${SVC_NAME}" -n "${BENCHMARK_NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
        [[ -n "${ELB_HOST}" ]] && break
        echo -n "."
        sleep 3
    done
    echo ""
    kubectl get svc "${SVC_NAME}" -n "${BENCHMARK_NAMESPACE}"
    
    # Auto-detect ELB Name and Register Karpenter Nodes
    ELB_NAME=$(echo "${ELB_HOST}" | cut -d'-' -f1)
    
    POD_NODE=$(kubectl get pods -n "${BENCHMARK_NAMESPACE}" -l "app=${SVC_NAME}" -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || echo "")
    if [[ -n "${POD_NODE}" ]]; then
        if [[ "${POD_NODE}" == i-* ]]; then
            INSTANCE_ID="${POD_NODE}"
        else
            INSTANCE_ID=$(aws ec2 describe-instances --filters "Name=private-dns-name,Values=${POD_NODE}*" --region "${AWS_REGION}" --query "Reservations[0].Instances[0].InstanceId" --output text 2>/dev/null || echo "")
        fi
        if [[ -n "${INSTANCE_ID}" && "${INSTANCE_ID}" != "None" && -n "${ELB_NAME}" ]]; then
            log_info "Registering Karpenter Node '${INSTANCE_ID}' with LoadBalancer '${ELB_NAME}'..."
            aws elb register-instances-with-load-balancer \
                --load-balancer-name "${ELB_NAME}" \
                --instances "${INSTANCE_ID}" \
                --region "${AWS_REGION}" >/dev/null 2>&1 || true

            # Authorize port 8000 on ELB Security Group
            ELB_SG_NAME=$(aws elb describe-load-balancers --load-balancer-names "${ELB_NAME}" --region "${AWS_REGION}" --query "LoadBalancerDescriptions[0].SourceSecurityGroup.GroupName" --output text 2>/dev/null || echo "")
            if [[ -n "${ELB_SG_NAME}" && "${ELB_SG_NAME}" != "None" ]]; then
                ELB_SG_ID=$(aws ec2 describe-security-groups --filters Name=group-name,Values="${ELB_SG_NAME}" --region "${AWS_REGION}" --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "")
                if [[ -n "${ELB_SG_ID}" && "${ELB_SG_ID}" != "None" ]]; then
                    log_info "Authorizing public port 8000 on ELB SG '${ELB_SG_ID}'..."
                    aws ec2 authorize-security-group-ingress \
                        --group-id "${ELB_SG_ID}" \
                        --protocol tcp \
                        --port 8000 \
                        --cidr 0.0.0.0/0 \
                        --region "${AWS_REGION}" >/dev/null 2>&1 || true
                fi
            fi
        fi
    fi

    echo ""
    log_info "PUBLIC ACCESS OPEN — Call your API from anywhere in the world:"
    echo "  curl http://<EXTERNAL-IP>:8000/v1/chat/completions \\"
    echo "    -H \"Content-Type: application/json\" \\"
    echo "    -d '{\"model\": \"${SERVED_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 100}'"
else
    if [[ -n "${CLUSTER_SG}" ]]; then
        log_info "Revoking AWS Security Group firewall rule on SG '${CLUSTER_SG}'..."
        aws ec2 revoke-security-group-ingress \
            --group-id "${CLUSTER_SG}" \
            --protocol tcp \
            --port 30000-32767 \
            --cidr 0.0.0.0/0 \
            --region "${AWS_REGION}" 2>/dev/null || true
    fi

    log_info "Reverting Service '${SVC_NAME}' to internal ClusterIP..."
    kubectl patch svc "${SVC_NAME}" -n "${BENCHMARK_NAMESPACE}" -p "{\"spec\": {\"type\": \"ClusterIP\"}}" 2>/dev/null || true
    log_info "PUBLIC ACCESS CLOSED — Cluster firewall and service locked down."
fi
