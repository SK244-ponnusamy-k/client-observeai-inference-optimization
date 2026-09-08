#!/bin/bash
# ==============================================================================
# cluster/teardown.sh
#
# Safely deletes the EKS cluster and ALL associated AWS resources.
# Handles VPC dependency cleanup automatically to prevent stuck deletions.
#
# What this deletes:
#   - Kubernetes: LoadBalancer services, Ingress, vLLM deployment, PVCs
#   - AWS: NAT Gateways, Internet Gateways, ENIs, Security Groups, VPC
#   - EKS cluster (via eksctl)
#   - Stuck CloudFormation stacks (force retry if DELETE_FAILED)
#
# What this KEEPS:
#   - S3 model files (s3://MODEL_BUCKET)
#   - IAM Role (ModelStorageRole)
#
# Usage:
#   cd llm-inference-framework
#   bash cluster/teardown.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }
log_step()  { echo ""; echo -e "${GREEN}━━━ $1 ━━━${NC}"; }

CF_STACK="eksctl-${CLUSTER_NAME}-cluster"

# ==============================================================================
# Validate
# ==============================================================================
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    log_error "AWS authentication failed."
    exit 1
fi

# ==============================================================================
# Confirm
# ==============================================================================
echo ""
echo "══════════════════════════════════════════════════"
echo "  TEARDOWN — ${CLUSTER_NAME}"
echo "  Region  : ${AWS_REGION}"
echo "  Kept    : s3://${MODEL_BUCKET}, IAM Role ${IAM_ROLE_NAME}"
echo "══════════════════════════════════════════════════"
echo ""
log_warn "This will DELETE the cluster and all AWS resources."
echo ""
read -r -p "Type cluster name '${CLUSTER_NAME}' to confirm: " CONFIRM
[[ "${CONFIRM}" == "${CLUSTER_NAME}" ]] || { log_info "Cancelled."; exit 0; }

# ==============================================================================
# Step 1 — Get VPC ID before deleting cluster
# ==============================================================================
log_step "Getting VPC ID"
VPC_ID=$(aws eks describe-cluster \
    --name "${CLUSTER_NAME}" \
    --region "${AWS_REGION}" \
    --query "cluster.resourcesVpcConfig.vpcId" \
    --output text 2>/dev/null || echo "")

if [[ -n "${VPC_ID}" && "${VPC_ID}" != "None" ]]; then
    log_info "VPC: ${VPC_ID}"
else
    log_warn "Cluster not found or VPC unavailable — checking CloudFormation stack only."
fi

# ==============================================================================
# Step 2 — Clean up Kubernetes resources first
# ==============================================================================
log_step "Cleaning up Kubernetes resources"

if kubectl cluster-info >/dev/null 2>&1; then

    # Delete vLLM deployment and service
    log_info "Removing vLLM deployment..."
    kubectl delete deployment vllm -n "${NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
    kubectl wait --for=delete pod -l app=vllm -n "${NAMESPACE}" --timeout=60s 2>/dev/null || true

    # Delete PVCs (releases EBS volumes)
    log_info "Removing PVCs..."
    kubectl delete pvc --all -n "${NAMESPACE}" --ignore-not-found=true 2>/dev/null || true

    # Delete LoadBalancer services (releases AWS NLB/ALB)
    log_info "Removing LoadBalancer services..."
    LB_SVCS=$(kubectl get svc -A --field-selector spec.type=LoadBalancer \
        -o jsonpath='{range .items[*]}{.metadata.namespace} {.metadata.name}{"\n"}{end}' 2>/dev/null || true)
    if [[ -n "${LB_SVCS}" ]]; then
        while read -r NS SVC; do
            [[ -n "${NS}" && -n "${SVC}" ]] && \
                kubectl delete svc "${SVC}" -n "${NS}" --ignore-not-found=true 2>/dev/null || true
        done <<< "${LB_SVCS}"
        log_info "Waiting 30s for NLB to be deregistered..."
        sleep 30
    else
        log_info "No LoadBalancer services found."
    fi

    # Delete Ingress
    log_info "Removing Ingress..."
    kubectl delete ingress --all -A --ignore-not-found=true 2>/dev/null || true

else
    log_warn "kubectl not connected — skipping Kubernetes cleanup."
fi

# ==============================================================================
# Step 3 — Clean up VPC dependencies
# ==============================================================================
if [[ -n "${VPC_ID}" && "${VPC_ID}" != "None" ]]; then

    log_step "Cleaning up VPC dependencies: ${VPC_ID}"

    # ── NAT Gateways ──────────────────────────────────────────────────────────
    log_info "Deleting NAT Gateways..."
    NAT_GWS=$(aws ec2 describe-nat-gateways \
        --filter "Name=vpc-id,Values=${VPC_ID}" \
        --region "${AWS_REGION}" \
        --query "NatGateways[?State!='deleted'].NatGatewayId" \
        --output text 2>/dev/null || true)

    if [[ -n "${NAT_GWS}" ]]; then
        for NGW in ${NAT_GWS}; do
            log_info "  Deleting NAT Gateway: ${NGW}"
            aws ec2 delete-nat-gateway --nat-gateway-id "${NGW}" --region "${AWS_REGION}" || true
        done
        log_info "Waiting 60s for NAT Gateways to delete..."
        sleep 60
    else
        log_info "  No NAT Gateways found."
    fi

    # ── Elastic IPs (released by NAT GW deletion) ─────────────────────────────
    log_info "Releasing orphaned Elastic IPs..."
    EIPS=$(aws ec2 describe-addresses \
        --region "${AWS_REGION}" \
        --query "Addresses[?AssociationId==null].AllocationId" \
        --output text 2>/dev/null || true)
    if [[ -n "${EIPS}" ]]; then
        for EIP in ${EIPS}; do
            log_info "  Releasing EIP: ${EIP}"
            aws ec2 release-address --allocation-id "${EIP}" --region "${AWS_REGION}" 2>/dev/null || true
        done
    else
        log_info "  No orphaned Elastic IPs found."
    fi

    # ── Detach and delete Internet Gateways ───────────────────────────────────
    log_info "Deleting Internet Gateways..."
    IGWS=$(aws ec2 describe-internet-gateways \
        --filters "Name=attachment.vpc-id,Values=${VPC_ID}" \
        --region "${AWS_REGION}" \
        --query "InternetGateways[*].InternetGatewayId" \
        --output text 2>/dev/null || true)

    if [[ -n "${IGWS}" ]]; then
        for IGW in ${IGWS}; do
            log_info "  Detaching IGW: ${IGW}"
            aws ec2 detach-internet-gateway \
                --internet-gateway-id "${IGW}" \
                --vpc-id "${VPC_ID}" \
                --region "${AWS_REGION}" 2>/dev/null || true
            log_info "  Deleting IGW: ${IGW}"
            aws ec2 delete-internet-gateway \
                --internet-gateway-id "${IGW}" \
                --region "${AWS_REGION}" 2>/dev/null || true
        done
    else
        log_info "  No Internet Gateways found."
    fi

    # ── Delete ENIs ───────────────────────────────────────────────────────────
    log_info "Deleting orphaned ENIs..."
    ENIS=$(aws ec2 describe-network-interfaces \
        --filters "Name=vpc-id,Values=${VPC_ID}" "Name=status,Values=available" \
        --region "${AWS_REGION}" \
        --query "NetworkInterfaces[*].NetworkInterfaceId" \
        --output text 2>/dev/null || true)

    if [[ -n "${ENIS}" ]]; then
        for ENI in ${ENIS}; do
            log_info "  Deleting ENI: ${ENI}"
            aws ec2 delete-network-interface \
                --network-interface-id "${ENI}" \
                --region "${AWS_REGION}" 2>/dev/null || true
        done
    else
        log_info "  No orphaned ENIs found."
    fi

    # ── Delete non-default Security Groups ────────────────────────────────────
    log_info "Deleting Security Groups..."
    SGS=$(aws ec2 describe-security-groups \
        --filters "Name=vpc-id,Values=${VPC_ID}" \
        --region "${AWS_REGION}" \
        --query "SecurityGroups[?GroupName!='default'].GroupId" \
        --output text 2>/dev/null || true)

    if [[ -n "${SGS}" ]]; then
        for SG in ${SGS}; do
            log_info "  Deleting SG: ${SG}"
            aws ec2 delete-security-group \
                --group-id "${SG}" \
                --region "${AWS_REGION}" 2>/dev/null || true
        done
    else
        log_info "  No custom Security Groups found."
    fi

fi

# ==============================================================================
# Step 4 — Delete EKS cluster via eksctl
# ==============================================================================
log_step "Deleting EKS cluster"

CLUSTER_EXISTS=$(aws eks describe-cluster \
    --name "${CLUSTER_NAME}" \
    --region "${AWS_REGION}" \
    --query "cluster.status" \
    --output text 2>/dev/null || echo "NOT_FOUND")

if [[ "${CLUSTER_EXISTS}" != "NOT_FOUND" ]]; then
    log_info "Running eksctl delete cluster..."
    eksctl delete cluster \
        --name "${CLUSTER_NAME}" \
        --region "${AWS_REGION}" \
        --wait || log_warn "eksctl timed out — CloudFormation deletion continues in background."
else
    log_warn "Cluster not found via EKS API — checking CloudFormation stack directly."
fi

# ==============================================================================
# Step 5 — Handle stuck CloudFormation stack
# ==============================================================================
log_step "Checking CloudFormation stack"

CF_STATUS=$(aws cloudformation describe-stacks \
    --stack-name "${CF_STACK}" \
    --region "${AWS_REGION}" \
    --query "Stacks[0].StackStatus" \
    --output text 2>/dev/null || echo "NOT_FOUND")

log_info "Stack status: ${CF_STATUS}"

if [[ "${CF_STATUS}" == "DELETE_FAILED" ]]; then
    log_warn "Stack is DELETE_FAILED — retrying deletion..."
    aws cloudformation delete-stack \
        --stack-name "${CF_STACK}" \
        --region "${AWS_REGION}"

    log_info "Waiting for stack deletion (up to 30 min)..."
    aws cloudformation wait stack-delete-complete \
        --stack-name "${CF_STACK}" \
        --region "${AWS_REGION}" && \
        log_info "Stack deleted." || \
        log_warn "Stack deletion still in progress. Check AWS Console → CloudFormation."

elif [[ "${CF_STATUS}" == "DELETE_IN_PROGRESS" ]]; then
    log_info "Stack deletion in progress — waiting..."
    aws cloudformation wait stack-delete-complete \
        --stack-name "${CF_STACK}" \
        --region "${AWS_REGION}" && \
        log_info "Stack deleted." || \
        log_warn "Stack deletion timed out. Check AWS Console → CloudFormation."

elif [[ "${CF_STATUS}" == "NOT_FOUND" ]]; then
    log_info "Stack already deleted."
fi

# ==============================================================================
# Step 6 — Check for orphaned EBS volumes
# ==============================================================================
log_step "Checking for orphaned EBS volumes"
ORPHANED=$(aws ec2 describe-volumes \
    --filters "Name=status,Values=available" \
              "Name=tag:kubernetes.io/cluster/${CLUSTER_NAME},Values=owned" \
    --region "${AWS_REGION}" \
    --query "Volumes[*].VolumeId" \
    --output text 2>/dev/null || true)

if [[ -n "${ORPHANED}" ]]; then
    log_warn "Orphaned EBS volumes found — deleting..."
    for VOL in ${ORPHANED}; do
        log_info "  Deleting volume: ${VOL}"
        aws ec2 delete-volume --volume-id "${VOL}" --region "${AWS_REGION}" || true
    done
else
    log_info "No orphaned EBS volumes."
fi

# ==============================================================================
# Done
# ==============================================================================
echo ""
echo "══════════════════════════════════════════════════"
echo "  TEARDOWN COMPLETE"
echo "══════════════════════════════════════════════════"
echo "  Deleted : EKS cluster, nodes, VPC, PVCs"
echo "  Kept    : s3://${MODEL_BUCKET}"
echo "  Kept    : IAM Role ${IAM_ROLE_NAME}"
echo "══════════════════════════════════════════════════"
