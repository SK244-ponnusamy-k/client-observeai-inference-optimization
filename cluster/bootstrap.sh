#!/bin/bash
# ==============================================================================
# cluster/bootstrap.sh
#
# Full cluster bootstrap — run once to get everything ready.
#
# What this does:
#   1. Reads config/config.env
#   2. Validates AWS credentials + tools
#   3. Creates EKS Auto Mode cluster
#   4. Updates kubeconfig
#   5. Tags the cluster
#   6. Applies gp3 StorageClass
#   7. Applies GPU NodePool
#   8. Sets up Pod Identity (IAM Role + ServiceAccount + association)
#
# Usage:
#   cd llm-inference-framework
#   bash cluster/bootstrap.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ------------------------------------------------------------------------------
# Load config
# ------------------------------------------------------------------------------
source "${FRAMEWORK_ROOT}/config/config.env"

# ------------------------------------------------------------------------------
# Colors + logging
# ------------------------------------------------------------------------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }
log_step()  { echo ""; echo -e "${GREEN}━━━ $1 ━━━${NC}"; }

# ==============================================================================
# Step 1 — Check tools
# ==============================================================================
log_step "Checking required tools"
for tool in aws eksctl kubectl; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        log_error "Missing required tool: ${tool}"
        exit 1
    fi
    log_info "${tool} — OK"
done

# ==============================================================================
# Step 2 — Validate AWS credentials
# ==============================================================================
log_step "Validating AWS credentials"
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    log_error "AWS authentication failed. Run: aws configure"
    exit 1
fi
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_ARN=$(aws sts get-caller-identity --query Arn --output text)
log_info "Account : ${AWS_ACCOUNT_ID}"
log_info "Identity: ${AWS_ARN}"

# ==============================================================================
# Step 3 — Show configuration
# ==============================================================================
echo ""
echo "══════════════════════════════════════════════════"
echo "  BOOTSTRAP CONFIGURATION"
echo "══════════════════════════════════════════════════"
echo "  Cluster      : ${CLUSTER_NAME}"
echo "  Region       : ${AWS_REGION}"
echo "  EKS Version  : ${EKS_VERSION}"
echo "  Account      : ${AWS_ACCOUNT_ID}"
echo "  S3 Bucket    : ${MODEL_BUCKET}"
echo "  IAM Role     : ${IAM_ROLE_NAME}"
echo "  Tags         : Project=${TAG_PROJECT}, Owner=${TAG_OWNER}, Env=${TAG_ENVIRONMENT}"
echo "══════════════════════════════════════════════════"
echo ""
read -r -p "Continue? [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

# ==============================================================================
# Step 4 — Create EKS cluster
# ==============================================================================
log_step "Creating EKS cluster"

if eksctl get cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log_warn "Cluster '${CLUSTER_NAME}' already exists — skipping creation."
else
    AZS=$(aws ec2 describe-availability-zones \
        --region "${AWS_REGION}" \
        --filters Name=state,Values=available \
        --query "AvailabilityZones[].ZoneName" \
        --output text | tr '\t' ',')

    eksctl create cluster \
        --name="${CLUSTER_NAME}" \
        --region="${AWS_REGION}" \
        --version="${EKS_VERSION}" \
        --enable-auto-mode \
        --zones="${AZS}" \
        --tags="Project=${TAG_PROJECT},Component=${TAG_COMPONENT},Environment=${TAG_ENVIRONMENT},Owner=${TAG_OWNER},ManagedBy=${TAG_MANAGED_BY},Workload=${TAG_WORKLOAD},Guide=${TAG_GUIDE}"

    log_info "Cluster created."
fi

# ==============================================================================
# Step 5 — Update kubeconfig
# ==============================================================================
log_step "Updating kubeconfig"
aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region "${AWS_REGION}"
log_info "kubeconfig updated."

# ==============================================================================
# Step 6 — Tag cluster (fixes Windows tag parsing bug)
# ==============================================================================
log_step "Tagging EKS cluster"
CLUSTER_ARN=$(aws eks describe-cluster \
    --name "${CLUSTER_NAME}" \
    --region "${AWS_REGION}" \
    --query "cluster.arn" --output text)

aws eks tag-resource \
    --resource-arn "${CLUSTER_ARN}" \
    --tags "Project=${TAG_PROJECT},Component=${TAG_COMPONENT},Environment=${TAG_ENVIRONMENT},Owner=${TAG_OWNER},ManagedBy=${TAG_MANAGED_BY},Workload=${TAG_WORKLOAD},Guide=${TAG_GUIDE}" \
    --region "${AWS_REGION}" || log_warn "Tag apply failed — apply manually if needed."

log_info "Cluster tagged."

# ==============================================================================
# Step 7 — Apply StorageClass
# ==============================================================================
log_step "Applying gp3 StorageClass"
kubectl apply -f "${SCRIPT_DIR}/storage-class.yaml"
log_info "StorageClass applied."

# ==============================================================================
# Step 8 — Apply GPU NodePool
# ==============================================================================
log_step "Applying GPU NodePool"
kubectl apply -f "${SCRIPT_DIR}/gpu-nodepool.yaml"
log_info "GPU NodePool applied."

# ==============================================================================
# Step 9 — Pod Identity setup
# ==============================================================================
log_step "Setting up Pod Identity"
bash "${FRAMEWORK_ROOT}/cluster/setup-pod-identity.sh"

# ==============================================================================
# Step 10 — Final status
# ==============================================================================
echo ""
echo "══════════════════════════════════════════════════"
echo "  BOOTSTRAP COMPLETE"
echo "══════════════════════════════════════════════════"
kubectl get nodes
echo ""
kubectl get nodepool
echo ""
kubectl get storageclass
echo ""
echo "  Next: download a model"
echo "  bash model-download/qwen-2.5-0.5b/download.sh"
echo "  bash model-download/gpt-oss-20b/download.sh"
echo "══════════════════════════════════════════════════"
