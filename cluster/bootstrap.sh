#!/bin/bash
# ==============================================================================
# cluster/bootstrap.sh
#
# Full cluster bootstrap — run once to provision everything.
#
# What this does:
#   1. Validates tools and AWS credentials
#   2. Creates the oai-infopt namespace
#   3. Creates S3 buckets (model + results) with encryption, versioning, blocking
#   4. Creates the EKS Auto Mode cluster (oai-infopt-eks)
#   5. Adds an S3 Gateway Endpoint to the cluster VPC (free, private S3 access)
#   6. Updates kubeconfig
#   7. Tags the EKS cluster
#   8. Applies gp3 StorageClass
#   9. Applies GPU NodePool (gpu-inf)
#  10. Sets up Pod Identity (3 least-privilege roles + ServiceAccounts)
#
# Cost guardrails created separately (CloudWatch alarms) — see docs/runbooks/.
#
# Usage:
#   cd llm-inference-framework
#   bash cluster/bootstrap.sh
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

# ==============================================================================
# Step 1 — Check required tools
# ==============================================================================
log_step "Checking required tools"
for tool in aws eksctl kubectl; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        log_error "Missing required tool: ${tool}"
        exit 1
    fi
    log_info "${tool} — OK ($(${tool} version --short 2>/dev/null || ${tool} --version 2>/dev/null | head -1 || echo 'version unknown'))"
done

# ==============================================================================
# Step 2 — Validate AWS credentials and resolve runtime values
# ==============================================================================
log_step "Validating AWS credentials"
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    log_error "AWS authentication failed."
    log_error "Local dev: configure IAM Roles Anywhere. CI: use OIDC role assumption."
    log_error "Never use long-lived access/secret keys — see shellkode-security.md."
    exit 1
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_ARN=$(aws sts get-caller-identity --query Arn --output text)

# Resolve bucket names at runtime — never hardcoded in config files
MODEL_BUCKET="${MODEL_BUCKET_PREFIX}-${AWS_ACCOUNT_ID}-${AWS_REGION}"
RESULTS_BUCKET="${RESULTS_BUCKET_PREFIX}-${AWS_ACCOUNT_ID}-${AWS_REGION}"

log_info "Account  : ${AWS_ACCOUNT_ID}"
log_info "Identity : ${AWS_ARN}"
log_info "Region   : ${AWS_REGION}"

# ==============================================================================
# Step 3 — Show configuration and confirm
# ==============================================================================
echo ""
echo "══════════════════════════════════════════════════════"
echo "  BOOTSTRAP CONFIGURATION"
echo "══════════════════════════════════════════════════════"
echo "  Cluster         : ${CLUSTER_NAME}"
echo "  Region          : ${AWS_REGION}"
echo "  EKS Version     : ${EKS_VERSION}"
echo "  Account         : ${AWS_ACCOUNT_ID}"
echo "  Model Bucket    : ${MODEL_BUCKET}"
echo "  Results Bucket  : ${RESULTS_BUCKET}"
echo "  Namespace       : ${BENCHMARK_NAMESPACE}"
echo "  Download Role   : ${DOWNLOAD_IAM_ROLE_NAME}"
echo "  Serving Role    : ${SERVING_IAM_ROLE_NAME}"
echo "  Benchmark Role  : ${BENCHMARK_IAM_ROLE_NAME}"
echo "  HF Token Secret : ${HF_TOKEN_SECRET_NAME}"
echo "  Tags            : Project=${TAG_PROJECT}, Env=${TAG_ENVIRONMENT}, Engagement=${TAG_ENGAGEMENT}"
echo "══════════════════════════════════════════════════════"
echo ""
read -r -p "Continue? [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

# ==============================================================================
# Step 4 — Create S3 buckets (encrypted, versioned, public-access blocked)
# ==============================================================================
log_step "Creating S3 buckets"

create_secure_bucket() {
    local BUCKET="$1"
    local PURPOSE="$2"

    if aws s3api head-bucket --bucket "${BUCKET}" 2>/dev/null; then
        log_warn "Bucket '${BUCKET}' already exists — verifying settings."
    else
        log_info "Creating bucket: ${BUCKET} (${PURPOSE})..."
        if [[ "${AWS_REGION}" == "us-east-1" ]]; then
            aws s3api create-bucket --bucket "${BUCKET}" --region "${AWS_REGION}"
        else
            aws s3api create-bucket \
                --bucket "${BUCKET}" \
                --region "${AWS_REGION}" \
                --create-bucket-configuration "LocationConstraint=${AWS_REGION}"
        fi
    fi

    # Block all public access
    aws s3api put-public-access-block \
        --bucket "${BUCKET}" \
        --public-access-block-configuration \
            "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

    # Enable server-side encryption (AES256 — or swap to aws:kms with your CMK)
    aws s3api put-bucket-encryption \
        --bucket "${BUCKET}" \
        --server-side-encryption-configuration '{
            "Rules": [{
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256"
                },
                "BucketKeyEnabled": true
            }]
        }'

    # Enable versioning (object recovery + audit trail)
    aws s3api put-bucket-versioning \
        --bucket "${BUCKET}" \
        --versioning-configuration Status=Enabled

    # Tag the bucket
    aws s3api put-bucket-tagging \
        --bucket "${BUCKET}" \
        --tagging "TagSet=[
            {Key=Project,Value=${TAG_PROJECT}},
            {Key=Engagement,Value=${TAG_ENGAGEMENT}},
            {Key=Environment,Value=${TAG_ENVIRONMENT}},
            {Key=Component,Value=${TAG_COMPONENT}},
            {Key=ManagedBy,Value=bootstrap},
            {Key=Purpose,Value=${PURPOSE}}
        ]"

    log_info "Bucket ready: s3://${BUCKET}"
}

create_secure_bucket "${MODEL_BUCKET}"   "model-weights-store"
create_secure_bucket "${RESULTS_BUCKET}" "benchmark-results-store"

# ==============================================================================
# Step 5 — Create EKS cluster
# ==============================================================================
log_step "Creating EKS cluster: ${CLUSTER_NAME}"

AZS=$(aws ec2 describe-availability-zones \
    --region "${AWS_REGION}" \
    --filters Name=state,Values=available \
    --query "AvailabilityZones[].ZoneName" \
    --output text | tr '\t' ',')

if eksctl get cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log_warn "Cluster '${CLUSTER_NAME}' already exists — skipping creation."
else
    eksctl create cluster \
        --name="${CLUSTER_NAME}" \
        --region="${AWS_REGION}" \
        --version="${EKS_VERSION}" \
        --enable-auto-mode \
        --zones="${AZS}" \
        --tags="Project=${TAG_PROJECT},Engagement=${TAG_ENGAGEMENT},Environment=${TAG_ENVIRONMENT},Component=${TAG_COMPONENT},ManagedBy=${TAG_MANAGED_BY},Owner=${TAG_OWNER}"
    log_info "Cluster created."
fi

# ==============================================================================
# Step 6 — Update kubeconfig
# ==============================================================================
log_step "Updating kubeconfig"
aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region "${AWS_REGION}"
log_info "kubeconfig updated."

# ==============================================================================
# Step 7 — Tag EKS cluster resource
# ==============================================================================
log_step "Tagging EKS cluster"
CLUSTER_ARN=$(aws eks describe-cluster \
    --name "${CLUSTER_NAME}" \
    --region "${AWS_REGION}" \
    --query "cluster.arn" --output text)

aws eks tag-resource \
    --resource-arn "${CLUSTER_ARN}" \
    --tags \
        "Project=${TAG_PROJECT}" \
        "Engagement=${TAG_ENGAGEMENT}" \
        "Environment=${TAG_ENVIRONMENT}" \
        "Component=${TAG_COMPONENT}" \
        "ManagedBy=${TAG_MANAGED_BY}" \
        "Owner=${TAG_OWNER}" \
    --region "${AWS_REGION}" || log_warn "Cluster tag apply failed — verify manually."

# ==============================================================================
# Step 8 — Add S3 Gateway Endpoint (free, private — no NAT required for S3)
# ==============================================================================
log_step "Adding S3 Gateway Endpoint to cluster VPC"

VPC_ID=$(aws eks describe-cluster \
    --name "${CLUSTER_NAME}" \
    --region "${AWS_REGION}" \
    --query "cluster.resourcesVpcConfig.vpcId" \
    --output text)

log_info "Cluster VPC: ${VPC_ID}"

# Get all route table IDs in the VPC
ROUTE_TABLE_IDS=$(aws ec2 describe-route-tables \
    --filters "Name=vpc-id,Values=${VPC_ID}" \
    --region "${AWS_REGION}" \
    --query "RouteTables[*].RouteTableId" \
    --output text | tr '\t' ' ')

# Check if S3 gateway endpoint already exists
EXISTING_EP=$(aws ec2 describe-vpc-endpoints \
    --filters \
        "Name=vpc-id,Values=${VPC_ID}" \
        "Name=service-name,Values=com.amazonaws.${AWS_REGION}.s3" \
        "Name=vpc-endpoint-type,Values=Gateway" \
    --region "${AWS_REGION}" \
    --query "VpcEndpoints[?State=='available'].VpcEndpointId" \
    --output text 2>/dev/null || echo "")

if [[ -n "${EXISTING_EP}" ]]; then
    log_warn "S3 Gateway Endpoint already exists: ${EXISTING_EP} — skipping."
else
    log_info "Creating S3 Gateway Endpoint (free — enables private S3 access from GPU nodes)..."
    EP_ID=$(aws ec2 create-vpc-endpoint \
        --vpc-id "${VPC_ID}" \
        --service-name "com.amazonaws.${AWS_REGION}.s3" \
        --vpc-endpoint-type Gateway \
        --route-table-ids ${ROUTE_TABLE_IDS} \
        --region "${AWS_REGION}" \
        --tag-specifications "ResourceType=vpc-endpoint,Tags=[
            {Key=Project,Value=${TAG_PROJECT}},
            {Key=Environment,Value=${TAG_ENVIRONMENT}},
            {Key=ManagedBy,Value=bootstrap},
            {Key=Purpose,Value=s3-private-access}
        ]" \
        --query "VpcEndpoint.VpcEndpointId" --output text)
    log_info "S3 Gateway Endpoint created: ${EP_ID}"
fi

# ==============================================================================
# Step 9 — Apply StorageClass
# ==============================================================================
log_step "Applying gp3 StorageClass"
kubectl apply -f "${SCRIPT_DIR}/storage-class.yaml"
log_info "StorageClass applied."

# ==============================================================================
# Step 10 — Apply GPU NodePool
# ==============================================================================
log_step "Applying GPU NodePool (gpu-inf)"
kubectl apply -f "${SCRIPT_DIR}/gpu-nodepool.yaml"
log_info "GPU NodePool applied."

# ==============================================================================
# Step 11 — Pod Identity setup (3 least-privilege roles)
# ==============================================================================
log_step "Setting up Pod Identity"
bash "${SCRIPT_DIR}/setup-pod-identity.sh"

# ==============================================================================
# Done
# ==============================================================================
echo ""
echo "══════════════════════════════════════════════════════"
echo "  BOOTSTRAP COMPLETE"
echo "══════════════════════════════════════════════════════"
kubectl get nodes
echo ""
kubectl get nodepool 2>/dev/null || true
echo ""
kubectl get storageclass
echo ""
echo "  Model bucket  : s3://${MODEL_BUCKET}"
echo "  Results bucket: s3://${RESULTS_BUCKET}"
echo "  Namespace     : ${BENCHMARK_NAMESPACE}"
echo ""
echo "  Next steps:"
echo "    1. Store HF token in Secrets Manager (if not done):"
echo "       aws secretsmanager create-secret \\"
echo "         --name '${HF_TOKEN_SECRET_NAME}' \\"
echo "         --secret-string '{\"token\":\"hf_YOUR_TOKEN\"}' \\"
echo "         --region '${AWS_REGION}'"
echo ""
echo "    2. Download a model:"
echo "       bash model-download/qwen-2.5-0.5b/download.sh"
echo "       bash model-download/gpt-oss-20b/download.sh"
echo "══════════════════════════════════════════════════════"
