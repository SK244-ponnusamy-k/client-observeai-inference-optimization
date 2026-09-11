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
log_step()  { echo ""; echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; \
              echo -e "${GREEN}  $1${NC}"; \
              echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# CLI Argument Parsing
usage() {
    echo "Usage: bash cluster/bootstrap.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --compute <gpu|g7|neuron|all>   Comma-separated list of compute pools to enable (e.g. --compute gpu,g7)"
    echo "  --enable-gpu / --disable-gpu     Enable/disable standard GPU pool (G5/G6/G6e)"
    echo "  --enable-g7 / --disable-g7       Enable/disable G7 Blackwell pool (G7/G7e)"
    echo "  --enable-neuron / --disable-neuron Enable/disable AWS Neuron pool (Trn1/Inf2)"
    echo "  --g7-ami <ami-id>               Specify custom AMI ID for G7 instances (NVIDIA Driver 595+)"
    echo "  -h, --help                      Show this help message"
    echo ""
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --compute)
            IFS=',' read -ra COMPUTE_ARR <<< "$2"
            ENABLE_GPU_STANDARD="false"
            ENABLE_G7_SUPPORT="false"
            ENABLE_NEURON_SUPPORT="false"
            for item in "${COMPUTE_ARR[@]}"; do
                case "$item" in
                    gpu|standard) ENABLE_GPU_STANDARD="true" ;;
                    g7|blackwell) ENABLE_G7_SUPPORT="true" ;;
                    neuron|trn1)  ENABLE_NEURON_SUPPORT="true" ;;
                    all)
                        ENABLE_GPU_STANDARD="true"
                        ENABLE_G7_SUPPORT="true"
                        ENABLE_NEURON_SUPPORT="true"
                        ;;
                    *) echo "Unknown compute type: $item"; exit 1 ;;
                esac
            done
            shift 2
            ;;
        --enable-gpu) ENABLE_GPU_STANDARD="true"; shift ;;
        --disable-gpu) ENABLE_GPU_STANDARD="false"; shift ;;
        --enable-g7) ENABLE_G7_SUPPORT="true"; shift ;;
        --disable-g7) ENABLE_G7_SUPPORT="false"; shift ;;
        --enable-neuron) ENABLE_NEURON_SUPPORT="true"; shift ;;
        --disable-neuron) ENABLE_NEURON_SUPPORT="false"; shift ;;
        --g7-ami) G7_CUSTOM_AMI_ID="$2"; ENABLE_G7_SUPPORT="true"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

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

# Bucket names come directly from config.env (already fully qualified)
# MODEL_BUCKET and RESULTS_BUCKET are exported there — no prefix construction needed.

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
echo "  Compute Enabled : Standard GPU (G5/G6/G6e)=${ENABLE_GPU_STANDARD}, G7 Blackwell=${ENABLE_G7_SUPPORT}, Neuron=${ENABLE_NEURON_SUPPORT}"
if [[ "${ENABLE_G7_SUPPORT}" == "true" ]]; then
    echo "  G7 Custom AMI   : ${G7_CUSTOM_AMI_ID:-Auto-discover via SSM (${G7_SSM_AMI_PARAM})}"
fi
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
if kubectl get storageclass gp3 >/dev/null 2>&1; then
    log_warn "StorageClass 'gp3' already exists — skipping (parameters are immutable)."
else
    kubectl apply -f "${SCRIPT_DIR}/storage-class.yaml"
    log_info "StorageClass applied."
fi

# ==============================================================================
# Step 10 — Apply Compute NodePools (Standard GPU, G7 Blackwell, Neuron)
# ==============================================================================
log_step "Applying Compute NodePools"

# Standard GPUs (G5, G6, G6e)
if [[ "${ENABLE_GPU_STANDARD}" == "true" ]]; then
    log_info "Applying Standard GPU NodePool (gpu-inf for G5/G6/G6e)..."
    kubectl apply -f "${SCRIPT_DIR}/gpu-nodepool.yaml"
    log_info "Standard GPU NodePool applied."
fi

# G7 Blackwell GPUs (G7, G7e)
if [[ "${ENABLE_G7_SUPPORT}" == "true" ]]; then
    log_info "Configuring G7 Blackwell GPU compute (G7/G7e)..."
    
    # Resolve G7 custom AMI ID
    G7_AMI_ID="${G7_CUSTOM_AMI_ID:-}"
    if [[ -z "${G7_AMI_ID}" ]]; then
        log_info "Querying SSM Parameter Store (${G7_SSM_AMI_PARAM}) for G7 custom AMI..."
        G7_AMI_ID=$(aws ssm get-parameter --name "${G7_SSM_AMI_PARAM}" --region "${AWS_REGION}" --query Parameter.Value --output text 2>/dev/null || echo "")
    fi

    if [[ -n "${G7_AMI_ID}" ]]; then
        log_info "Using G7 Custom AMI ID: ${G7_AMI_ID}"
        log_info "Applying G7 NodeClass (custom-g7-nodeclass)..."
        
        # Substitute environment variables into g7-nodeclass.yaml
        sed -e "s/\${G7_AMI_ID}/${G7_AMI_ID}/g" \
            -e "s/\${CLUSTER_NAME}/${CLUSTER_NAME}/g" \
            "${SCRIPT_DIR}/g7-nodeclass.yaml" | kubectl apply -f -

        log_info "Applying G7 NodePool (gpu-g7-inf)..."
        kubectl apply -f "${SCRIPT_DIR}/gpu-g7-nodepool.yaml"
        log_info "G7 Blackwell GPU compute configured."
    else
        log_warn "G7 support enabled, but no G7 Custom AMI ID found!"
        log_warn "Build AMI using: bash cluster/ami/build-g7-ami.sh"
        log_warn "Or pass AMI explicitly: bash cluster/bootstrap.sh --g7-ami <ami-id>"
        log_warn "Skipping G7 NodePool application for now."
    fi
fi

# AWS Neuron (Trn1, Trn1n, Inf2)
if [[ "${ENABLE_NEURON_SUPPORT}" == "true" ]]; then
    log_info "Applying Neuron NodePool (neuron-inf for Trn1/Trn2/Inf2)..."
    if [[ -f "${SCRIPT_DIR}/neuron-nodepool.yaml" ]]; then
        kubectl apply -f "${SCRIPT_DIR}/neuron-nodepool.yaml"
        log_info "Neuron NodePool (Karpenter) applied."
    fi

    # Set up trn2 managed node group via CloudFormation + eksctl.
    # This is needed because EKS Auto Mode's Fleet API does not support
    # trn2 provisioning — managed node group bypasses Auto Mode.
    log_info "Setting up trn2 managed node group (CloudFormation + eksctl)..."
    bash "${SCRIPT_DIR}/setup-trn2-nodegroup.sh" || {
        log_warn "trn2 node group setup failed — check capacity in ${AWS_REGION}."
        log_warn "Retry manually: bash cluster/setup-trn2-nodegroup.sh"
    }
fi


# ==============================================================================
# Step 11 — Pod Identity setup (3 least-privilege roles)
# ==============================================================================
log_step "Setting up Pod Identity"
bash "${SCRIPT_DIR}/setup-pod-identity.sh"

# ==============================================================================
# Step 12 — Monitoring stack (AMP + Grafana + DCGM + metrics collector)
# ==============================================================================
log_step "Step 12 — Monitoring stack"

if command -v helm >/dev/null 2>&1; then
    log_info "Installing monitoring stack..."
    bash "${FRAMEWORK_ROOT}/monitoring/setup-monitoring.sh"
else
    log_warn "helm not found — skipping monitoring setup."
    log_warn "Install helm and run:  bash monitoring/setup-monitoring.sh"
fi

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
echo "  Monitoring:"
echo "    kubectl get pods -n monitoring"
echo "    kubectl get ingress kube-prometheus-stack-grafana -n monitoring"
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
echo ""
echo "    3. Deploy a model:"
echo "       bash vllm/models/qwen-2.5-0.5b/deploy.sh"
echo "       bash vllm/models/gpt-oss-20b/deploy.sh"
echo "══════════════════════════════════════════════════════"
