#!/bin/bash
# ==============================================================================
# cluster/setup-pod-identity.sh
#
# Creates IAM Role, attaches S3 policy, enables Pod Identity addon,
# creates ServiceAccount and Pod Identity association.
#
# Called automatically by bootstrap.sh — can also be run standalone
# after cluster recreation.
#
# Usage:
#   cd llm-inference-framework
#   bash cluster/setup-pod-identity.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

# Validate
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    log_error "AWS authentication failed."
    exit 1
fi
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Validate S3 bucket
if ! aws s3api head-bucket --bucket "${MODEL_BUCKET}" 2>/dev/null; then
    log_error "S3 bucket '${MODEL_BUCKET}' not found. Create it first:"
    log_error "  aws s3 mb s3://${MODEL_BUCKET} --region ${AWS_REGION}"
    exit 1
fi

# Validate cluster
if ! aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log_error "Cluster '${CLUSTER_NAME}' not found."
    exit 1
fi

log_info "Account: ${AWS_ACCOUNT_ID} | Cluster: ${CLUSTER_NAME} | Bucket: ${MODEL_BUCKET}"

# ------------------------------------------------------------------------------
# Temp dir (Windows-compatible path handling)
# ------------------------------------------------------------------------------
TMPDIR_PATH="${FRAMEWORK_ROOT}/.tmp"
mkdir -p "${TMPDIR_PATH}"
WIN_TMPDIR=$(cygpath -w "${TMPDIR_PATH}" 2>/dev/null || echo "${TMPDIR_PATH}")
TRUST_FILE_UNIX="${TMPDIR_PATH}/trust.json"
S3_FILE_UNIX="${TMPDIR_PATH}/s3-policy.json"
TRUST_FILE_WIN="${WIN_TMPDIR}\\trust.json"
S3_FILE_WIN="${WIN_TMPDIR}\\s3-policy.json"

# ------------------------------------------------------------------------------
# IAM Trust Policy
# ------------------------------------------------------------------------------
cat > "${TRUST_FILE_UNIX}" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "pods.eks.amazonaws.com" },
    "Action": ["sts:AssumeRole", "sts:TagSession"]
  }]
}
EOF

# ------------------------------------------------------------------------------
# Create IAM Role
# ------------------------------------------------------------------------------
if aws iam get-role --role-name "${IAM_ROLE_NAME}" >/dev/null 2>&1; then
    log_warn "IAM Role '${IAM_ROLE_NAME}' exists — skipping creation."
else
    log_info "Creating IAM Role: ${IAM_ROLE_NAME}..."
    aws iam create-role \
        --role-name "${IAM_ROLE_NAME}" \
        --assume-role-policy-document "file://${TRUST_FILE_WIN}" \
        --description "EKS Pod Identity — S3 model storage access" \
        --tags \
            Key=Project,Value="${TAG_PROJECT}" \
            Key=Component,Value="${TAG_COMPONENT}" \
            Key=Environment,Value="${TAG_ENVIRONMENT}" \
            Key=Owner,Value="${TAG_OWNER}" \
            Key=ManagedBy,Value=eks-pod-identity \
            Key=Workload,Value="${TAG_WORKLOAD}" \
            Key=Guide,Value="${TAG_GUIDE}"
    log_info "IAM Role created."
fi

# Always ensure tags are applied
aws iam tag-role \
    --role-name "${IAM_ROLE_NAME}" \
    --tags \
        Key=Project,Value="${TAG_PROJECT}" \
        Key=Component,Value="${TAG_COMPONENT}" \
        Key=Environment,Value="${TAG_ENVIRONMENT}" \
        Key=Owner,Value="${TAG_OWNER}" \
        Key=ManagedBy,Value=eks-pod-identity \
        Key=Workload,Value="${TAG_WORKLOAD}" \
        Key=Guide,Value="${TAG_GUIDE}"

# ------------------------------------------------------------------------------
# Attach S3 policy
# ------------------------------------------------------------------------------
cat > "${S3_FILE_UNIX}" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "ModelBucketAccess",
    "Effect": "Allow",
    "Action": ["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket","s3:GetBucketLocation"],
    "Resource": [
      "arn:aws:s3:::${MODEL_BUCKET}",
      "arn:aws:s3:::${MODEL_BUCKET}/*"
    ]
  }]
}
EOF

aws iam put-role-policy \
    --role-name "${IAM_ROLE_NAME}" \
    --policy-name "ModelBucketPolicy" \
    --policy-document "file://${S3_FILE_WIN}"
log_info "S3 policy attached → s3://${MODEL_BUCKET}"

# ------------------------------------------------------------------------------
# Enable Pod Identity addon
# ------------------------------------------------------------------------------
ADDON_STATUS=$(aws eks describe-addon \
    --cluster-name "${CLUSTER_NAME}" \
    --addon-name eks-pod-identity-agent \
    --region "${AWS_REGION}" \
    --query "addon.status" --output text 2>/dev/null || echo "NOT_FOUND")

if [[ "${ADDON_STATUS}" == "ACTIVE" ]]; then
    log_warn "Pod Identity addon already ACTIVE."
else
    log_info "Enabling Pod Identity addon..."
    aws eks create-addon \
        --cluster-name "${CLUSTER_NAME}" \
        --addon-name eks-pod-identity-agent \
        --region "${AWS_REGION}"
    aws eks wait addon-active \
        --cluster-name "${CLUSTER_NAME}" \
        --addon-name eks-pod-identity-agent \
        --region "${AWS_REGION}"
    log_info "Pod Identity addon ACTIVE."
fi

# ------------------------------------------------------------------------------
# Apply ServiceAccount
# ------------------------------------------------------------------------------
log_info "Applying ServiceAccount..."
kubectl apply -f "${SCRIPT_DIR}/model-storage-sa.yaml"
log_info "ServiceAccount '${SERVICE_ACCOUNT}' ready."

# ------------------------------------------------------------------------------
# Create Pod Identity association
# ------------------------------------------------------------------------------
EXISTING=$(aws eks list-pod-identity-associations \
    --cluster-name "${CLUSTER_NAME}" \
    --namespace "${NAMESPACE}" \
    --service-account "${SERVICE_ACCOUNT}" \
    --region "${AWS_REGION}" \
    --query "associations[0].associationId" --output text 2>/dev/null || echo "None")

if [[ "${EXISTING}" != "None" && "${EXISTING}" != "null" && -n "${EXISTING}" ]]; then
    log_warn "Pod Identity association already exists (ID: ${EXISTING})."
else
    log_info "Creating Pod Identity association..."
    aws eks create-pod-identity-association \
        --cluster-name "${CLUSTER_NAME}" \
        --namespace "${NAMESPACE}" \
        --service-account "${SERVICE_ACCOUNT}" \
        --role-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:role/${IAM_ROLE_NAME}" \
        --region "${AWS_REGION}"
    log_info "Pod Identity association created."
fi

rm -rf "${TMPDIR_PATH}"
log_info "Pod Identity setup complete."
echo ""
echo "  IAM Role : arn:aws:iam::${AWS_ACCOUNT_ID}:role/${IAM_ROLE_NAME}"
echo "  S3 access: s3://${MODEL_BUCKET}"
echo "  Linked to: ${NAMESPACE}/${SERVICE_ACCOUNT}"
