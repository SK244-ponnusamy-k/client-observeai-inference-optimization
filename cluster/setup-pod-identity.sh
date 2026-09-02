#!/bin/bash
# ==============================================================================
# cluster/setup-pod-identity.sh
#
# Purpose : Create least-privilege IAM roles for each workload, store HF token
#           in Secrets Manager, enable EKS Pod Identity, and bind each role to
#           its ServiceAccount.
#
# Three roles (principle of least privilege):
#   oai-infopt-download-role   — S3 GetObject/PutObject on model bucket prefix
#                                 + Secrets Manager read for HF token
#   oai-infopt-serving-role    — S3 GetObject (read-only) on model bucket prefix
#   oai-infopt-benchmark-role  — S3 PutObject/GetObject on results bucket prefix
#                                 + CloudWatch PutMetricData for cost/perf metrics
#
# Called automatically by bootstrap.sh — can also be run standalone.
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
log_step()  { echo ""; echo -e "${GREEN}━━━ $1 ━━━${NC}"; }

# ==============================================================================
# Validate
# ==============================================================================
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    log_error "AWS authentication failed. Configure credentials first."
    exit 1
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Bucket names come directly from config.env (your existing buckets)
# MODEL_BUCKET and RESULTS_BUCKET are already exported by config.env

log_info "Account       : ${AWS_ACCOUNT_ID}"
log_info "Region        : ${AWS_REGION}"
log_info "Cluster       : ${CLUSTER_NAME}"
log_info "Model bucket  : ${MODEL_BUCKET}"
log_info "Results bucket: ${RESULTS_BUCKET}"

# ==============================================================================
# Validate cluster and buckets exist
# ==============================================================================
if ! aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log_error "Cluster '${CLUSTER_NAME}' not found. Run bootstrap.sh first."
    exit 1
fi

for BUCKET in "${MODEL_BUCKET}" "${RESULTS_BUCKET}"; do
    if aws s3api head-bucket --bucket "${BUCKET}" 2>/dev/null; then
        log_info "Bucket exists: s3://${BUCKET}"
    else
        if [[ "${BUCKET}" == "${RESULTS_BUCKET}" ]]; then
            log_info "Creating results bucket: s3://${RESULTS_BUCKET} ..."
            if [[ "${AWS_REGION}" == "us-east-1" ]]; then
                aws s3api create-bucket --bucket "${RESULTS_BUCKET}" --region "${AWS_REGION}"
            else
                aws s3api create-bucket \
                    --bucket "${RESULTS_BUCKET}" \
                    --region "${AWS_REGION}" \
                    --create-bucket-configuration "LocationConstraint=${AWS_REGION}"
            fi
            aws s3api put-public-access-block \
                --bucket "${RESULTS_BUCKET}" \
                --public-access-block-configuration \
                    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
            aws s3api put-bucket-encryption \
                --bucket "${RESULTS_BUCKET}" \
                --server-side-encryption-configuration \
                    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":true}]}'
            log_info "Results bucket created: s3://${RESULTS_BUCKET}"
        else
            log_warn "Bucket '${BUCKET}' not found — create it before running download jobs."
            log_warn "  aws s3 mb s3://${BUCKET} --region ${AWS_REGION}"
        fi
    fi
done

# ==============================================================================
# Temp dir for policy documents
# ==============================================================================
# Use a local subdir — avoids all /tmp and file:// path issues on Windows/MINGW64.
TMPDIR_PATH="${FRAMEWORK_ROOT}/.tmp-pod-identity-$$"
mkdir -p "${TMPDIR_PATH}"
trap 'rm -rf "${TMPDIR_PATH}"' EXIT

# Helper: read a JSON file and echo its contents.
# Used to pass policy documents inline to AWS CLI (--policy-document "$(read_json ...)")
# This avoids ALL file:// URI path issues on Windows/MINGW64/Git Bash where the
# AWS CLI (a Windows binary) cannot resolve Unix-style or mixed paths.
read_json() {
    cat "$1"
}

# ==============================================================================
# Shared Pod Identity trust policy
# ==============================================================================
cat > "${TMPDIR_PATH}/trust.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "pods.eks.amazonaws.com" },
    "Action": ["sts:AssumeRole", "sts:TagSession"]
  }]
}
EOF

# Standard tag set applied to every IAM resource
COMMON_TAGS=(
    Key=Project,Value="${TAG_PROJECT}"
    Key=Engagement,Value="${TAG_ENGAGEMENT}"
    Key=Environment,Value="${TAG_ENVIRONMENT}"
    Key=Component,Value="${TAG_COMPONENT}"
    Key=ManagedBy,Value="setup-pod-identity"
)

# ==============================================================================
# Helper: create-or-tag IAM role
# ==============================================================================
create_or_update_role() {
    local ROLE_NAME="$1"
    local DESCRIPTION="$2"
    if aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
        log_warn "IAM Role '${ROLE_NAME}' already exists — updating trust policy."
        aws iam update-assume-role-policy \
            --role-name "${ROLE_NAME}" \
            --policy-document "$(read_json "${TMPDIR_PATH}/trust.json")"
    else
        log_info "Creating IAM Role: ${ROLE_NAME}..."
        aws iam create-role \
            --role-name "${ROLE_NAME}" \
            --assume-role-policy-document "$(read_json "${TMPDIR_PATH}/trust.json")" \
            --description "${DESCRIPTION}" \
            --tags "${COMMON_TAGS[@]}"
        log_info "Role created: ${ROLE_NAME}"
    fi
    # Ensure tags are always current
    aws iam tag-role --role-name "${ROLE_NAME}" --tags "${COMMON_TAGS[@]}"
}

# ==============================================================================
# Step 1 — HF Token: read from config/.env → push to Secrets Manager
# ==============================================================================
log_step "Secrets Manager — HF Token"

ENV_FILE="${FRAMEWORK_ROOT}/config/.env"

# Load HF_TOKEN from config/.env if it exists and token not already in env
if [[ -f "${ENV_FILE}" ]]; then
    # Source only the HF_TOKEN line — avoid overwriting other env vars
    HF_TOKEN_FROM_FILE=$(grep -E '^HF_TOKEN=' "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"' | tr -d "'")
    if [[ -n "${HF_TOKEN_FROM_FILE}" && "${HF_TOKEN_FROM_FILE}" != "hf_REPLACE_WITH_YOUR_TOKEN" ]]; then
        export HF_TOKEN="${HF_TOKEN_FROM_FILE}"
        log_info "HF token loaded from config/.env"
    fi
fi

SECRET_EXISTS=$(aws secretsmanager describe-secret \
    --secret-id "${HF_TOKEN_SECRET_NAME}" \
    --region "${AWS_REGION}" \
    --query "ARN" --output text 2>/dev/null || echo "NOT_FOUND")

if [[ "${SECRET_EXISTS}" == "NOT_FOUND" ]]; then
    # Secret does not exist — create it now from HF_TOKEN
    if [[ -z "${HF_TOKEN:-}" || "${HF_TOKEN}" == "hf_REPLACE_WITH_YOUR_TOKEN" ]]; then
        log_warn "HF token not set. To enable gated model download (openai/gpt-oss-20b):"
        log_warn "  1. Get your token: https://huggingface.co/settings/tokens"
        log_warn "  2. Set it in config/.env:  HF_TOKEN=\"hf_your_token_here\""
        log_warn "  3. Re-run this script."
        log_warn "Continuing without HF token — Qwen (public model) is unaffected."
    else
        log_info "Creating secret '${HF_TOKEN_SECRET_NAME}' in Secrets Manager..."
        aws secretsmanager create-secret \
            --name "${HF_TOKEN_SECRET_NAME}" \
            --description "HuggingFace API token for gated model download (oai-infopt)" \
            --secret-string "{\"token\":\"${HF_TOKEN}\"}" \
            --region "${AWS_REGION}" \
            --tags \
                Key=Project,Value="${TAG_PROJECT}" \
                Key=Environment,Value="${TAG_ENVIRONMENT}" \
                Key=ManagedBy,Value="setup-pod-identity" \
            >/dev/null
        log_info "Secret created: ${HF_TOKEN_SECRET_NAME}"
        # Clear from env immediately — no longer needed in memory
        unset HF_TOKEN
        unset HF_TOKEN_FROM_FILE
    fi
else
    HF_SECRET_ARN="${SECRET_EXISTS}"
    log_info "HF token secret already exists: ${HF_SECRET_ARN}"
    # Clear from env if loaded — Secrets Manager is the source of truth now
    unset HF_TOKEN 2>/dev/null || true
    unset HF_TOKEN_FROM_FILE 2>/dev/null || true
fi

# ==============================================================================
# Step 2 — Download Role (S3 write + Secrets Manager read for HF token)
# ==============================================================================
log_step "Download IAM Role: ${DOWNLOAD_IAM_ROLE_NAME}"

create_or_update_role "${DOWNLOAD_IAM_ROLE_NAME}" \
    "oai-infopt: model download - S3 write + Secrets Manager HF token read"

# Resolve secret ARN (may not exist yet if skipped above)
HF_SECRET_ARN=$(aws secretsmanager describe-secret \
    --secret-id "${HF_TOKEN_SECRET_NAME}" \
    --region "${AWS_REGION}" \
    --query "ARN" --output text 2>/dev/null || echo "arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${HF_TOKEN_SECRET_NAME}-PENDING")

cat > "${TMPDIR_PATH}/download-policy.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ModelBucketWrite",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::${MODEL_BUCKET}",
        "arn:aws:s3:::${MODEL_BUCKET}/*"
      ]
    },
    {
      "Sid": "HFTokenRead",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "${HF_SECRET_ARN}"
    }
  ]
}
EOF

aws iam put-role-policy \
    --role-name "${DOWNLOAD_IAM_ROLE_NAME}" \
    --policy-name "oai-infopt-download-policy" \
    --policy-document "$(read_json "${TMPDIR_PATH}/download-policy.json")"

log_info "Download policy attached → s3://${MODEL_BUCKET}/* + Secrets Manager HF token"

# ==============================================================================
# Step 3 — Serving Role (S3 read-only on model prefix — no write, no delete)
# ==============================================================================
log_step "Serving IAM Role: ${SERVING_IAM_ROLE_NAME}"

create_or_update_role "${SERVING_IAM_ROLE_NAME}" \
    "oai-infopt: vLLM serving - S3 read-only on model prefix"

cat > "${TMPDIR_PATH}/serving-policy.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ModelBucketReadOnly",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::${MODEL_BUCKET}",
        "arn:aws:s3:::${MODEL_BUCKET}/*"
      ]
    }
  ]
}
EOF

aws iam put-role-policy \
    --role-name "${SERVING_IAM_ROLE_NAME}" \
    --policy-name "oai-infopt-serving-policy" \
    --policy-document "$(read_json "${TMPDIR_PATH}/serving-policy.json")"

log_info "Serving policy attached → s3://${MODEL_BUCKET}/* (read-only)"

# ==============================================================================
# Step 4 — Benchmark Role (results S3 write + CloudWatch metrics)
# ==============================================================================
log_step "Benchmark IAM Role: ${BENCHMARK_IAM_ROLE_NAME}"

create_or_update_role "${BENCHMARK_IAM_ROLE_NAME}" \
    "oai-infopt: benchmarking - results S3 write + CloudWatch PutMetricData"

cat > "${TMPDIR_PATH}/benchmark-policy.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ResultsBucketWrite",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::${RESULTS_BUCKET}",
        "arn:aws:s3:::${RESULTS_BUCKET}/results/*"
      ]
    },
    {
      "Sid": "CloudWatchMetrics",
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "cloudwatch:namespace": "oai-infopt/benchmark"
        }
      }
    }
  ]
}
EOF

aws iam put-role-policy \
    --role-name "${BENCHMARK_IAM_ROLE_NAME}" \
    --policy-name "oai-infopt-benchmark-policy" \
    --policy-document "$(read_json "${TMPDIR_PATH}/benchmark-policy.json")"

log_info "Benchmark policy attached → s3://${RESULTS_BUCKET}/results/* + CloudWatch"

# ==============================================================================
# Step 5 — Enable Pod Identity addon
# ==============================================================================
log_step "EKS Pod Identity Addon"

ADDON_STATUS=$(aws eks describe-addon \
    --cluster-name "${CLUSTER_NAME}" \
    --addon-name eks-pod-identity-agent \
    --region "${AWS_REGION}" \
    --query "addon.status" --output text 2>/dev/null || echo "NOT_FOUND")

if [[ "${ADDON_STATUS}" == "ACTIVE" ]]; then
    log_warn "Pod Identity addon already ACTIVE — skipping."
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

# ==============================================================================
# Step 6 — Ensure namespace exists and apply ServiceAccounts
# ==============================================================================
log_step "Namespace + ServiceAccounts"

kubectl get namespace "${BENCHMARK_NAMESPACE}" >/dev/null 2>&1 || \
    kubectl create namespace "${BENCHMARK_NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -

kubectl label namespace "${BENCHMARK_NAMESPACE}" \
    project=observeai-inference-optimization \
    environment=sandbox \
    engagement=shellkode-sow \
    --overwrite

kubectl apply -f "${SCRIPT_DIR}/model-storage-sa.yaml"
log_info "ServiceAccounts applied in namespace '${BENCHMARK_NAMESPACE}'."

# ==============================================================================
# Step 7 — Create Pod Identity associations
# ==============================================================================
log_step "Pod Identity Associations"

associate_pod_identity() {
    local SA_NAME="$1"
    local ROLE_NAME="$2"
    local ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"

    EXISTING=$(aws eks list-pod-identity-associations \
        --cluster-name "${CLUSTER_NAME}" \
        --namespace "${BENCHMARK_NAMESPACE}" \
        --service-account "${SA_NAME}" \
        --region "${AWS_REGION}" \
        --query "associations[0].associationId" --output text 2>/dev/null || echo "None")

    if [[ "${EXISTING}" != "None" && "${EXISTING}" != "null" && -n "${EXISTING}" ]]; then
        log_warn "Association for ${SA_NAME} already exists (ID: ${EXISTING}) — skipping."
    else
        log_info "Creating association: ${SA_NAME} → ${ROLE_NAME}..."
        aws eks create-pod-identity-association \
            --cluster-name "${CLUSTER_NAME}" \
            --namespace "${BENCHMARK_NAMESPACE}" \
            --service-account "${SA_NAME}" \
            --role-arn "${ROLE_ARN}" \
            --region "${AWS_REGION}"
        log_info "Association created: ${SA_NAME} → ${ROLE_ARN}"
    fi
}

associate_pod_identity "${DOWNLOAD_SERVICE_ACCOUNT}"  "${DOWNLOAD_IAM_ROLE_NAME}"
associate_pod_identity "${SERVING_SERVICE_ACCOUNT}"   "${SERVING_IAM_ROLE_NAME}"
associate_pod_identity "${BENCHMARK_SERVICE_ACCOUNT}" "${BENCHMARK_IAM_ROLE_NAME}"

# ==============================================================================
# Done
# ==============================================================================
echo ""
echo "══════════════════════════════════════════════════════"
echo "  POD IDENTITY SETUP COMPLETE"
echo "══════════════════════════════════════════════════════"
echo ""
echo "  Roles created (least-privilege):"
echo "    Download  : arn:aws:iam::${AWS_ACCOUNT_ID}:role/${DOWNLOAD_IAM_ROLE_NAME}"
echo "                → s3://${MODEL_BUCKET}/* (rw) + Secrets Manager HF token"
echo "    Serving   : arn:aws:iam::${AWS_ACCOUNT_ID}:role/${SERVING_IAM_ROLE_NAME}"
echo "                → s3://${MODEL_BUCKET}/* (read-only)"
echo "    Benchmark : arn:aws:iam::${AWS_ACCOUNT_ID}:role/${BENCHMARK_IAM_ROLE_NAME}"
echo "                → s3://${RESULTS_BUCKET}/results/* (rw) + CloudWatch"
echo ""
echo "  Namespace   : ${BENCHMARK_NAMESPACE}"
echo "  HF Token    : aws secretsmanager get-secret-value --secret-id ${HF_TOKEN_SECRET_NAME}"
echo "══════════════════════════════════════════════════════"
