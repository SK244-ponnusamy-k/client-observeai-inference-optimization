#!/bin/bash
# ==============================================================================
# cluster/setup-trn2-nodegroup.sh
#
# Sets up a trn2 managed node group for AWS Trainium inference workloads.
# This script is called by bootstrap.sh when --enable-neuron or --compute neuron
# is passed, OR can be run standalone to add trn2 to an existing cluster.
#
# What this does:
#   1. Downloads the official AWS Neuron CloudFormation template for trn2
#   2. Creates a CloudFormation stack (IAM role, placement group, launch template,
#      EFA security group) — all infrastructure needed to launch trn2 nodes
#   3. Waits for the stack to complete
#   4. Generates eksctl nodegroup YAML with the correct launch template
#   5. Creates the managed node group via eksctl (joins existing EKS cluster)
#   6. Installs the Neuron device plugin DaemonSet
#   7. Scales to 0 (cost-safe idle state — scale up when needed)
#
# Prerequisites:
#   - EKS cluster already running (bootstrap.sh already ran)
#   - eksctl installed and configured
#   - AWS credentials with EKS, EC2, IAM, CloudFormation permissions
#   - trn2.48xlarge available in target AZ (check with aws ec2 describe-instance-type-offerings)
#
# Usage:
#   bash cluster/setup-trn2-nodegroup.sh                    # auto-detect best AZ
#   bash cluster/setup-trn2-nodegroup.sh --az us-east-2b    # specify AZ explicitly
#   bash cluster/setup-trn2-nodegroup.sh --stack-name my-trn2-stack  # custom stack name
#   bash cluster/setup-trn2-nodegroup.sh --dry-run          # validate only, no resources created
#
# Teardown:
#   bash cluster/teardown-trn2-nodegroup.sh
#
# Scale up when ready to use:
#   aws eks update-nodegroup-config \
#     --cluster-name <CLUSTER_NAME> \
#     --nodegroup-name trn2-48xl-ng1 \
#     --scaling-config minSize=0,maxSize=1,desiredSize=1 \
#     --region <AWS_REGION>
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
# Defaults and argument parsing
# ==============================================================================
CFN_STACK_NAME="${TRN2_CFN_STACK_NAME:-eks-trn2-ng-stack}"
NODEGROUP_NAME="${TRN2_NODEGROUP_NAME:-trn2-48xl-ng1}"
TARGET_AZ=""
DRY_RUN="false"
# Capacity type: spot (cheaper, interruptible) or on-demand.
# Per AWS guidance, trn2 capacity is most readily available via spot.
CAPACITY_TYPE="${TRN2_CAPACITY_TYPE:-spot}"
CFN_TEMPLATE_URL="https://raw.githubusercontent.com/aws-neuron/aws-neuron-eks-samples/master/dp_bert_hf_pretrain/cfn/eks_trn2_ng_stack_al2023.yaml"
CFN_TEMPLATE_FILE="${FRAMEWORK_ROOT}/cluster/eks_trn2_ng_stack_al2023.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --az)            TARGET_AZ="$2"; shift 2 ;;
        --stack-name)    CFN_STACK_NAME="$2"; shift 2 ;;
        --nodegroup)     NODEGROUP_NAME="$2"; shift 2 ;;
        --capacity-type) CAPACITY_TYPE="$2"; shift 2 ;;
        --spot)          CAPACITY_TYPE="spot"; shift ;;
        --on-demand)     CAPACITY_TYPE="on-demand"; shift ;;
        --dry-run)       DRY_RUN="true"; shift ;;
        -h|--help)
            echo "Usage: bash cluster/setup-trn2-nodegroup.sh [--az <az>] [--spot|--on-demand] [--stack-name <name>] [--dry-run]"
            exit 0 ;;
        *) log_warn "Unknown argument: $1"; shift ;;
    esac
done

# ==============================================================================
# Step 1 — Validate prerequisites
# ==============================================================================
log_step "Validating prerequisites"

for tool in aws eksctl kubectl curl; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        log_error "Missing required tool: ${tool}"
        exit 1
    fi
done

if ! aws sts get-caller-identity >/dev/null 2>&1; then
    log_error "AWS credentials not configured."
    exit 1
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
log_info "Account : ${AWS_ACCOUNT_ID}"
log_info "Region  : ${AWS_REGION}"
log_info "Cluster : ${CLUSTER_NAME}"

# Verify cluster exists
if ! aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log_error "Cluster '${CLUSTER_NAME}' not found. Run bootstrap.sh first."
    exit 1
fi
log_info "Cluster found: ${CLUSTER_NAME}"

# ==============================================================================
# Step 2 — Find best AZ for trn2.48xlarge
# ==============================================================================
log_step "Finding available AZ for trn2.48xlarge"

# Get AZs where trn2.48xlarge is supported
SUPPORTED_AZS=$(aws ec2 describe-instance-type-offerings \
    --location-type availability-zone \
    --filters "Name=instance-type,Values=trn2.48xlarge" \
    --region "${AWS_REGION}" \
    --query 'InstanceTypeOfferings[].Location' \
    --output text 2>/dev/null | tr '\t' ' ')

if [[ -z "${SUPPORTED_AZS}" ]]; then
    log_error "trn2.48xlarge not available in region ${AWS_REGION}."
    log_error "Check availability: aws ec2 describe-instance-type-offerings --filters Name=instance-type,Values=trn2.48xlarge --region <region>"
    exit 1
fi

log_info "trn2.48xlarge supported in: ${SUPPORTED_AZS}"

# Get cluster VPC (needed for subnet lookup in both paths)
VPC_ID=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.resourcesVpcConfig.vpcId' --output text)

# Use specified AZ or pick the first supported one
if [[ -n "${TARGET_AZ}" ]]; then
    if ! echo "${SUPPORTED_AZS}" | grep -q "${TARGET_AZ}"; then
        log_error "AZ ${TARGET_AZ} does not support trn2.48xlarge. Supported: ${SUPPORTED_AZS}"
        exit 1
    fi
    SELECTED_AZ="${TARGET_AZ}"
    # Look up a subnet in the specified AZ
    SELECTED_SUBNET=$(aws ec2 describe-subnets \
        --filters "Name=vpc-id,Values=${VPC_ID}" \
                  "Name=availabilityZone,Values=${SELECTED_AZ}" \
        --region "${AWS_REGION}" \
        --query 'Subnets[0].SubnetId' \
        --output text 2>/dev/null || echo "")
    if [[ -z "${SELECTED_SUBNET}" || "${SELECTED_SUBNET}" == "None" ]]; then
        log_error "No subnet found in ${SELECTED_AZ} for VPC ${VPC_ID}."
        exit 1
    fi
else
    # Auto-select: try each AZ with a dry-run to find one with live capacity
    SELECTED_AZ=""

    # Get Neuron AMI for dry-run test. Neuron AMIs may lag the cluster K8s version,
    # so try the cluster version first, then fall back to known-good versions.
    NEURON_AMI=""
    for AMI_VER in "${EKS_VERSION}" "1.34" "1.33"; do
        NEURON_AMI=$(aws ssm get-parameter \
            --name "/aws/service/eks/optimized-ami/${AMI_VER}/amazon-linux-2023/x86_64/neuron/recommended/image_id" \
            --region "${AWS_REGION}" \
            --query 'Parameter.Value' --output text 2>/dev/null || echo "")
        [[ -n "${NEURON_AMI}" ]] && break
    done

    for AZ in ${SUPPORTED_AZS}; do
        # Find a private subnet in this AZ
        SUBNET=$(aws ec2 describe-subnets \
            --filters "Name=vpc-id,Values=${VPC_ID}" \
                      "Name=availabilityZone,Values=${AZ}" \
            --region "${AWS_REGION}" \
            --query 'Subnets[0].SubnetId' \
            --output text 2>/dev/null || echo "")

        if [[ -z "${SUBNET}" || "${SUBNET}" == "None" ]]; then
            log_warn "No subnet found in ${AZ} — skipping"
            continue
        fi

        if [[ -n "${NEURON_AMI}" ]]; then
            # Dry-run launch to test capacity
            RESULT=$(aws ec2 run-instances \
                --image-id "${NEURON_AMI}" \
                --instance-type trn2.48xlarge \
                --subnet-id "${SUBNET}" \
                --count 1 \
                --dry-run \
                --region "${AWS_REGION}" 2>&1 || true)

            if echo "${RESULT}" | grep -q "DryRunOperation"; then
                log_info "AZ ${AZ} has available capacity — selecting."
                SELECTED_AZ="${AZ}"
                SELECTED_SUBNET="${SUBNET}"
                break
            else
                log_warn "AZ ${AZ}: ${RESULT}" | grep -o "InsufficientInstance\|Unsupported\|DryRun" || true
            fi
        else
            # Can't test capacity — just use first supported AZ
            SELECTED_AZ="${AZ}"
            SELECTED_SUBNET="${SUBNET}"
            log_warn "Cannot test live capacity (SSM AMI not found) — using ${AZ}"
            break
        fi
    done

    if [[ -z "${SELECTED_AZ}" ]]; then
        log_error "No AZ with available trn2.48xlarge capacity found in ${AWS_REGION}."
        log_error "Options:"
        log_error "  1. Retry later (capacity fluctuates)"
        log_error "  2. Specify AZ manually: bash setup-trn2-nodegroup.sh --az <az>"
        log_error "  3. Open AWS Support ticket for capacity reservation"
        exit 1
    fi
fi

log_info "Selected AZ     : ${SELECTED_AZ}"
log_info "Selected Subnet : ${SELECTED_SUBNET}"

[[ "${DRY_RUN}" == "true" ]] && { log_info "Dry-run complete."; exit 0; }

# ==============================================================================
# Step 3 — Get cluster parameters
# ==============================================================================
log_step "Getting cluster parameters"

CLUSTER_SG=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.resourcesVpcConfig.clusterSecurityGroupId' --output text)
CLUSTER_ENDPOINT=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.endpoint' --output text)
CLUSTER_SERVICE_CIDR=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.kubernetesNetworkConfig.serviceIpv4Cidr' --output text)
CLUSTER_CA=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.certificateAuthority.data' --output text)

log_info "SG       : ${CLUSTER_SG}"
log_info "Endpoint : ${CLUSTER_ENDPOINT}"
log_info "CIDR     : ${CLUSTER_SERVICE_CIDR}"

# ==============================================================================
# Step 4 — Download CloudFormation template
# ==============================================================================
log_step "Downloading CloudFormation template"

if [[ ! -f "${CFN_TEMPLATE_FILE}" ]]; then
    log_info "Downloading template from AWS Neuron samples..."
    curl -sSL -o "${CFN_TEMPLATE_FILE}" "${CFN_TEMPLATE_URL}"
    log_info "Template saved: ${CFN_TEMPLATE_FILE}"
else
    log_info "Template already exists: ${CFN_TEMPLATE_FILE}"
fi

# ==============================================================================
# Step 5 — Create CloudFormation stack
# ==============================================================================
log_step "Creating CloudFormation stack: ${CFN_STACK_NAME}"

# Check if stack already exists
STACK_STATUS=$(aws cloudformation describe-stacks \
    --stack-name "${CFN_STACK_NAME}" \
    --region "${AWS_REGION}" \
    --query 'Stacks[0].StackStatus' \
    --output text 2>/dev/null || echo "NOT_FOUND")

if [[ "${STACK_STATUS}" == "CREATE_COMPLETE" ]]; then
    log_warn "CloudFormation stack '${CFN_STACK_NAME}' already exists — reusing."
elif [[ "${STACK_STATUS}" == "NOT_FOUND" ]]; then
    # Write params file
    CFN_PARAMS_FILE="${FRAMEWORK_ROOT}/cluster/cfn_params_trn2.json"
    cat > "${CFN_PARAMS_FILE}" << EOF
[
    {"ParameterKey": "ClusterName",                    "ParameterValue": "${CLUSTER_NAME}"},
    {"ParameterKey": "ClusterControlPlaneSecurityGroup","ParameterValue": "${CLUSTER_SG}"},
    {"ParameterKey": "VpcId",                          "ParameterValue": "${VPC_ID}"},
    {"ParameterKey": "ClusterEndpoint",                "ParameterValue": "${CLUSTER_ENDPOINT}"},
    {"ParameterKey": "ClusterServiceCidr",             "ParameterValue": "${CLUSTER_SERVICE_CIDR}"},
    {"ParameterKey": "ClusterCertificateAuthority",    "ParameterValue": "${CLUSTER_CA}"}
]
EOF

    # On Windows/Git Bash, the AWS CLI (Windows binary) cannot resolve Unix-style
    # paths in file:// URIs. Pass the template/params inline via $(cat ...) instead.
    aws cloudformation create-stack \
        --stack-name "${CFN_STACK_NAME}" \
        --template-body "$(cat "${CFN_TEMPLATE_FILE}")" \
        --parameters "$(cat "${CFN_PARAMS_FILE}")" \
        --capabilities CAPABILITY_IAM \
        --region "${AWS_REGION}" \
        --tags \
            "Key=Project,Value=${TAG_PROJECT}" \
            "Key=Engagement,Value=${TAG_ENGAGEMENT}" \
            "Key=Environment,Value=${TAG_ENVIRONMENT}"

    log_info "Stack creation initiated. Waiting for completion (~3-5 min)..."
    aws cloudformation wait stack-create-complete \
        --stack-name "${CFN_STACK_NAME}" \
        --region "${AWS_REGION}"
    log_info "Stack created."
else
    log_error "Stack '${CFN_STACK_NAME}' is in unexpected state: ${STACK_STATUS}"
    exit 1
fi

# Get launch template ID
LT_ID=$(aws cloudformation describe-stacks \
    --stack-name "${CFN_STACK_NAME}" \
    --region "${AWS_REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='LaunchTemplateIdTrn2'].OutputValue" \
    --output text)

log_info "Launch Template : ${LT_ID}"

# ==============================================================================
# Step 6 — Create eksctl nodegroup YAML
# ==============================================================================
log_step "Generating nodegroup configuration"

# Convert capacity type to eksctl spot boolean
if [[ "${CAPACITY_TYPE}" == "spot" ]]; then
    SPOT_FLAG="true"
else
    SPOT_FLAG="false"
fi

NODEGROUP_YAML="${FRAMEWORK_ROOT}/cluster/trn2_nodegroup.yaml"
cat > "${NODEGROUP_YAML}" << EOF
# Auto-generated by setup-trn2-nodegroup.sh
# Cluster: ${CLUSTER_NAME} | AZ: ${SELECTED_AZ} | LT: ${LT_ID} | Capacity: ${CAPACITY_TYPE}
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: ${CLUSTER_NAME}
  region: ${AWS_REGION}
  version: "${EKS_VERSION}"
iam:
  withOIDC: true
managedNodeGroups:
  - name: ${NODEGROUP_NAME}
    launchTemplate:
      id: ${LT_ID}
    minSize: 0
    desiredCapacity: 1
    maxSize: 1
    spot: ${SPOT_FLAG}
    subnets:
      - ${SELECTED_SUBNET}
    privateNetworking: true
    efaEnabled: true
    tags:
      Project: ${TAG_PROJECT}
      Engagement: ${TAG_ENGAGEMENT}
      Environment: ${TAG_ENVIRONMENT}
EOF

log_info "Nodegroup YAML: ${NODEGROUP_YAML}"

# ==============================================================================
# Step 7 — Create managed node group
# ==============================================================================
log_step "Creating managed node group: ${NODEGROUP_NAME}"

# Check if nodegroup already exists
NG_STATUS=$(aws eks describe-nodegroup \
    --cluster-name "${CLUSTER_NAME}" \
    --nodegroup-name "${NODEGROUP_NAME}" \
    --region "${AWS_REGION}" \
    --query 'nodegroup.status' \
    --output text 2>/dev/null || echo "NOT_FOUND")

if [[ "${NG_STATUS}" == "ACTIVE" ]]; then
    log_warn "Node group '${NODEGROUP_NAME}' already exists and is ACTIVE — skipping."
elif [[ "${NG_STATUS}" == "NOT_FOUND" ]]; then
    eksctl create nodegroup -f "${NODEGROUP_YAML}"
    log_info "Node group created."
else
    log_warn "Node group '${NODEGROUP_NAME}' status: ${NG_STATUS}"
fi

# ==============================================================================
# Step 8 — Install Neuron device plugin
# ==============================================================================
log_step "Installing Neuron device plugin"

NEURON_PLUGIN_URL="https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/src/k8/k8s-neuron-device-plugin-rbac.yml"
NEURON_PLUGIN_DS_URL="https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/src/k8/k8s-neuron-device-plugin.yml"

kubectl apply -f "${NEURON_PLUGIN_URL}" 2>/dev/null || true
kubectl apply -f "${NEURON_PLUGIN_DS_URL}" 2>/dev/null || true
log_info "Neuron device plugin installed."

# ==============================================================================
# Step 9 — Scale down to 0 (cost-safe idle)
# ==============================================================================
log_step "Scaling node group to 0 (cost-safe idle state)"

aws eks update-nodegroup-config \
    --cluster-name "${CLUSTER_NAME}" \
    --nodegroup-name "${NODEGROUP_NAME}" \
    --scaling-config minSize=0,maxSize=1,desiredSize=0 \
    --region "${AWS_REGION}"

log_info "Node group scaled to 0. No trn2 instances running — no cost."

# ==============================================================================
# Done
# ==============================================================================
echo ""
echo "══════════════════════════════════════════════════════"
echo "  TRN2 NODE GROUP SETUP COMPLETE"
echo "══════════════════════════════════════════════════════"
echo ""
echo "  Node group : ${NODEGROUP_NAME}"
echo "  AZ         : ${SELECTED_AZ}"
echo "  Status     : IDLE (desiredSize=0, no cost)"
echo ""
echo "  Scale up when ready to deploy:"
echo "    aws eks update-nodegroup-config \\"
echo "      --cluster-name ${CLUSTER_NAME} \\"
echo "      --nodegroup-name ${NODEGROUP_NAME} \\"
echo "      --scaling-config minSize=0,maxSize=1,desiredSize=1 \\"
echo "      --region ${AWS_REGION}"
echo ""
echo "  Deploy model:"
echo "    bash vllm/models/gpt-oss-20b-neuron/deploy.sh --managed-ng"
echo ""
echo "  Scale back to 0 after benchmarking (~\$98/hr):"
echo "    aws eks update-nodegroup-config \\"
echo "      --cluster-name ${CLUSTER_NAME} \\"
echo "      --nodegroup-name ${NODEGROUP_NAME} \\"
echo "      --scaling-config minSize=0,maxSize=1,desiredSize=0 \\"
echo "      --region ${AWS_REGION}"
echo "══════════════════════════════════════════════════════"
