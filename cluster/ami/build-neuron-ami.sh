#!/bin/bash
# ==============================================================================
# cluster/ami/build-neuron-ami.sh
#
# Builds a custom EKS Neuron AMI with updated Neuron driver 2.31 for
# ap-southeast-4 where AWS hasn't published a newer Neuron AMI yet.
#
# This is needed because:
#   - Latest AWS Neuron AMI in ap-southeast-4: v20261003 (driver 2.30.2)
#   - gpt-oss-20b requires vLLM-Neuron sdk2.31+ (NRT 2.33+)
#   - Driver 2.30.2 is incompatible with NRT 2.33
#
# What this does:
#   1. Takes the existing Neuron AMI as base
#   2. Updates the Neuron driver to 2.31-compatible version via Packer
#   3. Creates a new AMI and stores its ID in SSM Parameter Store
#   4. Updates the launch template to use the new AMI
#
# Prerequisites:
#   - Packer installed: https://developer.hashicorp.com/packer/downloads
#   - AWS credentials with EC2 + SSM permissions
#
# Usage:
#   bash cluster/ami/build-neuron-ami.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

SSM_PARAM="/eks/ami/al2023-neuron-sdk231"

# Check Packer
if ! command -v packer >/dev/null 2>&1; then
    log_error "Packer is required. Install it:"
    log_error "  https://developer.hashicorp.com/packer/downloads"
    log_error "  Or: choco install packer (Windows)"
    exit 1
fi

log_info "Region  : ${AWS_REGION}"
log_info "Cluster : ${CLUSTER_NAME}"
log_info "Base AMI: ami-012051d1c93d39733 (v20261003, driver 2.30.2)"
log_info "Target  : Neuron driver 2.31-compatible"

echo ""
read -r -p "Build custom Neuron AMI? This takes ~10-15 minutes. [y/N]: " CONFIRM
[[ "${CONFIRM}" =~ ^[yY]$ ]] || { log_info "Cancelled."; exit 0; }

# Initialize Packer plugins
log_info "Initializing Packer plugins..."
packer init "${SCRIPT_DIR}/neuron-eks-al2023.pkr.hcl"

# Build AMI
log_info "Building custom Neuron AMI (10-15 min)..."
PACKER_LOG_FILE=$(mktemp /tmp/packer-neuron-XXXXXX.log)

packer build \
  -var "aws_region=${AWS_REGION}" \
  -var "eks_version=${EKS_VERSION}" \
  -machine-readable \
  "${SCRIPT_DIR}/neuron-eks-al2023.pkr.hcl" | tee "${PACKER_LOG_FILE}"

# Parse AMI ID
AMI_ID=$(grep -E 'artifact,[0-9]+,id,' "${PACKER_LOG_FILE}" | \
  awk -F',' '{print $NF}' | awk -F':' '{print $NF}' | tr -d '[:space:]' | tail -1)
rm -f "${PACKER_LOG_FILE}"

if [[ -z "${AMI_ID}" ]]; then
    log_error "Failed to parse AMI ID from Packer output."
    exit 1
fi

log_info "AMI built: ${AMI_ID}"

# Store in SSM
log_info "Storing AMI ID in SSM Parameter Store (${SSM_PARAM})..."
MSYS_NO_PATHCONV=1 aws ssm put-parameter \
  --name "${SSM_PARAM}" \
  --value "${AMI_ID}" \
  --type "String" \
  --overwrite \
  --region "${AWS_REGION}"

log_info "Stored in SSM: ${SSM_PARAM} = ${AMI_ID}"

# Create new launch template version with updated AMI
log_info "Creating new launch template version with custom AMI..."
NEW_LT_VERSION=$(aws ec2 create-launch-template-version \
  --launch-template-id "${TRN2_LAUNCH_TEMPLATE_ID}" \
  --source-version '$Latest' \
  --launch-template-data "{\"ImageId\":\"${AMI_ID}\"}" \
  --region "${AWS_REGION}" \
  --query 'LaunchTemplateVersion.VersionNumber' \
  --output text)

log_info "New LT version: ${NEW_LT_VERSION}"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  CUSTOM NEURON AMI BUILD COMPLETE"
echo "══════════════════════════════════════════════════════"
echo ""
echo "  AMI ID         : ${AMI_ID}"
echo "  SSM Parameter  : ${SSM_PARAM}"
echo "  LT Version     : ${NEW_LT_VERSION}"
echo ""
echo "  Next steps:"
echo "  1. Scale node group to 0:"
echo "     aws eks update-nodegroup-config \\"
echo "       --cluster-name ${CLUSTER_NAME} \\"
echo "       --nodegroup-name ${TRN2_NODEGROUP_NAME} \\"
echo "       --scaling-config minSize=0,maxSize=1,desiredSize=0 \\"
echo "       --region ${AWS_REGION}"
echo ""
echo "  2. Update node group to use new LT version ${NEW_LT_VERSION}:"
echo "     aws eks update-nodegroup-version \\"
echo "       --cluster-name ${CLUSTER_NAME} \\"
echo "       --nodegroup-name ${TRN2_NODEGROUP_NAME} \\"
echo "       --launch-template id=${TRN2_LAUNCH_TEMPLATE_ID},version=${NEW_LT_VERSION} \\"
echo "       --region ${AWS_REGION}"
echo ""
echo "  3. Scale back to 1:"
echo "     aws eks update-nodegroup-config \\"
echo "       --cluster-name ${CLUSTER_NAME} \\"
echo "       --nodegroup-name ${TRN2_NODEGROUP_NAME} \\"
echo "       --scaling-config minSize=0,maxSize=1,desiredSize=1 \\"
echo "       --region ${AWS_REGION}"
echo ""
echo "  4. Deploy:"
echo "     oai deploy gpt-oss-20b-neuron --managed-ng"
echo "══════════════════════════════════════════════════════"
