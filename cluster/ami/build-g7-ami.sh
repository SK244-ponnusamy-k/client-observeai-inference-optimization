#!/bin/bash
# ==============================================================================
# cluster/ami/build-g7-ami.sh
#
# Builds the custom EKS AL2023 AMI with NVIDIA Driver 595+ for EC2 G7 instances,
# and publishes the resulting AMI ID to AWS SSM Parameter Store.
#
# Usage:
#   bash cluster/ami/build-g7-ami.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${FRAMEWORK_ROOT}/config/config.env" ]]; then
    source "${FRAMEWORK_ROOT}/config/config.env"
fi

AWS_REGION="${AWS_REGION:-us-east-2}"
SSM_PARAM="${G7_SSM_AMI_PARAM:-/eks/ami/al2023-nvidia-595}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

if ! command -v packer >/dev/null 2>&1; then
    log_error "Packer is required to build custom G7 AMI. Please install packer: https://developer.hashicorp.com/packer/downloads"
    exit 1
fi

log_info "Initializing Packer plugins..."
packer init "${SCRIPT_DIR}/g7-eks-al2023.pkr.hcl"

log_info "Building custom G7 AMI (NVIDIA Driver 595+)..."
PACKER_LOG_FILE=$(mktemp /tmp/packer-build-XXXXXX.log)
packer build -machine-readable "${SCRIPT_DIR}/g7-eks-al2023.pkr.hcl" | tee "${PACKER_LOG_FILE}"

# Parse the AMI ID from the saved log file (avoids /dev/tty capture issues on Windows/MinGW)
AMI_ID=$(grep -E 'artifact,[0-9]+,id,' "${PACKER_LOG_FILE}" | awk -F',' '{print $NF}' | awk -F':' '{print $NF}' | tr -d '[:space:]' | tail -1)
rm -f "${PACKER_LOG_FILE}"

if [[ -z "${AMI_ID}" ]]; then
    log_error "Failed to parse built AMI ID from Packer output."
    exit 1
fi

log_info "Successfully built AMI: ${AMI_ID}"

log_info "Publishing AMI ID to AWS SSM Parameter Store (${SSM_PARAM})..."
MSYS_NO_PATHCONV=1 aws ssm put-parameter \
    --name "${SSM_PARAM}" \
    --value "${AMI_ID}" \
    --type "String" \
    --overwrite \
    --region "${AWS_REGION}"

log_info "Saved AMI ID ${AMI_ID} to SSM Parameter ${SSM_PARAM}."
log_info "Next cluster setup will automatically pick up this G7 AMI."
