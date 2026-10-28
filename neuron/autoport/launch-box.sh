#!/bin/bash
# ==============================================================================
# neuron/autoport/launch-box.sh
#
# Launches a Trainium "AutoPort box": a standalone trn2.48xlarge from the Neuron
# vLLM DLAMI, placed in an EC2 Capacity Block, reachable only via SSM Session
# Manager (private subnet of the EKS VPC, no SSH key, no inbound rules).
#
# The block's instance is then NOT available to the EKS neuron-inf NodePool
# until this box is terminated - porting and serving share the same capacity.
#
# Usage:
#   bash neuron/autoport/launch-box.sh --reservation cr-0123... --name autoport-gemma-4-31b [--dry-run]
#   bash neuron/autoport/launch-box.sh --terminate i-0123...
#
#   --dry-run       EC2 permission/parameter check only (launches nothing)
#   --disk-gb N     root volume size (default 1000: weights + compile cache)
#   --ami-param P   SSM parameter of the DLAMI (default: vLLM 0.29 / SDK 2.33 DLAMI)
#   --profile NAME  instance profile with SSM + S3 access (default SSMRoleForEC2)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

RESERVATION=""; NAME=""; DRY_RUN=""; TERMINATE=""
DISK_GB=1000
AMI_PARAM="/aws/service/neuron/dlami/pytorch-inference-vllm-0.29.0.1.2.0/ubuntu-24.04/latest/image_id"
PROFILE="SSMRoleForEC2"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reservation) RESERVATION="$2"; shift 2 ;;
        --name)        NAME="$2"; shift 2 ;;
        --dry-run)     DRY_RUN="--dry-run"; shift ;;
        --disk-gb)     DISK_GB="$2"; shift 2 ;;
        --ami-param)   AMI_PARAM="$2"; shift 2 ;;
        --profile)     PROFILE="$2"; shift 2 ;;
        --terminate)   TERMINATE="$2"; shift 2 ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -n "${TERMINATE}" ]]; then
    log_info "Terminating ${TERMINATE} (frees the Capacity Block slot for EKS)..."
    aws ec2 terminate-instances --region "${AWS_REGION}" --instance-ids "${TERMINATE}" \
        --query 'TerminatingInstances[0].CurrentState.Name' --output text
    exit 0
fi
[[ -n "${RESERVATION}" && -n "${NAME}" ]] || { log_error "--reservation and --name are required."; exit 1; }

read -r ITYPE AZ STATE AVAIL END < <(aws ec2 describe-capacity-reservations --region "${AWS_REGION}" \
    --capacity-reservation-ids "${RESERVATION}" \
    --query 'CapacityReservations[0].[InstanceType,AvailabilityZone,State,AvailableInstanceCount,EndDate]' --output text)
log_info "Reservation ${RESERVATION}: ${ITYPE} in ${AZ}, ${STATE}, ${AVAIL} free, ends ${END}"
[[ "${STATE}" == "active" ]] || { log_error "Reservation is ${STATE}, not active."; exit 1; }
[[ "${AVAIL}" -ge 1 ]] || { log_error "No free instance left in ${RESERVATION} (is EKS or another box using it?)."; exit 1; }

VPC=$(aws eks describe-cluster --region "${AWS_REGION}" --name "${CLUSTER_NAME}" \
    --query 'cluster.resourcesVpcConfig.vpcId' --output text)
SUBNET=$(aws ec2 describe-subnets --region "${AWS_REGION}" \
    --filters "Name=vpc-id,Values=${VPC}" "Name=availability-zone,Values=${AZ}" "Name=map-public-ip-on-launch,Values=false" \
    --query 'Subnets[0].SubnetId' --output text)
SG=$(aws ec2 describe-security-groups --region "${AWS_REGION}" \
    --filters "Name=vpc-id,Values=${VPC}" "Name=group-name,Values=default" --query 'SecurityGroups[0].GroupId' --output text)
# AWS-owned public SSM parameters (the DLAMI image-id paths) are not resolvable
# by the bare "/aws/service/..." name on every AWS CLI build - some require the
# full public ARN "arn:aws:ssm:<region>::parameter/aws/service/...". Try the bare
# name first, then fall back to the ARN form so this works across CLI versions.
AMI=$(aws ssm get-parameter --region "${AWS_REGION}" --name "${AMI_PARAM}" --query 'Parameter.Value' --output text 2>/dev/null || true)
if [[ -z "${AMI}" || "${AMI}" == "None" ]]; then
    AMI=$(aws ssm get-parameter --region "${AWS_REGION}" \
        --name "arn:aws:ssm:${AWS_REGION}::parameter${AMI_PARAM}" \
        --query 'Parameter.Value' --output text 2>/dev/null || true)
fi
[[ -n "${AMI}" && "${AMI}" != "None" ]] || { log_error "Could not resolve DLAMI AMI from SSM parameter ${AMI_PARAM} (tried bare name and public ARN). Pass an explicit AMI path with --ami-param, or check the parameter exists in ${AWS_REGION}."; exit 1; }
AMI_NAME=$(aws ec2 describe-images --region "${AWS_REGION}" --image-ids "${AMI}" --query 'Images[0].Name' --output text)
[[ "${SUBNET}" != "None" ]] || { log_error "No private subnet in ${AZ} in ${VPC}."; exit 1; }

echo ""
echo "=============================================================="
echo "  AutoPort box: ${NAME}${DRY_RUN:+   (dry run)}"
echo "  Instance  : ${ITYPE} in Capacity Block ${RESERVATION} (${AZ})"
echo "  AMI       : ${AMI}  ${AMI_NAME}"
echo "  Network   : ${SUBNET} (private) / ${SG} (default SG) - SSM only"
echo "  Disk      : ${DISK_GB} GB gp3     Profile: ${PROFILE}"
echo "=============================================================="

set +e
# MSYS_NO_PATHCONV=1: on Git Bash / MSYS2 (Windows) the shell rewrites any arg
# that looks like a Unix path (e.g. DeviceName=/dev/sda1) into a Windows path
# (C:/Program Files/Git/dev/sda1), which AWS rejects as InvalidBlockDeviceMapping.
# Disabling path conversion for this one call keeps /dev/sda1 intact. The var is
# ignored on native Linux, so this is a no-op there.
OUT=$(MSYS_NO_PATHCONV=1 aws ec2 run-instances --region "${AWS_REGION}" ${DRY_RUN} \
    --image-id "${AMI}" --instance-type "${ITYPE}" --count 1 \
    --instance-market-options MarketType=capacity-block \
    --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=${RESERVATION}}" \
    --placement "AvailabilityZone=${AZ}" \
    --subnet-id "${SUBNET}" --security-group-ids "${SG}" \
    --iam-instance-profile "Name=${PROFILE}" \
    --metadata-options HttpTokens=required,HttpEndpoint=enabled \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${DISK_GB},VolumeType=gp3,Iops=6000,Throughput=500,DeleteOnTermination=true,Encrypted=true}" \
    --tag-specifications \
      "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}},{Key=Project,Value=${TAG_PROJECT}},{Key=Engagement,Value=${TAG_ENGAGEMENT}},{Key=Environment,Value=${TAG_ENVIRONMENT}},{Key=Component,Value=autoport}]" \
      "ResourceType=volume,Tags=[{Key=Name,Value=${NAME}},{Key=Project,Value=${TAG_PROJECT}}]" \
    --query 'Instances[0].InstanceId' --output text 2>&1)
RC=$?
set -e

if [[ -n "${DRY_RUN}" ]]; then
    if grep -q "DryRunOperation" <<<"${OUT}"; then
        log_info "Dry run OK: you have permission and the request is valid. Nothing was launched."
        exit 0
    fi
    log_error "Dry run failed: ${OUT}"; exit 1
fi
[[ ${RC} -eq 0 ]] || { log_error "${OUT}"; exit 1; }

log_info "Launched ${OUT}. Waiting for it to run (and register with SSM)..."
aws ec2 wait instance-running --region "${AWS_REGION}" --instance-ids "${OUT}"
echo ""
echo "  Connect : aws ssm start-session --region ${AWS_REGION} --target ${OUT}"
echo "            (or EC2 console -> ${OUT} -> Connect -> Session Manager)"
echo "  Release : bash neuron/autoport/launch-box.sh --terminate ${OUT}"
