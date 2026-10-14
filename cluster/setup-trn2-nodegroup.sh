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
NODEGROUP_NAME="${TRN2_NODEGROUP_NAME:-trn2-neuron}"
TARGET_AZ=""
DRY_RUN="false"
# Capacity type: spot (cheaper, interruptible) or on-demand.
# Per AWS guidance, trn2 capacity is most readily available via spot.
CAPACITY_TYPE="${TRN2_CAPACITY_TYPE:-spot}"
# Trainium instance type — trn2.3xlarge is the available size in sa-east-1.
# trn2.48xlarge is not offered in sa-east-1 as of 2026-10.
TRN2_INSTANCE_TYPE="${TRN2_INSTANCE_TYPE:-trn2.3xlarge}"
CFN_TEMPLATE_URL="https://raw.githubusercontent.com/aws-neuron/aws-neuron-eks-samples/master/dp_bert_hf_pretrain/cfn/eks_trn2_ng_stack_al2023.yaml"
CFN_TEMPLATE_FILE="${FRAMEWORK_ROOT}/cluster/eks_trn2_ng_stack_al2023.yaml"

# Capacity Reservation mode — skip CloudFormation entirely and use a pre-existing
# capacity reservation. Set via --capacity-reservation or TRN2_CAPACITY_RESERVATION_ID.
USE_CAPACITY_RESERVATION="false"
CAPACITY_RESERVATION_ID="${TRN2_CAPACITY_RESERVATION_ID:-}"
LT_NAME="${TRN2_LAUNCH_TEMPLATE_NAME:-trn2-minimal-lt}"
LT_ID="${TRN2_LAUNCH_TEMPLATE_ID:-}"
ROOT_VOLUME_SIZE="${TRN2_ROOT_VOLUME_SIZE:-500}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --az)                    TARGET_AZ="$2"; shift 2 ;;
        --stack-name)            CFN_STACK_NAME="$2"; shift 2 ;;
        --nodegroup)             NODEGROUP_NAME="$2"; shift 2 ;;
        --capacity-type)         CAPACITY_TYPE="$2"; shift 2 ;;
        --spot)                  CAPACITY_TYPE="spot"; shift ;;
        --on-demand)             CAPACITY_TYPE="on-demand"; shift ;;
        --dry-run)               DRY_RUN="true"; shift ;;
        --capacity-reservation)  USE_CAPACITY_RESERVATION="true"; CAPACITY_RESERVATION_ID="$2"; shift 2 ;;
        --lt-name)               LT_NAME="$2"; shift 2 ;;
        --root-volume-size)      ROOT_VOLUME_SIZE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash cluster/setup-trn2-nodegroup.sh [OPTIONS]"
            echo ""
            echo "Standard mode (CloudFormation):"
            echo "  --az <az>              Target availability zone"
            echo "  --stack-name <name>    CloudFormation stack name"
            echo "  --spot|--on-demand     Capacity type"
            echo "  --dry-run              Validate only, no resources created"
            echo ""
            echo "Capacity Reservation mode (skip CloudFormation):"
            echo "  --capacity-reservation <cr-id>  Use existing capacity reservation"
            echo "  --lt-name <name>                Launch template name (default: trn2-minimal-lt)"
            echo "  --root-volume-size <gb>         Root EBS volume size in GB (default: 500)"
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
# Step 2 — Capacity Reservation fast path (skip CloudFormation entirely)
# ==============================================================================
if [[ "${USE_CAPACITY_RESERVATION}" == "true" || -n "${CAPACITY_RESERVATION_ID}" ]]; then
    log_step "Capacity Reservation mode — skipping CloudFormation"

    if [[ -z "${CAPACITY_RESERVATION_ID}" ]]; then
        log_error "No capacity reservation ID provided. Use --capacity-reservation <cr-id> or set TRN2_CAPACITY_RESERVATION_ID."
        exit 1
    fi

    # Validate the reservation exists and is active
    CR_STATE=$(aws ec2 describe-capacity-reservations \
        --capacity-reservation-ids "${CAPACITY_RESERVATION_ID}" \
        --region "${AWS_REGION}" \
        --query 'CapacityReservations[0].State' \
        --output text 2>/dev/null || echo "NOT_FOUND")

    if [[ "${CR_STATE}" != "active" ]]; then
        log_error "Capacity reservation '${CAPACITY_RESERVATION_ID}' is not active (state: ${CR_STATE})."
        exit 1
    fi

    CR_AZ=$(aws ec2 describe-capacity-reservations \
        --capacity-reservation-ids "${CAPACITY_RESERVATION_ID}" \
        --region "${AWS_REGION}" \
        --query 'CapacityReservations[0].AvailabilityZone' \
        --output text)
    CR_INSTANCE_TYPE=$(aws ec2 describe-capacity-reservations \
        --capacity-reservation-ids "${CAPACITY_RESERVATION_ID}" \
        --region "${AWS_REGION}" \
        --query 'CapacityReservations[0].InstanceType' \
        --output text)

    log_info "Capacity Reservation : ${CAPACITY_RESERVATION_ID}"
    log_info "Instance Type        : ${CR_INSTANCE_TYPE}"
    log_info "AZ                   : ${CR_AZ}"
    TRN2_INSTANCE_TYPE="${CR_INSTANCE_TYPE}"
    SELECTED_AZ="${CR_AZ}"

    [[ "${DRY_RUN}" == "true" ]] && { log_info "Dry-run complete."; exit 0; }

    # Get subnet in the reservation's AZ
    VPC_ID=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
        --query 'cluster.resourcesVpcConfig.vpcId' --output text)
    SELECTED_SUBNET=$(aws ec2 describe-subnets \
        --filters "Name=vpc-id,Values=${VPC_ID}" \
                  "Name=availabilityZone,Values=${SELECTED_AZ}" \
                  "Name=tag:Name,Values=*Private*" \
        --region "${AWS_REGION}" \
        --query 'Subnets[0].SubnetId' --output text 2>/dev/null || echo "")
    # Fallback: any subnet in the AZ if no private subnet tagged
    if [[ -z "${SELECTED_SUBNET}" || "${SELECTED_SUBNET}" == "None" ]]; then
        SELECTED_SUBNET=$(aws ec2 describe-subnets \
            --filters "Name=vpc-id,Values=${VPC_ID}" \
                      "Name=availabilityZone,Values=${SELECTED_AZ}" \
            --region "${AWS_REGION}" \
            --query 'Subnets[0].SubnetId' --output text 2>/dev/null || echo "")
    fi
    if [[ -z "${SELECTED_SUBNET}" || "${SELECTED_SUBNET}" == "None" ]]; then
        log_error "No subnet found in ${SELECTED_AZ} for VPC ${VPC_ID}."
        exit 1
    fi
    log_info "Subnet : ${SELECTED_SUBNET}"

    # Create or reuse launch template
    EXISTING_LT=$(aws ec2 describe-launch-templates \
        --filters "Name=launch-template-name,Values=${LT_NAME}" \
        --region "${AWS_REGION}" \
        --query 'LaunchTemplates[0].LaunchTemplateId' \
        --output text 2>/dev/null || echo "None")

    if [[ -n "${LT_ID}" && "${LT_ID}" != "None" ]]; then
        log_info "Reusing existing launch template: ${LT_ID} (${LT_NAME})"
        # Create new version with updated root volume size
        log_info "Creating new LT version with ${ROOT_VOLUME_SIZE}GB root volume..."
        LT_VERSION=$(aws ec2 create-launch-template-version \
            --launch-template-id "${LT_ID}" \
            --source-version '$Latest' \
            --launch-template-data "{\"BlockDeviceMappings\":[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":${ROOT_VOLUME_SIZE},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]}" \
            --region "${AWS_REGION}" \
            --query 'LaunchTemplateVersion.VersionNumber' \
            --output text)
        log_info "New LT version: ${LT_VERSION}"
    elif [[ -n "${EXISTING_LT}" && "${EXISTING_LT}" != "None" ]]; then
        LT_ID="${EXISTING_LT}"
        log_info "Found existing launch template by name: ${LT_ID}"
        LT_VERSION=$(aws ec2 describe-launch-templates \
            --launch-template-ids "${LT_ID}" \
            --region "${AWS_REGION}" \
            --query 'LaunchTemplates[0].LatestVersionNumber' \
            --output text)
    else
        # Get Neuron AMI
        NEURON_AMI=""
        for AMI_VER in "${EKS_VERSION}" "1.34" "1.33"; do
            NEURON_AMI=$(MSYS_NO_PATHCONV=1 aws ssm get-parameter \
                --name "/aws/service/eks/optimized-ami/${AMI_VER}/amazon-linux-2023/x86_64/neuron/recommended/image_id" \
                --region "${AWS_REGION}" \
                --query 'Parameter.Value' --output text 2>/dev/null || echo "")
            [[ -n "${NEURON_AMI}" ]] && break
        done
        [[ -z "${NEURON_AMI}" ]] && { log_error "Cannot find Neuron AMI for EKS ${EKS_VERSION} in ${AWS_REGION}."; exit 1; }
        log_info "Neuron AMI: ${NEURON_AMI}"

        log_info "Creating launch template: ${LT_NAME}"
        LT_ID=$(aws ec2 create-launch-template \
            --launch-template-name "${LT_NAME}" \
            --launch-template-data "{
                \"ImageId\": \"${NEURON_AMI}\",
                \"InstanceType\": \"${TRN2_INSTANCE_TYPE}\",
                \"BlockDeviceMappings\": [{
                    \"DeviceName\": \"/dev/xvda\",
                    \"Ebs\": {
                        \"VolumeSize\": ${ROOT_VOLUME_SIZE},
                        \"VolumeType\": \"gp3\",
                        \"DeleteOnTermination\": true
                    }
                }],
                \"InstanceMarketOptions\": {
                    \"MarketType\": \"capacity-block\"
                },
                \"CapacityReservationSpecification\": {
                    \"CapacityReservationTarget\": {
                        \"CapacityReservationId\": \"${CAPACITY_RESERVATION_ID}\"
                    }
                },
                \"TagSpecifications\": [{
                    \"ResourceType\": \"instance\",
                    \"Tags\": [
                        {\"Key\": \"Project\",   \"Value\": \"${TAG_PROJECT}\"},
                        {\"Key\": \"Engagement\",\"Value\": \"${TAG_ENGAGEMENT}\"},
                        {\"Key\": \"Environment\",\"Value\": \"${TAG_ENVIRONMENT}\"}
                    ]
                }]
            }" \
            --region "${AWS_REGION}" \
            --query 'LaunchTemplate.LaunchTemplateId' \
            --output text)
        LT_VERSION="1"
        log_info "Launch template created: ${LT_ID} (version ${LT_VERSION})"
    fi

    # Generate nodegroup YAML
    SECOND_AZ=$(aws ec2 describe-availability-zones \
        --region "${AWS_REGION}" \
        --filters "Name=state,Values=available" \
        --query "AvailabilityZones[?ZoneName!='${SELECTED_AZ}'].ZoneName | [0]" \
        --output text)

    NODEGROUP_YAML="${FRAMEWORK_ROOT}/cluster/trn2_nodegroup.yaml"
    cat > "${NODEGROUP_YAML}" << EOF
# Auto-generated by setup-trn2-nodegroup.sh --capacity-reservation
# Capacity Reservation: ${CAPACITY_RESERVATION_ID} | AZ: ${SELECTED_AZ} | LT: ${LT_ID} v${LT_VERSION} | Instance: ${TRN2_INSTANCE_TYPE}
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: ${CLUSTER_NAME}
  region: ${AWS_REGION}
  version: "${EKS_VERSION}"
iam:
  withOIDC: true
availabilityZones: ["${SELECTED_AZ}", "${SECOND_AZ}"]
managedNodeGroups:
  - name: ${NODEGROUP_NAME}
    launchTemplate:
      id: ${LT_ID}
      version: "${LT_VERSION}"
    minSize: 0
    desiredCapacity: 0
    maxSize: 1
    availabilityZones: ["${SELECTED_AZ}"]
    privateNetworking: true
    tags:
      Project: ${TAG_PROJECT}
      Engagement: ${TAG_ENGAGEMENT}
      Environment: ${TAG_ENVIRONMENT}
EOF
    log_info "Nodegroup YAML: ${NODEGROUP_YAML}"

    # Create or reuse node group
    NG_STATUS=$(aws eks describe-nodegroup \
        --cluster-name "${CLUSTER_NAME}" \
        --nodegroup-name "${NODEGROUP_NAME}" \
        --region "${AWS_REGION}" \
        --query 'nodegroup.status' \
        --output text 2>/dev/null || echo "NOT_FOUND")

    if [[ "${NG_STATUS}" == "ACTIVE" ]]; then
        log_warn "Node group '${NODEGROUP_NAME}' already exists — updating launch template version..."
        aws eks update-nodegroup-version \
            --cluster-name "${CLUSTER_NAME}" \
            --nodegroup-name "${NODEGROUP_NAME}" \
            --launch-template "id=${LT_ID},version=${LT_VERSION}" \
            --region "${AWS_REGION}"
        log_info "Node group LT version updated."
    elif [[ "${NG_STATUS}" == "NOT_FOUND" ]]; then
        eksctl create nodegroup -f "${NODEGROUP_YAML}"
        log_info "Node group created."
    else
        log_warn "Node group '${NODEGROUP_NAME}' status: ${NG_STATUS}"
    fi

    # Install Neuron device plugin
    log_step "Installing Neuron device plugin"
    kubectl apply -f "https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/src/k8/k8s-neuron-device-plugin-rbac.yml" 2>/dev/null || true
    kubectl apply -f "https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/src/k8/k8s-neuron-device-plugin.yml" 2>/dev/null || true
    log_info "Neuron device plugin installed."

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  TRN2 NODE GROUP SETUP COMPLETE (Capacity Reservation)"
    echo "══════════════════════════════════════════════════════"
    echo ""
    echo "  Node group          : ${NODEGROUP_NAME}"
    echo "  Capacity Reservation: ${CAPACITY_RESERVATION_ID}"
    echo "  Instance Type       : ${TRN2_INSTANCE_TYPE}"
    echo "  AZ                  : ${SELECTED_AZ}"
    echo "  Launch Template     : ${LT_ID} (version ${LT_VERSION}, ${ROOT_VOLUME_SIZE}GB root)"
    echo "  Status              : IDLE (desiredSize=0, no cost)"
    echo ""
    echo "  Scale up when ready:"
    echo "    aws eks update-nodegroup-config \\"
    echo "      --cluster-name ${CLUSTER_NAME} \\"
    echo "      --nodegroup-name ${NODEGROUP_NAME} \\"
    echo "      --scaling-config minSize=0,maxSize=1,desiredSize=1 \\"
    echo "      --region ${AWS_REGION}"
    echo "══════════════════════════════════════════════════════"
    exit 0
fi

# ==============================================================================
# Step 2 — Find best AZ for ${TRN2_INSTANCE_TYPE}
# ==============================================================================
log_step "Finding available AZ for ${TRN2_INSTANCE_TYPE}"

# Get AZs where the target instance type is supported
SUPPORTED_AZS=$(aws ec2 describe-instance-type-offerings \
    --location-type availability-zone \
    --filters "Name=instance-type,Values=${TRN2_INSTANCE_TYPE}" \
    --region "${AWS_REGION}" \
    --query 'InstanceTypeOfferings[].Location' \
    --output text 2>/dev/null | tr '\t' ' ')

if [[ -z "${SUPPORTED_AZS}" ]]; then
    log_error "${TRN2_INSTANCE_TYPE} not available in region ${AWS_REGION}."
    log_error "Check availability: aws ec2 describe-instance-type-offerings --filters Name=instance-type,Values=${TRN2_INSTANCE_TYPE} --region ${AWS_REGION}"
    exit 1
fi

log_info "${TRN2_INSTANCE_TYPE} supported in: ${SUPPORTED_AZS}"

# Get cluster VPC (needed for subnet lookup in both paths)
VPC_ID=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.resourcesVpcConfig.vpcId' --output text)

# Use specified AZ or pick the first supported one
if [[ -n "${TARGET_AZ}" ]]; then
    if ! echo "${SUPPORTED_AZS}" | grep -q "${TARGET_AZ}"; then
        log_error "AZ ${TARGET_AZ} does not support ${TRN2_INSTANCE_TYPE}. Supported: ${SUPPORTED_AZS}"
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

    # Get Neuron AMI for dry-run test
    NEURON_AMI=""
    for AMI_VER in "${EKS_VERSION}" "1.34" "1.33"; do
        NEURON_AMI=$(MSYS_NO_PATHCONV=1 aws ssm get-parameter \
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
                --instance-type "${TRN2_INSTANCE_TYPE}" \
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
                log_warn "AZ ${AZ}: capacity check inconclusive — trying next."
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
        log_error "No AZ with available ${TRN2_INSTANCE_TYPE} capacity found in ${AWS_REGION}."
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
# Step 4 — Download CloudFormation template and patch instance type
# ==============================================================================
log_step "Downloading CloudFormation template"

if [[ ! -f "${CFN_TEMPLATE_FILE}" ]]; then
    log_info "Downloading template from AWS Neuron samples..."
    curl -sSL -o "${CFN_TEMPLATE_FILE}" "${CFN_TEMPLATE_URL}"
    log_info "Template saved: ${CFN_TEMPLATE_FILE}"
else
    log_info "Template already exists: ${CFN_TEMPLATE_FILE}"
fi

# Patch instance type in the template if it differs from default trn2.48xlarge.
# The AWS template hardcodes trn2.48xlarge — replace it with TRN2_INSTANCE_TYPE.
if [[ "${TRN2_INSTANCE_TYPE}" != "trn2.48xlarge" ]]; then
    log_info "Patching template instance type: trn2.48xlarge → ${TRN2_INSTANCE_TYPE}"
    sed -i "s/trn2\.48xlarge/${TRN2_INSTANCE_TYPE}/g" "${CFN_TEMPLATE_FILE}"
fi

# Remove Placement Group from the template.
# Placement Groups are incompatible with Capacity Block reservations (AWS restriction).
# For single-node inference this is not needed anyway.
log_info "Removing Placement Group from template (incompatible with Capacity Block reservations)"
sed -i '/PlacementGroup\|PlacementGroupId/d' "${CFN_TEMPLATE_FILE}"
sed -i '/GroupName:/d' "${CFN_TEMPLATE_FILE}"

# trn2.3xlarge has only 1 network card (index 0) and supports only 1 EFA interface.
# trn2.48xlarge has 16 EFA interfaces across multiple network cards.
# For trn2.3xlarge: keep only the first interface entry in each NetworkInterfaces block.
if [[ "${TRN2_INSTANCE_TYPE}" == "trn2.3xlarge" ]]; then
    log_info "Patching template for trn2.3xlarge: removing extra EFA interfaces (keeping only 1)"
    awk '
    /^        NetworkInterfaces:/ {
        in_ni = 1
        first_entry = 0
        print
        next
    }
    in_ni && /^          - / {
        if (first_entry == 0) {
            first_entry = 1
            skip = 0
        } else {
            skip = 1
        }
    }
    in_ni && /^        [A-Z]/ {
        in_ni = 0
        skip = 0
    }
    !skip { print }
    ' "${CFN_TEMPLATE_FILE}" > "${CFN_TEMPLATE_FILE}.patched"
    mv "${CFN_TEMPLATE_FILE}.patched" "${CFN_TEMPLATE_FILE}"
    log_info "Patched: kept only first EFA interface per NetworkInterfaces block"
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
    log_info "Stack created successfully."
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

# eksctl requires >= 2 AZs at cluster level even when the node group pins to 1.
# Pick a second AZ from the region — just needs to exist, not have trn2 capacity.
SECOND_AZ=$(aws ec2 describe-availability-zones \
    --region "${AWS_REGION}" \
    --filters "Name=state,Values=available" \
    --query "AvailabilityZones[?ZoneName!='${SELECTED_AZ}'].ZoneName | [0]" \
    --output text)
log_info "Cluster-level AZs: ${SELECTED_AZ}, ${SECOND_AZ}"

cat > "${NODEGROUP_YAML}" << EOF
# Auto-generated by setup-trn2-nodegroup.sh
# Docs: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/containers/tutorials/k8s-setup.html
# Cluster: ${CLUSTER_NAME} | AZ: ${SELECTED_AZ} | LT: ${LT_ID} | Instance: ${TRN2_INSTANCE_TYPE} | Capacity: ${CAPACITY_TYPE}
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: ${CLUSTER_NAME}
  region: ${AWS_REGION}
  version: "${EKS_VERSION}"
iam:
  withOIDC: true
availabilityZones: ["${SELECTED_AZ}", "${SECOND_AZ}"]
managedNodeGroups:
  - name: ${NODEGROUP_NAME}
    launchTemplate:
      id: ${LT_ID}
    minSize: 0
    desiredCapacity: 0
    maxSize: 1
    availabilityZones: ["${SELECTED_AZ}"]
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
echo "  Scale back to 0 after benchmarking (~\$2.235/hr for trn2.3xlarge):"
echo "    aws eks update-nodegroup-config \\"
echo "      --cluster-name ${CLUSTER_NAME} \\"
echo "      --nodegroup-name ${NODEGROUP_NAME} \\"
echo "      --scaling-config minSize=0,maxSize=1,desiredSize=0 \\"
echo "      --region ${AWS_REGION}"
echo "══════════════════════════════════════════════════════"
