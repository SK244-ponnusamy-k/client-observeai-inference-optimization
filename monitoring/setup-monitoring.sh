#!/bin/bash
# ==============================================================================
# monitoring/setup-monitoring.sh
#
# Full monitoring stack setup — run once after cluster is ready.
#
# What this does:
#   1.  Validates tools + AWS credentials
#   2.  Creates Amazon Managed Prometheus (AMP) workspace
#   3.  Creates IAM policy for AMP read/write + CloudWatch
#   4.  Creates monitoring namespace + service accounts (Prometheus, Grafana, collector)
#   5.  Creates EKS Pod Identity associations for all three SAs
#   6.  Adds Helm repos (prometheus-community, gpu-helm-charts)
#   7.  Renders kube-prometheus-stack values (injects AMP endpoint + region)
#   8.  Installs kube-prometheus-stack (Prometheus + Grafana + node-exporter)
#   9.  Installs DCGM Exporter (GPU metrics DaemonSet)
#   10. Applies metrics-collector CronJob (pulls from AMP → writes report to S3)
#   11. Prints Grafana URL and admin password
#
# Called by: cluster/bootstrap.sh (Step 12) and cluster/setup-cluster.sh
# Can also be run standalone:
#   cd llm-inference-framework
#   bash monitoring/setup-monitoring.sh
#
# Prerequisites:
#   - EKS cluster running + kubeconfig configured
#   - AWS Load Balancer Controller installed (for Grafana ALB Ingress)
#   - config/config.env sourced
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

MONITORING_NAMESPACE="monitoring"
AMP_ALIAS="amp-ws-${CLUSTER_NAME}"
AMP_POLICY_NAME="${CLUSTER_NAME}-amp-grafana-policy"
AMP_INGEST_ROLE="${CLUSTER_NAME}-amp-ingest-role"
GRAFANA_ROLE="${CLUSTER_NAME}-grafana-role"
COLLECTOR_ROLE="${CLUSTER_NAME}-collector-role"
COLLECTOR_SA="metrics-collector-sa"

# ==============================================================================
# Step 1 — Validate tools
# ==============================================================================
log_step "Step 1 — Checking required tools"

for tool in aws kubectl helm; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        log_error "Missing required tool: ${tool}"
        exit 1
    fi
    log_info "${tool} — OK"
done

if ! aws sts get-caller-identity >/dev/null 2>&1; then
    log_error "AWS credentials not configured."
    exit 1
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
log_info "Account : ${AWS_ACCOUNT_ID}"
log_info "Region  : ${AWS_REGION}"
log_info "Cluster : ${CLUSTER_NAME}"

# Verify kubeconfig
if ! kubectl get nodes >/dev/null 2>&1; then
    log_warn "kubectl cannot reach cluster — updating kubeconfig..."
    aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region "${AWS_REGION}"
fi

# ==============================================================================
# Step 2 — Create AMP workspace
# ==============================================================================
log_step "Step 2 — Amazon Managed Prometheus workspace"

EXISTING_WS=$(aws amp list-workspaces \
    --alias "${AMP_ALIAS}" \
    --query 'workspaces[0].workspaceId' \
    --output text \
    --region "${AWS_REGION}" 2>/dev/null || echo "None")

if [[ "${EXISTING_WS}" == "None" || -z "${EXISTING_WS}" ]]; then
    log_info "Creating AMP workspace: ${AMP_ALIAS}..."
    aws amp create-workspace \
        --alias "${AMP_ALIAS}" \
        --region "${AWS_REGION}" \
        --tags "{\"Project\":\"${TAG_PROJECT}\",\"Environment\":\"${TAG_ENVIRONMENT}\",\"ManagedBy\":\"setup-monitoring\"}" \
        >/dev/null
    log_info "Waiting for workspace to become ACTIVE..."
    sleep 10
fi

AMP_WORKSPACE_ID=$(aws amp list-workspaces \
    --alias "${AMP_ALIAS}" \
    --query 'workspaces[0].workspaceId' \
    --output text \
    --region "${AWS_REGION}")

AMP_ENDPOINT=$(aws amp describe-workspace \
    --workspace-id "${AMP_WORKSPACE_ID}" \
    --query 'workspace.prometheusEndpoint' \
    --output text \
    --region "${AWS_REGION}")

log_info "Workspace ID : ${AMP_WORKSPACE_ID}"
log_info "Endpoint     : ${AMP_ENDPOINT}"

# ==============================================================================
# Step 3 — IAM policy for AMP + CloudWatch
# ==============================================================================
log_step "Step 3 — IAM policy: AMP + CloudWatch"

EXISTING_POLICY=$(aws iam list-policies \
    --query "Policies[?PolicyName=='${AMP_POLICY_NAME}'].Arn" \
    --output text 2>/dev/null || echo "")

if [[ -z "${EXISTING_POLICY}" ]]; then
    log_info "Creating IAM policy: ${AMP_POLICY_NAME}..."
    AMP_POLICY_ARN=$(aws iam create-policy \
        --policy-name "${AMP_POLICY_NAME}" \
        --policy-document "{
          \"Version\": \"2012-10-17\",
          \"Statement\": [
            {
              \"Sid\": \"AllowAMPReadWrite\",
              \"Effect\": \"Allow\",
              \"Action\": [
                \"aps:ListWorkspaces\",
                \"aps:DescribeWorkspace\",
                \"aps:GetMetricMetadata\",
                \"aps:GetSeries\",
                \"aps:QueryMetrics\",
                \"aps:RemoteWrite\",
                \"aps:GetLabels\"
              ],
              \"Resource\": \"arn:aws:aps:${AWS_REGION}:${AWS_ACCOUNT_ID}:workspace/*\"
            },
            {
              \"Sid\": \"AllowS3ResultsWrite\",
              \"Effect\": \"Allow\",
              \"Action\": [
                \"s3:PutObject\",
                \"s3:PutObjectTagging\",
                \"s3:GetObject\",
                \"s3:ListBucket\"
              ],
              \"Resource\": [
                \"arn:aws:s3:::${RESULTS_BUCKET}\",
                \"arn:aws:s3:::${RESULTS_BUCKET}/*\"
              ]
            },
            {
              \"Sid\": \"AllowCloudWatchMetrics\",
              \"Effect\": \"Allow\",
              \"Action\": [
                \"cloudwatch:DescribeAlarmsForMetric\",
                \"cloudwatch:ListMetrics\",
                \"cloudwatch:GetMetricData\",
                \"cloudwatch:GetMetricStatistics\"
              ],
              \"Resource\": \"*\"
            }
          ]
        }" \
        --query 'Policy.Arn' \
        --output text)
    log_info "Policy created: ${AMP_POLICY_ARN}"
else
    AMP_POLICY_ARN="${EXISTING_POLICY}"
    log_warn "Policy already exists: ${AMP_POLICY_ARN} — updating to ensure correct permissions..."
    # Create a new policy version with the correct permissions
    aws iam create-policy-version \
        --policy-arn "${AMP_POLICY_ARN}" \
        --policy-document "{
          \"Version\": \"2012-10-17\",
          \"Statement\": [
            {
              \"Sid\": \"AllowAMPReadWrite\",
              \"Effect\": \"Allow\",
              \"Action\": [
                \"aps:ListWorkspaces\",
                \"aps:DescribeWorkspace\",
                \"aps:GetMetricMetadata\",
                \"aps:GetSeries\",
                \"aps:QueryMetrics\",
                \"aps:RemoteWrite\",
                \"aps:GetLabels\"
              ],
              \"Resource\": \"arn:aws:aps:${AWS_REGION}:${AWS_ACCOUNT_ID}:workspace/*\"
            },
            {
              \"Sid\": \"AllowS3ResultsWrite\",
              \"Effect\": \"Allow\",
              \"Action\": [
                \"s3:PutObject\",
                \"s3:PutObjectTagging\",
                \"s3:GetObject\",
                \"s3:ListBucket\"
              ],
              \"Resource\": [
                \"arn:aws:s3:::${RESULTS_BUCKET}\",
                \"arn:aws:s3:::${RESULTS_BUCKET}/*\"
              ]
            },
            {
              \"Sid\": \"AllowCloudWatchMetrics\",
              \"Effect\": \"Allow\",
              \"Action\": [
                \"cloudwatch:DescribeAlarmsForMetric\",
                \"cloudwatch:ListMetrics\",
                \"cloudwatch:GetMetricData\",
                \"cloudwatch:GetMetricStatistics\"
              ],
              \"Resource\": \"*\"
            }
          ]
        }" \
        --set-as-default >/dev/null 2>&1 || log_warn "Policy version update failed — may have hit 5-version limit. Check AWS console."
    log_info "Policy updated: ${AMP_POLICY_ARN}"
fi

# ==============================================================================
# Step 4 — monitoring namespace + service accounts
# ==============================================================================
log_step "Step 4 — Namespace + ServiceAccounts"

kubectl get namespace "${MONITORING_NAMESPACE}" >/dev/null 2>&1 || \
    kubectl create namespace "${MONITORING_NAMESPACE}"
log_info "Namespace '${MONITORING_NAMESPACE}' ready."

for SA in amp-iamproxy-ingest-service-account grafana-sa "${COLLECTOR_SA}"; do
    if kubectl get serviceaccount "${SA}" -n "${MONITORING_NAMESPACE}" >/dev/null 2>&1; then
        log_warn "ServiceAccount '${SA}' already exists — skipping."
    else
        kubectl create serviceaccount "${SA}" -n "${MONITORING_NAMESPACE}"
        log_info "ServiceAccount created: ${SA}"
    fi
done

# ==============================================================================
# Step 5 — IAM roles + EKS Pod Identity associations
# ==============================================================================
log_step "Step 5 — Pod Identity associations"

TMPDIR_MON="${FRAMEWORK_ROOT}/.tmp-monitoring-$$"
mkdir -p "${TMPDIR_MON}"
trap 'rm -rf "${TMPDIR_MON}"' EXIT

# Trust policy for Pod Identity
cat > "${TMPDIR_MON}/trust.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "pods.eks.amazonaws.com" },
    "Action": ["sts:AssumeRole", "sts:TagSession"]
  }]
}
EOF

# Helper: create role if not exists, attach AMP policy, create Pod Identity association
setup_monitoring_role() {
    local SA_NAME="$1"
    local ROLE_NAME="$2"

    # Create IAM role if not exists
    if aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
        log_warn "IAM role '${ROLE_NAME}' already exists — updating trust policy."
        aws iam update-assume-role-policy \
            --role-name "${ROLE_NAME}" \
            --policy-document "$(cat "${TMPDIR_MON}/trust.json")"
    else
        log_info "Creating IAM role: ${ROLE_NAME}..."
        aws iam create-role \
            --role-name "${ROLE_NAME}" \
            --assume-role-policy-document "$(cat "${TMPDIR_MON}/trust.json")" \
            --description "oai-infopt monitoring: ${SA_NAME} AMP access" \
            >/dev/null
        log_info "Role created: ${ROLE_NAME}"
    fi

    # Attach AMP policy — idempotent
    ATTACHED=$(aws iam list-attached-role-policies \
        --role-name "${ROLE_NAME}" \
        --query "AttachedPolicies[?PolicyArn=='${AMP_POLICY_ARN}'].PolicyArn" \
        --output text 2>/dev/null || echo "")

    if [[ -z "${ATTACHED}" ]]; then
        log_info "Attaching AMP policy to ${ROLE_NAME}..."
        aws iam attach-role-policy \
            --role-name "${ROLE_NAME}" \
            --policy-arn "${AMP_POLICY_ARN}"
        log_info "Policy attached."
    else
        log_warn "AMP policy already attached to ${ROLE_NAME} — skipping."
    fi

    # Create Pod Identity association if not exists
    local ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"
    EXISTING_ASSOC=$(aws eks list-pod-identity-associations \
        --cluster-name "${CLUSTER_NAME}" \
        --namespace "${MONITORING_NAMESPACE}" \
        --service-account "${SA_NAME}" \
        --region "${AWS_REGION}" \
        --query "associations[0].associationId" \
        --output text 2>/dev/null || echo "None")

    if [[ "${EXISTING_ASSOC}" != "None" && "${EXISTING_ASSOC}" != "null" && -n "${EXISTING_ASSOC}" ]]; then
        log_warn "Pod Identity association for '${SA_NAME}' already exists — skipping."
    else
        log_info "Creating Pod Identity association: ${SA_NAME} -> ${ROLE_NAME}..."
        aws eks create-pod-identity-association \
            --cluster-name "${CLUSTER_NAME}" \
            --namespace "${MONITORING_NAMESPACE}" \
            --service-account "${SA_NAME}" \
            --role-arn "${ROLE_ARN}" \
            --region "${AWS_REGION}" \
            >/dev/null
        log_info "Association created: ${SA_NAME} -> ${ROLE_ARN}"
    fi
}

setup_monitoring_role "amp-iamproxy-ingest-service-account" "${AMP_INGEST_ROLE}"
setup_monitoring_role "grafana-sa"                          "${GRAFANA_ROLE}"
setup_monitoring_role "${COLLECTOR_SA}"                     "${COLLECTOR_ROLE}"

# ==============================================================================
# Step 6 — Helm repos
# ==============================================================================
log_step "Step 6 — Helm repos"

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts 2>/dev/null || true
helm repo update
log_info "Helm repos updated."

# ==============================================================================
# Step 7 — Render kube-prometheus-stack values
# ==============================================================================
log_step "Step 7 — Rendering kube-prometheus-stack values"

# Get caller IP for Grafana ALB CIDR restriction
MY_CIDR=$(curl -s --max-time 5 https://checkip.amazonaws.com 2>/dev/null || echo "0.0.0.0")/32
log_info "Grafana ALB restricted to: ${MY_CIDR}"

RENDERED_VALUES="/tmp/kube-prometheus-values-${CLUSTER_NAME}.yaml"

# Use envsubst to inject live values into the template
export AMP_ENDPOINT AWS_REGION MY_CIDR CLUSTER_NAME MONITORING_NAMESPACE
envsubst < "${SCRIPT_DIR}/values/kube-prometheus-values.yaml.tmpl" > "${RENDERED_VALUES}"

# Validate critical values were injected
if ! grep -q "aps-workspaces" "${RENDERED_VALUES}"; then
    log_error "AMP endpoint not injected into values file. Check AMP workspace."
    exit 1
fi
log_info "Values rendered to: ${RENDERED_VALUES}"

# Create/update ConfigMap with local vLLM+GPU dashboard JSON
log_info "Creating grafana-dashboards-vllm ConfigMap..."
kubectl create configmap grafana-dashboards-vllm \
    --from-file=vllm-overview.json="${SCRIPT_DIR}/dashboards/vllm-overview.json" \
    --namespace "${MONITORING_NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -
kubectl label configmap grafana-dashboards-vllm \
    --namespace "${MONITORING_NAMESPACE}" \
    grafana_dashboard="1" \
    --overwrite
log_info "Dashboard ConfigMap applied."

# ==============================================================================
# Step 8 — Install kube-prometheus-stack
# ==============================================================================
log_step "Step 8 — kube-prometheus-stack"

if helm status kube-prometheus-stack -n "${MONITORING_NAMESPACE}" >/dev/null 2>&1; then
    log_warn "kube-prometheus-stack already installed — upgrading..."
    helm upgrade kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace "${MONITORING_NAMESPACE}" \
        -f "${RENDERED_VALUES}" \
        --wait --timeout 10m
else
    log_info "Installing kube-prometheus-stack..."
    helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace "${MONITORING_NAMESPACE}" \
        -f "${RENDERED_VALUES}" \
        --wait --timeout 10m
fi
log_info "kube-prometheus-stack ready."

# ==============================================================================
# Step 9 — Install DCGM Exporter
# ==============================================================================
log_step "Step 9 — NVIDIA DCGM Exporter"

if helm status dcgm-exporter -n "${MONITORING_NAMESPACE}" >/dev/null 2>&1; then
    log_warn "dcgm-exporter already installed — upgrading..."
    helm upgrade dcgm-exporter gpu-helm-charts/dcgm-exporter \
        --namespace "${MONITORING_NAMESPACE}" \
        -f "${SCRIPT_DIR}/values/dcgm-exporter-values.yaml" \
        --wait --timeout 5m
else
    log_info "Installing DCGM exporter..."
    helm install dcgm-exporter gpu-helm-charts/dcgm-exporter \
        --namespace "${MONITORING_NAMESPACE}" \
        -f "${SCRIPT_DIR}/values/dcgm-exporter-values.yaml" \
        --wait --timeout 5m
fi
log_info "DCGM exporter ready (runs on GPU nodes when provisioned)."

# ==============================================================================
# Step 10 — Apply metrics-collector ConfigMap + CronJob
# ==============================================================================
log_step "Step 10 — Metrics collector CronJob"

# Create/update ConfigMap with collector.py source
log_info "Creating metrics-collector-script ConfigMap..."
kubectl create configmap metrics-collector-script \
    --from-file=collector.py="${SCRIPT_DIR}/metrics-collector/collector.py" \
    --namespace "${MONITORING_NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -
log_info "ConfigMap applied."

# Inject runtime values into the CronJob manifest and apply
log_info "Applying CronJob..."
export AMP_WORKSPACE_ID AMP_ENDPOINT RESULTS_BUCKET AWS_REGION MONITORING_NAMESPACE COLLECTOR_SA CLUSTER_NAME
envsubst < "${SCRIPT_DIR}/metrics-collector/collector-cronjob.yaml" | kubectl apply -f -
log_info "Metrics collector CronJob applied (runs every 30 min)."

# ==============================================================================
# Step 11 — Print access info
# ==============================================================================
log_step "Step 11 — Access information"

echo ""
echo "  Waiting for Grafana ALB to provision (~60s)..."
sleep 30

GRAFANA_URL=$(kubectl get ingress kube-prometheus-stack-grafana \
    -n "${MONITORING_NAMESPACE}" \
    -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "pending")

GRAFANA_PASS=$(kubectl get secrets kube-prometheus-stack-grafana \
    -n "${MONITORING_NAMESPACE}" \
    -o jsonpath="{.data.admin-password}" 2>/dev/null | base64 -d 2>/dev/null || echo "not-ready")

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  MONITORING STACK READY"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  AMP Workspace    : ${AMP_WORKSPACE_ID}"
echo "  AMP Endpoint     : ${AMP_ENDPOINT}"
echo ""
echo "  Grafana URL      : http://${GRAFANA_URL}"
echo "  Grafana User     : admin"
echo "  Grafana Password : ${GRAFANA_PASS}"
echo "  Grafana Restricted to: ${MY_CIDR}"
echo ""
echo "  Dashboards pre-loaded:"
echo "    - GPU Monitoring > NVIDIA DCGM Exporter Dashboard"
echo "    - GPU Monitoring > vLLM Dashboard"
echo "    - GPU Monitoring > vLLM Load Analysis"
echo ""
echo "  Metrics reports  : s3://${RESULTS_BUCKET}/monitoring/reports/"
echo "  Collector runs   : every 30 minutes (CronJob in ${MONITORING_NAMESPACE})"
echo ""
echo "  Verify pods:"
echo "    kubectl get pods -n ${MONITORING_NAMESPACE}"
echo "══════════════════════════════════════════════════════════════"
