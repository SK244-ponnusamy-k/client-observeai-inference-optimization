#!/bin/bash
# ==============================================================================
# scripts/download.sh
#
# Generic model download — reads everything from models/<model-id>.yaml.
# Downloads from HuggingFace → S3 via a Kubernetes Job.
# No per-model scripts needed. No Python dependency on the local machine.
#
# Usage:
#   bash scripts/download.sh --model gpt-oss-20b
#   bash scripts/download.sh --model qwen-2.5-0.5b
#   bash scripts/download.sh --model qwen3-35b-nvfp4
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

# ── Pure-bash YAML field reader — no Python needed ────────────────────────────
yaml_field() {
    local file="$1" key="$2"
    grep -m1 "^${key}:" "${file}" | sed "s/^${key}:[[:space:]]*//" | tr -d "'\""
}
yaml_download() {
    local file="$1" key="$2"
    awk "/^download:/{found=1} found && /^  ${key}:/{gsub(/^  ${key}:[[:space:]]*/,\"\"); gsub(/['\"]*/,\"\"); print; exit}" "${file}"
}

# ── Parse args ────────────────────────────────────────────────────────────────
MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    log_error "Usage: bash scripts/download.sh --model <model-id>"
    log_error "Available models:"
    ls "${FRAMEWORK_ROOT}/models/"*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | sed 's/^/  /'
    exit 1
fi

# ── Resolve model file ────────────────────────────────────────────────────────
MODEL_FILE="${FRAMEWORK_ROOT}/models/${MODEL}.yaml"
if [[ ! -f "${MODEL_FILE}" ]]; then
    FOUND=""
    for f in "${FRAMEWORK_ROOT}/models/"*.yaml; do
        ALIAS=$(yaml_field "${f}" "model_alias")
        if [[ "${ALIAS}" == "${MODEL}" ]]; then FOUND="${f}"; break; fi
    done
    [[ -n "${FOUND}" ]] && MODEL_FILE="${FOUND}" || {
        log_error "No model file found for '${MODEL}'"
        exit 1
    }
fi

# ── Load model fields ─────────────────────────────────────────────────────────
MODEL_ID=$(yaml_field "${MODEL_FILE}" "model_id")
MODEL_HF_ID=$(yaml_field "${MODEL_FILE}" "hf_id")
MODEL_S3_FOLDER=$(yaml_field "${MODEL_FILE}" "s3_folder")
DOWNLOAD_SIZE_GB=$(yaml_download "${MODEL_FILE}" "size_gb")
DOWNLOAD_TMP_GI=$(yaml_download "${MODEL_FILE}" "tmp_size_gi")
IS_GATED=$(yaml_download "${MODEL_FILE}" "gated")

if [[ -z "${MODEL_ID}" ]]; then
    log_error "Could not read model_id from ${MODEL_FILE}"; exit 1
fi

JOB_NAME="oai-infopt-download-${MODEL_ID}"

echo ""
echo "══════════════════════════════════════════════════"
echo "  MODEL DOWNLOAD — ${MODEL_HF_ID}"
echo "  Bucket  : s3://${MODEL_BUCKET}/${MODEL_S3_FOLDER}/"
echo "  Size    : ~${DOWNLOAD_SIZE_GB} GB"
echo "  Gated   : ${IS_GATED}"
echo "══════════════════════════════════════════════════"
echo ""

# ── Skip if already in S3 ─────────────────────────────────────────────────────
log_info "Checking S3 for existing model files..."
EXISTING=$(aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_S3_FOLDER}/" \
    --region "${AWS_REGION}" 2>/dev/null | head -1 || true)
if [[ -n "${EXISTING}" ]]; then
    log_warn "Model already exists in S3 — skipping download."
    aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_S3_FOLDER}/" --human-readable --region "${AWS_REGION}"
    exit 0
fi

# ── Check HF token (gated models only) ───────────────────────────────────────
if [[ "${IS_GATED}" == "true" ]]; then
    log_info "Checking HuggingFace token secret in namespace ${BENCHMARK_NAMESPACE}..."
    if ! kubectl get secret hf-token -n "${BENCHMARK_NAMESPACE}" >/dev/null 2>&1; then
        log_error "HuggingFace token secret not found in '${BENCHMARK_NAMESPACE}'."
        log_error "  kubectl create secret generic hf-token --from-literal=token=hf_YOURTOKEN -n ${BENCHMARK_NAMESPACE}"
        exit 1
    fi
    log_info "HF token secret found."
fi

# ── Check ServiceAccount ──────────────────────────────────────────────────────
if ! kubectl get serviceaccount "${DOWNLOAD_SERVICE_ACCOUNT}" -n "${BENCHMARK_NAMESPACE}" >/dev/null 2>&1; then
    log_error "ServiceAccount '${DOWNLOAD_SERVICE_ACCOUNT}' not found. Run: bash cluster/bootstrap.sh"
    exit 1
fi

# ── Clean up any existing job ─────────────────────────────────────────────────
if kubectl get job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" >/dev/null 2>&1; then
    log_warn "Existing job '${JOB_NAME}' found — deleting..."
    kubectl delete job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}"
    sleep 3
fi

# ── HF token volume (gated models only) ──────────────────────────────────────
if [[ "${IS_GATED}" == "true" ]]; then
    HF_TOKEN_ENV="
            - name: HF_TOKEN_PATH
              value: \"/var/secrets/hf/token\""
    HF_TOKEN_VOLUME_MOUNT="
            - name: hf-token-vol
              mountPath: /var/secrets/hf
              readOnly: true"
    HF_TOKEN_VOLUME="
        - name: hf-token-vol
          secret:
            secretName: hf-token
            defaultMode: 0440"
    HF_TOKEN_READ="
              try:
                  with open('/var/secrets/hf/token') as f:
                      token = f.read().strip()
              except FileNotFoundError:
                  raise RuntimeError('HF token not found — check ExternalSecret sync')"
else
    HF_TOKEN_ENV=""
    HF_TOKEN_VOLUME_MOUNT=""
    HF_TOKEN_VOLUME=""
    HF_TOKEN_READ="              token = None"
fi

# ── Submit download Job ───────────────────────────────────────────────────────
log_info "Submitting download job: ${JOB_NAME}..."
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${BENCHMARK_NAMESPACE}
  labels:
    app.kubernetes.io/name: ${JOB_NAME}
    app.kubernetes.io/component: model-download
    app.kubernetes.io/managed-by: kubectl
    project: observeai-inference-optimization
    model: ${MODEL_ID}
spec:
  backoffLimit: 2
  activeDeadlineSeconds: 86400
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ${JOB_NAME}
        app.kubernetes.io/component: model-download
        project: observeai-inference-optimization
        model: ${MODEL_ID}
    spec:
      restartPolicy: Never
      serviceAccountName: ${DOWNLOAD_SERVICE_ACCOUNT}
      automountServiceAccountToken: false
      hostPID: false
      hostIPC: false
      hostNetwork: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 65534
        runAsGroup: 65534
        fsGroup: 65534
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: downloader
          image: python:3.11-slim@sha256:d1053354624536b044162aaab1e418bd000ea35184fb1ae098ab3166b1072e72
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -euo pipefail
              HF_ID="${MODEL_HF_ID}"
              S3_FOLDER="${MODEL_S3_FOLDER}"
              LOCAL_DIR="/tmp/\${S3_FOLDER}"

              echo "Model  : \${HF_ID}"
              echo "Bucket : s3://${MODEL_BUCKET}/\${S3_FOLDER}/"

              pip install --quiet --no-cache-dir --target /tmp/pip-packages \
                huggingface_hub==0.24.7 boto3==1.34.0
              export PYTHONPATH="/tmp/pip-packages:\${PYTHONPATH:-}"

              python3 -c "
              import os, sys
              sys.path.insert(0, '/tmp/pip-packages')
              from huggingface_hub import snapshot_download
              ${HF_TOKEN_READ}
              snapshot_download(
                  repo_id='\${HF_ID}',
                  local_dir='\${LOCAL_DIR}',
                  token=token,
                  ignore_patterns=['*.msgpack','*.h5','flax_*','tf_*','rust_*'],
              )
              print('Download complete.')
              "

              echo 'Uploading to S3...'
              python3 << 'PYEOF'
              import boto3, os, sys
              from pathlib import Path
              sys.path.insert(0, '/tmp/pip-packages')
              s3 = boto3.client('s3', region_name=os.environ['AWS_DEFAULT_REGION'])
              bucket = os.environ['MODEL_BUCKET']
              folder = os.environ['S3_FOLDER']
              local_dir = Path(f'/tmp/{folder}')
              files = [f for f in local_dir.rglob('*') if f.is_file() and '.cache' not in f.parts]
              print(f'Uploading {len(files)} files...')
              for i, f in enumerate(files, 1):
                  key = f'{folder}/' + str(f.relative_to(local_dir))
                  print(f'[{i}/{len(files)}] {f.name}')
                  s3.upload_file(str(f), bucket, key)
              print('Upload complete.')
              PYEOF
          env:
            - name: AWS_DEFAULT_REGION
              value: "${AWS_REGION}"
            - name: MODEL_BUCKET
              value: "${MODEL_BUCKET}"
            - name: S3_FOLDER
              value: "${MODEL_S3_FOLDER}"
            - name: HF_HUB_DISABLE_TELEMETRY
              value: "1"
            - name: HF_HUB_DISABLE_XET
              value: "1"
            - name: TRANSFORMERS_CACHE
              value: "/tmp/hf-cache"
            - name: HF_HOME
              value: "/tmp/hf-home"
            - name: PIP_NO_CACHE_DIR
              value: "1"${HF_TOKEN_ENV}
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            runAsUser: 65534
            runAsGroup: 65534
            capabilities:
              drop: ["ALL"]
            seccompProfile:
              type: RuntimeDefault
          resources:
            requests:
              cpu: "4"
              memory: "16Gi"
            limits:
              cpu: "8"
              memory: "32Gi"
          volumeMounts:
            - name: tmp-workspace
              mountPath: /tmp${HF_TOKEN_VOLUME_MOUNT}
      volumes:
        - name: tmp-workspace
          emptyDir:
            sizeLimit: ${DOWNLOAD_TMP_GI}Gi${HF_TOKEN_VOLUME}
EOF

log_info "Job '${JOB_NAME}' created. (~${DOWNLOAD_SIZE_GB} GB — may take 30-90 minutes)"

# ── Wait for pod ──────────────────────────────────────────────────────────────
log_info "Waiting for pod to start..."
for i in $(seq 1 36); do
    POD=$(kubectl get pods -l "job-name=${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [[ -n "${POD}" ]] && break
    echo -n "."; sleep 5
done
echo ""
[[ -z "${POD:-}" ]] && { log_error "Pod not found after 3 minutes."; exit 1; }
log_info "Pod: ${POD}"

# ── Stream logs ───────────────────────────────────────────────────────────────
log_info "Streaming logs (Ctrl+C to detach — job continues in cluster)..."
kubectl logs "${POD}" -n "${BENCHMARK_NAMESPACE}" -f 2>/dev/null || \
    kubectl logs "${POD}" -n "${BENCHMARK_NAMESPACE}" || true

# ── Wait for completion ───────────────────────────────────────────────────────
log_info "Waiting for job completion (timeout: 24h)..."
kubectl wait job "${JOB_NAME}" \
    --for=condition=complete \
    --timeout=86400s \
    -n "${BENCHMARK_NAMESPACE}" || {
    log_error "Job did not complete. Check: kubectl logs ${POD} -n ${BENCHMARK_NAMESPACE}"
    exit 1
}

log_info "Verifying S3 upload..."
aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_S3_FOLDER}/" --recursive --human-readable --region "${AWS_REGION}"
kubectl delete job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}"

echo ""
echo "══════════════════════════════════════════════════"
echo "  DOWNLOAD COMPLETE"
echo "  Model  : s3://${MODEL_BUCKET}/${MODEL_S3_FOLDER}/"
echo "  Next   : bash scripts/deploy.sh --model ${MODEL}"
echo "══════════════════════════════════════════════════"
