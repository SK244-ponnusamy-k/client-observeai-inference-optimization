#!/bin/bash
# ==============================================================================
# inference/run-quality.sh
#
# Runs the AutoQA QUALITY eval (accuracy / precision / recall / F1) against a
# DEPLOYED vLLM endpoint, in-cluster, and uploads the quality result to S3.
#
# This is the QUALITY stage — separate from run-benchmark.sh (performance).
# Accuracy is hardware-independent, so run it ONCE per (model, quantization);
# --hw only selects which running service to hit.
#
# The labelled dataset (customer-shared synthetic AutoQA) is staged to S3 and
# pulled into the pod — it is NEVER baked into a ConfigMap or committed to git.
#
# Usage:
#   # one-time: stage the dataset to S3 (encrypted bucket)
#   bash inference/run-quality.sh --stage-dataset "/path/to/synthetic_autoqa_transcripts.csv"
#
#   # then evaluate a deployed model
#   bash inference/run-quality.sh --model gpt-oss-20b --hw g6e --quant mxfp4
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

: "${RESULTS_BUCKET:?ERROR: RESULTS_BUCKET not set in config/config.env.}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

MODEL="gpt-oss-20b"
HW="g6e"
QUANT="mxfp4"
CONFIG="configs/quality/autoqa_v1.yaml"
DATASET_S3_KEY="quality-datasets/autoqa_v1.csv"
STAGE_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2"; shift 2 ;;
        --hw)      HW="$2"; shift 2 ;;
        --quant)   QUANT="$2"; shift 2 ;;
        --config)  CONFIG="$2"; shift 2 ;;
        --stage-dataset) STAGE_FILE="$2"; shift 2 ;;
        *) log_error "Unknown: $1"; exit 1 ;;
    esac
done

# ── One-time: stage the dataset to S3 (kept out of git/configmaps) ───────────
if [[ -n "${STAGE_FILE}" ]]; then
    [[ -f "${STAGE_FILE}" ]] || { log_error "File not found: ${STAGE_FILE}"; exit 1; }
    log_info "Staging dataset → s3://${RESULTS_BUCKET}/${DATASET_S3_KEY} (SSE)..."
    aws s3 cp "${STAGE_FILE}" "s3://${RESULTS_BUCKET}/${DATASET_S3_KEY}" \
        --sse aws:kms --region "${AWS_REGION}"
    log_info "Staged. Now run: bash inference/run-quality.sh --model ${MODEL} --hw ${HW} --quant ${QUANT}"
    exit 0
fi

# Resolve service name from model (accuracy is hw-independent; --hw picks the live svc)
case "${MODEL}" in
    gpt-oss-20b)                         SVC="oai-infopt-vllm-gpt-oss-20b" ;;
    qwen3.5-4b|qwen3-5-4b)               SVC="oai-infopt-vllm-qwen3-5-4b" ;;
    gemma-4-26b-a4b|gemma)               SVC="oai-infopt-vllm-gemma-4-26b-a4b" ;;
    qwen-0.5b|qwen-0-5b|qwen)            SVC="oai-infopt-vllm-qwen-0-5b" ;;
    *) log_error "Unknown model: ${MODEL}"; exit 1 ;;
esac

[[ -f "${FRAMEWORK_ROOT}/${CONFIG}" ]] || { log_error "Config not found: ${CONFIG}"; exit 1; }

ENDPOINT="http://${SVC}:8000"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="oai-infopt-quality-${MODEL//\./-}-${TIMESTAMP}"

echo ""
echo "======================================================"
echo "  QUALITY EVAL (AutoQA F1)"
echo "  Model    : ${MODEL}  [${QUANT}]"
echo "  Endpoint : ${ENDPOINT}"
echo "  Config   : ${CONFIG}"
echo "  Dataset  : s3://${RESULTS_BUCKET}/${DATASET_S3_KEY}"
echo "  Results  : s3://${RESULTS_BUCKET}/quality/${TIMESTAMP}/"
echo "======================================================"
echo ""

cd "${FRAMEWORK_ROOT}"

log_info "Creating ConfigMaps (script + config)..."
kubectl create configmap oai-infopt-quality-script \
    --from-file=quality-eval.py="${FRAMEWORK_ROOT}/inference/quality-eval.py" \
    -n "${BENCHMARK_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl create configmap oai-infopt-quality-config \
    --from-file=config.yaml="${FRAMEWORK_ROOT}/${CONFIG}" \
    -n "${BENCHMARK_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
log_info "ConfigMaps ready."

log_info "Submitting quality Job: ${JOB_NAME}..."
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${BENCHMARK_NAMESPACE}
  labels:
    app.kubernetes.io/component: quality-runner
    project: observeai-inference-optimization
    model: "${MODEL}"
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 10800
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ${JOB_NAME}
        app.kubernetes.io/component: quality-runner
        project: observeai-inference-optimization
        model: "${MODEL}"
    spec:
      restartPolicy: Never
      serviceAccountName: oai-infopt-benchmark-sa
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
        - name: quality
          image: python:3.11-slim@sha256:d1053354624536b044162aaab1e418bd000ea35184fb1ae098ab3166b1072e72
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -euo pipefail
              export HOME=/tmp
              export PYTHONPATH="/tmp/pip-packages:\${PYTHONPATH:-}"
              pip install --quiet --no-cache-dir --target=/tmp/pip-packages boto3==1.34.0 pyyaml==6.0.1
              echo "Downloading dataset from s3://${RESULTS_BUCKET}/${DATASET_S3_KEY} ..."
              python3 -c "
              import boto3
              boto3.client('s3', region_name='${AWS_REGION}').download_file(
                  '${RESULTS_BUCKET}', '${DATASET_S3_KEY}', '/tmp/autoqa.csv')
              print('dataset ready')
              "
              TEST_EXIT=0
              python3 /app/quality-eval.py \
                --config   /configs/quality/config.yaml \
                --dataset  /tmp/autoqa.csv \
                --endpoint "${ENDPOINT}" \
                --model    "${MODEL}" \
                --quantization "${QUANT}" \
                --output   /results \
                --wait-timeout 600 || TEST_EXIT=\$?
              echo "=== Uploading quality results to S3 ==="
              python3 -c "
              import boto3, os, glob
              s3 = boto3.client('s3', region_name='${AWS_REGION}')
              for f in glob.glob('/results/*.jsonl'):
                  key = 'quality/${TIMESTAMP}/' + os.path.basename(f)
                  s3.upload_file(f, '${RESULTS_BUCKET}', key); print('Uploaded', key)
              " || true
              exit \${TEST_EXIT}
          env:
            - name: AWS_DEFAULT_REGION
              value: "${AWS_REGION}"
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
              cpu: "1"
              memory: "2Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          volumeMounts:
            - name: app-code
              mountPath: /app
              readOnly: true
            - name: config
              mountPath: /configs/quality
              readOnly: true
            - name: results
              mountPath: /results
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: app-code
          configMap:
            name: oai-infopt-quality-script
        - name: config
          configMap:
            name: oai-infopt-quality-config
        - name: results
          emptyDir:
            sizeLimit: 256Mi
        - name: tmp
          emptyDir:
            sizeLimit: 2Gi
EOF

log_info "Job submitted."

log_info "Waiting for pod..."
for i in $(seq 1 30); do
    POD=$(kubectl get pods -l "job-name=${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    [[ -n "${POD}" ]] && break; echo -n "."; sleep 5
done
echo ""
log_info "Pod: ${POD:-not found}"
[[ -n "${POD:-}" ]] && kubectl logs "${POD}" -n "${BENCHMARK_NAMESPACE}" -f 2>/dev/null || true

echo ""
log_info "Quality results:"
aws s3 ls "s3://${RESULTS_BUCKET}/quality/${TIMESTAMP}/" \
    --recursive --human-readable --region "${AWS_REGION}" 2>/dev/null || \
    log_warn "No results in S3 yet — may still be uploading"
