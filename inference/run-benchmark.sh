#!/bin/bash
# ==============================================================================
# inference/run-benchmark.sh
#
# One command to run a benchmark Job inside the cluster and upload results to S3.
#
# Usage:
#   cd llm-inference-framework
#   bash inference/run-benchmark.sh                          # gpt-oss-20b realtime (default)
#   bash inference/run-benchmark.sh --model qwen-0.5b
#   bash inference/run-benchmark.sh --model gpt-oss-20b --profile batch
#   bash inference/run-benchmark.sh --model gpt-oss-20b --profile realtime
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

# Defaults
MODEL="gpt-oss-20b"
PROFILE="realtime"
DATASET_XLSX=""   # optional override; defaults to qa_eval_v1.xlsx

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2";   shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --dataset) DATASET_XLSX="$2"; shift 2 ;;
        *) log_error "Unknown: $1"; exit 1 ;;
    esac
done

# Resolve manifest and endpoint from model name
case "${MODEL}" in
    gpt-oss-20b)
        MANIFEST="configs/manifests/gpt-oss-20b-baseline.yaml"
        SVC="oai-infopt-vllm-gpt-oss-20b"
        ;;
    qwen-0.5b|qwen-0-5b|qwen)
        MANIFEST="configs/manifests/qwen-2.5-0.5b-baseline.yaml"
        SVC="oai-infopt-vllm-qwen-0-5b"
        ;;
    *)
        log_error "Unknown model: ${MODEL}. Use gpt-oss-20b or qwen-0.5b"
        exit 1
        ;;
esac

PROFILE_FILE="configs/workload_profiles/${PROFILE}_v1.yaml"
ENDPOINT="http://${SVC}:8000"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="oai-infopt-bench-${MODEL//\./-}-${PROFILE}-${TIMESTAMP}"

echo ""
echo "======================================================"
echo "  BENCHMARK JOB"
echo "  Model   : ${MODEL}"
echo "  Profile : ${PROFILE}"
echo "  Endpoint: ${ENDPOINT}"
echo "  Job     : ${JOB_NAME}"
echo "  Results : s3://${RESULTS_BUCKET}/results/${TIMESTAMP}/"
echo "======================================================"
echo ""

cd "${FRAMEWORK_ROOT}"

# ==============================================================================
# Step 1 — Create / update ConfigMaps
# ==============================================================================
log_info "Creating ConfigMaps..."

kubectl create configmap oai-infopt-benchmark-script \
    --from-file=load-test.py="${FRAMEWORK_ROOT}/inference/load-test.py" \
    -n "${BENCHMARK_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null

kubectl create configmap oai-infopt-benchmark-manifest \
    --from-file=manifest.yaml="${FRAMEWORK_ROOT}/${MANIFEST}" \
    -n "${BENCHMARK_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null

kubectl create configmap "oai-infopt-benchmark-profile-${PROFILE}" \
    --from-file=profile.yaml="${FRAMEWORK_ROOT}/${PROFILE_FILE}" \
    -n "${BENCHMARK_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null

# Dataset paths — use --dataset override if provided, else default
if [[ -n "${DATASET_XLSX}" ]]; then
    # User passed a custom dataset path
    DATASET_JSONL="${DATASET_XLSX%.xlsx}.jsonl"
    DATASET_S3_KEY="datasets/$(basename "${DATASET_JSONL}")"
else
    DATASET_XLSX="${FRAMEWORK_ROOT}/configs/workload_profiles/datasets/qa_eval_v1.xlsx"
    DATASET_JSONL="${FRAMEWORK_ROOT}/configs/workload_profiles/datasets/qa_eval_v1.jsonl"
    DATASET_S3_KEY="datasets/qa_eval_v1.jsonl"
fi

if [[ ! -f "${DATASET_JSONL}" && -f "${DATASET_XLSX}" ]]; then
    log_info "Exporting dataset from Excel..."
    # cygpath converts /c/Users/... → C:\Users\... for Windows Python
    WIN_XLSX=$(cygpath -w "${DATASET_XLSX}" 2>/dev/null || echo "${DATASET_XLSX}")
    python "${FRAMEWORK_ROOT}/inference/validate_dataset.py" "${WIN_XLSX}" --export-jsonl
fi
if [[ -f "${DATASET_JSONL}" ]]; then
    # Pure bash file size — avoids Git Bash MINGW path issues with python -c
    DATASET_SIZE_BYTES=$(wc -c < "${DATASET_JSONL}")
    DATASET_SIZE_MB=$(awk "BEGIN {printf \"%.1f\", ${DATASET_SIZE_BYTES}/1024/1024}")
    # Only upload if file doesn't exist in S3 or checksum differs
    LOCAL_MD5=$(md5sum "${DATASET_JSONL}" | awk '{print $1}')
    REMOTE_ETAG=$(aws s3api head-object \
        --bucket "${RESULTS_BUCKET}" \
        --key "${DATASET_S3_KEY}" \
        --region "${AWS_REGION}" \
        --query 'ETag' --output text 2>/dev/null | tr -d '"' || echo "")
    if [[ "${LOCAL_MD5}" == "${REMOTE_ETAG}" ]]; then
        log_info "Dataset already up to date in S3 — skipping upload."
    else
        log_info "Uploading dataset (${DATASET_SIZE_MB} MB) to s3://${RESULTS_BUCKET}/${DATASET_S3_KEY} ..."
        aws s3 cp "${DATASET_JSONL}" "s3://${RESULTS_BUCKET}/${DATASET_S3_KEY}" \
            --region "${AWS_REGION}"
        log_info "Dataset uploaded."
    fi
else
    log_warn "Dataset JSONL not found — benchmark will use built-in fallback prompts."
    DATASET_S3_KEY=""
fi

log_info "ConfigMaps ready."

# ==============================================================================
# Step 2 — Submit Job
# ==============================================================================
log_info "Submitting benchmark Job: ${JOB_NAME}..."

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${BENCHMARK_NAMESPACE}
  labels:
    app.kubernetes.io/component: benchmark-runner
    project: observeai-inference-optimization
    model: "${MODEL}"
    profile: "${PROFILE}"
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 7200
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ${JOB_NAME}
        app.kubernetes.io/component: benchmark-runner
        project: observeai-inference-optimization
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
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchLabels:
                  app.kubernetes.io/component: inference-server
              topologyKey: kubernetes.io/hostname
      tolerations: []
      containers:
        - name: benchmark
          image: python:3.11-slim
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -euo pipefail
              export HOME=/tmp
              pip install --quiet --no-cache-dir --target=/tmp/pip-packages \
                "openai==1.57.0" \
                pyyaml==6.0.1 \
                boto3==1.34.0 \
                "aiohttp==3.10.0" \
                openpyxl==3.1.2
              export PYTHONPATH=/tmp/pip-packages

              # Download dataset from S3 if a key was provided
              DATASET_S3_KEY="${DATASET_S3_KEY}"
              if [[ -n "\${DATASET_S3_KEY}" ]]; then
                  echo "Downloading dataset from s3://${RESULTS_BUCKET}/\${DATASET_S3_KEY} ..."
                  mkdir -p /tmp/datasets
                  python -c "
              import boto3, sys
              sys.path.insert(0, '/tmp/pip-packages')
              boto3.client('s3', region_name='${AWS_REGION}').download_file(
                  '${RESULTS_BUCKET}', '\${DATASET_S3_KEY}', '/tmp/datasets/qa_eval_v1.jsonl')
              print('Dataset downloaded: /tmp/datasets/qa_eval_v1.jsonl')
              "
              fi
              TEST_EXIT=0
              python /app/load-test.py \
                --manifest /configs/manifests/manifest.yaml \
                --profile  /configs/profiles/profile.yaml \
                --endpoint "${ENDPOINT}" \
                --output   /results \
                --wait-timeout 300 || TEST_EXIT=\$?
              echo "=== Uploading results to S3 ==="
              python -c "
              import boto3, os, glob, sys
              sys.path.insert(0, '/tmp/pip-packages')
              s3 = boto3.client('s3', region_name='${AWS_REGION}')
              for f in glob.glob('/results/*.jsonl'):
                  key = 'results/${TIMESTAMP}/${PROFILE}/' + os.path.basename(f)
                  s3.upload_file(f, '${RESULTS_BUCKET}', key)
                  print('Uploaded: s3://${RESULTS_BUCKET}/' + key)
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
            - name: manifest
              mountPath: /configs/manifests
              readOnly: true
            - name: profile
              mountPath: /configs/profiles
              readOnly: true
            - name: results
              mountPath: /results
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: app-code
          configMap:
            name: oai-infopt-benchmark-script
        - name: manifest
          configMap:
            name: oai-infopt-benchmark-manifest
        - name: profile
          configMap:
            name: oai-infopt-benchmark-profile-${PROFILE}
        - name: results
          emptyDir:
            sizeLimit: 1Gi
        - name: tmp
          emptyDir:
            sizeLimit: 512Mi
EOF

log_info "Job submitted."

# ==============================================================================
# Step 3 — Wait for pod and stream logs
# ==============================================================================
log_info "Waiting for pod to start..."
for i in $(seq 1 30); do
    POD=$(kubectl get pods -l "job-name=${JOB_NAME}" \
        -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    [[ -n "${POD}" ]] && break
    echo -n "."
    sleep 5
done
echo ""
log_info "Pod: ${POD:-not found}"

if [[ -n "${POD:-}" ]]; then
    log_info "Streaming logs (Ctrl+C detaches — job keeps running in cluster)..."
    echo ""
    kubectl logs "${POD}" -n "${BENCHMARK_NAMESPACE}" -f 2>/dev/null || true
fi

# ==============================================================================
# Step 4 — Wait for completion
# ==============================================================================
log_info "Waiting for job completion..."
kubectl wait job "${JOB_NAME}" \
    --for=condition=complete \
    --timeout=7200s \
    -n "${BENCHMARK_NAMESPACE}" && STATUS="PASSED" || STATUS="FAILED"

echo ""
echo "======================================================"
echo "  BENCHMARK ${STATUS}"
echo "  Results : s3://${RESULTS_BUCKET}/results/${TIMESTAMP}/${PROFILE}/"
echo "======================================================"

# ==============================================================================
# Step 5 — Show S3 results
# ==============================================================================
log_info "S3 results:"
aws s3 ls "s3://${RESULTS_BUCKET}/results/${TIMESTAMP}/" \
    --recursive --human-readable --region "${AWS_REGION}" 2>/dev/null || \
    log_warn "No results found in S3 yet — may still be uploading"

echo ""
echo "  Grafana  : kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring"
echo "  Download : aws s3 cp s3://${RESULTS_BUCKET}/results/${TIMESTAMP}/ results/ --recursive --region ${AWS_REGION}"
echo ""

[[ "${STATUS}" == "PASSED" ]] || exit 1
