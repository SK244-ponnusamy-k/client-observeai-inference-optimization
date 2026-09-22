#!/bin/bash
# ==============================================================================
# inference/run-quality.sh
#
# Submit the AutoQA QUALITY evaluation (accuracy / precision / recall / F1)
# against a deployed vLLM endpoint and upload results to S3.
#
# Product entry point:
#   oai quality <id> [--hw ...] [--tag ...] [--dump-samples]
#
# Direct usage:
#   bash inference/run-quality.sh --model gpt-oss-20b \
#     --served-name gpt-oss-20b --hw g6e.2xlarge --quant mxfp4 \
#     --svc oai-infopt-vllm-gpt-oss-20b-g6e --tag g6e
#
# A deploy pipeline may pass --wait-for-marker. The Job is still submitted
# immediately, but its init container waits for that S3 completion marker before
# evaluating, which guarantees realtime -> batch -> quality ordering in-cluster.
# Closing the terminal does not stop an already-submitted Kubernetes Job.
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

MODEL="gpt-oss-20b"                    # framework/catalog ID
SERVED_NAME=""                         # exact vLLM --served-model-name
HW="unknown"                           # result metadata (prefer full instance type)
QUANT="unknown"
CONFIG="configs/quality/autoqa_v1.yaml"
DATASET_S3_KEY="quality-datasets/autoqa_v1.csv"
STAGE_FILE=""
SVC_OVERRIDE=""
TAG=""
RUN_TIMESTAMP=""
WAIT_FOR_MARKER=""                     # exact key in RESULTS_BUCKET; empty = start now
DUMP_SAMPLES="false"
MAX_DUMP_SAMPLES=200
DUMP_INPUTS="false"                    # include input transcripts in the samples file
DETACH="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)            MODEL="$2"; shift 2 ;;
        --served-name)      SERVED_NAME="$2"; shift 2 ;;
        --hw)               HW="$2"; shift 2 ;;
        --quant)            QUANT="$2"; shift 2 ;;
        --config)           CONFIG="$2"; shift 2 ;;
        --dataset-key)      DATASET_S3_KEY="$2"; shift 2 ;;
        --svc)              SVC_OVERRIDE="$2"; shift 2 ;;
        --tag)              TAG="$2"; shift 2 ;;
        --run-timestamp)    RUN_TIMESTAMP="$2"; shift 2 ;;
        --wait-for-marker)  WAIT_FOR_MARKER="$2"; shift 2 ;;
        --dump-samples)     DUMP_SAMPLES="true"; shift ;;
        --max-dump-samples) MAX_DUMP_SAMPLES="$2"; shift 2 ;;
        --dump-inputs)      DUMP_INPUTS="true"; shift ;;
        --detach)           DETACH="true"; shift ;;
        --stage-dataset)    STAGE_FILE="$2"; shift 2 ;;
        *) log_error "Unknown: $1"; exit 1 ;;
    esac
done

[[ "${MAX_DUMP_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
    log_error "--max-dump-samples must be a positive integer (got '${MAX_DUMP_SAMPLES}')."; exit 1;
}

# One-time dataset staging. This path intentionally remains independent of model,
# tag, and run timestamp because all quality jobs consume the same frozen dataset.
if [[ -n "${STAGE_FILE}" ]]; then
    [[ -f "${STAGE_FILE}" ]] || { log_error "File not found: ${STAGE_FILE}"; exit 1; }
    log_info "Staging dataset -> s3://${RESULTS_BUCKET}/${DATASET_S3_KEY} (SSE-KMS)..."
    aws s3 cp "${STAGE_FILE}" "s3://${RESULTS_BUCKET}/${DATASET_S3_KEY}" \
        --sse aws:kms --region "${AWS_REGION}"
    log_info "Dataset staged successfully."
    exit 0
fi

# Generic convention works for newly onboarded models. Known aliases are kept for
# backward-compatible direct usage; --svc from the oai wrapper always wins.
case "${MODEL}" in
    gpt-oss-20b)                         SVC="oai-infopt-vllm-gpt-oss-20b" ;;
    qwen3.5-4b|qwen3-5-4b)               SVC="oai-infopt-vllm-qwen3-5-4b" ;;
    gemma-4-31b|gemma-4-31b-it|gemma)    SVC="oai-infopt-vllm-gemma-4-31b" ;;
    qwen-0.5b|qwen-0-5b|qwen)            SVC="oai-infopt-vllm-qwen-0-5b" ;;
    *)                                    SVC="oai-infopt-vllm-${MODEL}" ;;
esac
[[ -n "${SVC_OVERRIDE}" ]] && SVC="${SVC_OVERRIDE}"
[[ -n "${SERVED_NAME}" ]] || SERVED_NAME="${MODEL}"

[[ -f "${FRAMEWORK_ROOT}/${CONFIG}" ]] || { log_error "Config not found: ${CONFIG}"; exit 1; }

TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
TAG_SEG=""
S3_TAG_SEG=""
if [[ -n "${TAG}" ]]; then
    TAG_SEG="-${TAG}"
    S3_TAG_SEG="${TAG}/"
fi

# Keep Kubernetes names <=63 chars while preserving a stable suffix for each
# resource. Current built-in IDs fit without truncation; this also supports longer
# future catalog IDs/tags.
k8s_name() {
    local raw="$1"
    if [[ ${#raw} -le 63 ]]; then
        printf '%s' "${raw}"
        return
    fi
    local digest prefix
    digest=$(printf '%s' "${raw}" | cksum | awk '{print $1}' | cut -c1-8)
    prefix="${raw:0:54}"
    prefix="${prefix%-}"
    printf '%s-%s' "${prefix}" "${digest}"
}

JOB_NAME=$(k8s_name "oai-infopt-quality-${MODEL//./-}${TAG_SEG}-${TIMESTAMP}")
SCRIPT_CM=$(k8s_name "${JOB_NAME}-script")
CONFIG_CM=$(k8s_name "${JOB_NAME}-config")
ENDPOINT="http://${SVC}:8000"
QUALITY_S3_PREFIX="quality/${TIMESTAMP}/${S3_TAG_SEG}"

QUALITY_EXTRA_ARGS=""
if [[ "${DUMP_SAMPLES}" == "true" ]]; then
    QUALITY_EXTRA_ARGS="--dump-samples --max-dump-samples ${MAX_DUMP_SAMPLES}"
    if [[ "${DUMP_INPUTS}" == "true" ]]; then
        QUALITY_EXTRA_ARGS="${QUALITY_EXTRA_ARGS} --dump-inputs"
    fi
elif [[ "${DUMP_INPUTS}" == "true" ]]; then
    log_error "--dump-inputs requires --dump-samples."; exit 1
fi

# Optional strict gate. Unlike the historical benchmark gate, this fails closed:
# quality must never overlap the performance stage it is meant to follow.
INIT_CONTAINER=""
JOB_DEADLINE=10800
if [[ -n "${WAIT_FOR_MARKER}" ]]; then
    # Quality is submitted at the same time as realtime. Its Job deadline and
    # marker wait must cover the complete realtime (1.5h) + batch (4h) budgets,
    # plus quality itself; init-container time counts toward activeDeadlineSeconds.
    JOB_DEADLINE=32400  # 9h total pipeline envelope
    INIT_CONTAINER=$(cat <<INITEOF
      initContainers:
        - name: wait-for-performance
          image: python:3.11-slim@sha256:d1053354624536b044162aaab1e418bd000ea35184fb1ae098ab3166b1072e72
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -euo pipefail
              export HOME=/tmp
              export PYTHONPATH="/tmp/pip-packages:\${PYTHONPATH:-}"
              pip install --quiet --no-cache-dir --target=/tmp/pip-packages boto3==1.34.0
              echo "Waiting for performance marker s3://${RESULTS_BUCKET}/${WAIT_FOR_MARKER}"
              python3 -c "
              import sys, time
              import boto3
              from botocore.exceptions import ClientError
              s3 = boto3.client('s3', region_name='${AWS_REGION}')
              deadline = time.time() + 21600  # 6h: realtime + batch + coordination headroom
              while time.time() < deadline:
                  try:
                      obj = s3.get_object(Bucket='${RESULTS_BUCKET}', Key='${WAIT_FOR_MARKER}')
                      exit_code = obj['Body'].read().decode().strip()
                      if exit_code != '0':
                          print('Performance stage failed (exit=' + exit_code + '); quality will NOT start.')
                          sys.exit(1)
                      print('Performance marker found (success) — starting quality.'); sys.exit(0)
                  except ClientError as exc:
                      code = str(exc.response.get('Error', {}).get('Code', ''))
                      if code not in ('404', 'NoSuchKey', 'NotFound'):
                          raise
                      print('...performance not done; sleeping 20s'); time.sleep(20)
              print('Timed out waiting for performance marker; quality will NOT start.')
              sys.exit(1)
              "
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
              cpu: "100m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          volumeMounts:
            - name: tmp
              mountPath: /tmp
INITEOF
)
fi

echo ""
echo "======================================================"
echo "  QUALITY EVAL (AutoQA F1)"
echo "  Catalog ID   : ${MODEL}"
echo "  Served model : ${SERVED_NAME}"
echo "  Hardware     : ${HW}"
echo "  Quantization : ${QUANT}"
echo "  Endpoint     : ${ENDPOINT}"
echo "  Config       : ${CONFIG}"
echo "  Dataset      : s3://${RESULTS_BUCKET}/${DATASET_S3_KEY}"
echo "  Tag          : ${TAG:-<none>}"
echo "  Starts after : ${WAIT_FOR_MARKER:-<immediately>}"
echo "  Samples      : ${DUMP_SAMPLES} (max ${MAX_DUMP_SAMPLES})"
echo "  Inputs       : ${DUMP_INPUTS} (transcripts included in samples when true)"
echo "  Results      : s3://${RESULTS_BUCKET}/${QUALITY_S3_PREFIX}"
echo "  Job          : ${JOB_NAME}"
echo "======================================================"
echo ""

cd "${FRAMEWORK_ROOT}"

log_info "Creating isolated ConfigMaps..."
JOB_SUBMITTED="false"
cleanup_unsubmitted_quality() {
    if [[ "${JOB_SUBMITTED}" != "true" ]]; then
        kubectl delete configmap "${SCRIPT_CM}" "${CONFIG_CM}" -n "${BENCHMARK_NAMESPACE}" \
            --ignore-not-found=true >/dev/null 2>&1 || true
    fi
}
trap cleanup_unsubmitted_quality EXIT

kubectl create configmap "${SCRIPT_CM}" \
    --from-file=quality-eval.py="${FRAMEWORK_ROOT}/inference/quality-eval.py" \
    -n "${BENCHMARK_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl label configmap "${SCRIPT_CM}" -n "${BENCHMARK_NAMESPACE}" \
    app.kubernetes.io/component=quality-runner model="${MODEL}" \
    run-tag="${TAG:-untagged}" run-job="${JOB_NAME}" --overwrite >/dev/null

kubectl create configmap "${CONFIG_CM}" \
    --from-file=config.yaml="${FRAMEWORK_ROOT}/${CONFIG}" \
    -n "${BENCHMARK_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl label configmap "${CONFIG_CM}" -n "${BENCHMARK_NAMESPACE}" \
    app.kubernetes.io/component=quality-runner model="${MODEL}" \
    run-tag="${TAG:-untagged}" run-job="${JOB_NAME}" --overwrite >/dev/null
log_info "ConfigMaps ready: ${SCRIPT_CM}, ${CONFIG_CM}"

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
    run-tag: "${TAG:-untagged}"
spec:
  backoffLimit: 0
  activeDeadlineSeconds: ${JOB_DEADLINE}
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ${JOB_NAME}
        app.kubernetes.io/component: quality-runner
        project: observeai-inference-optimization
        model: "${MODEL}"
        run-tag: "${TAG:-untagged}"
      annotations:
        karpenter.sh/do-not-disrupt: "true"
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
${INIT_CONTAINER}
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
                --config /configs/quality/config.yaml \
                --dataset /tmp/autoqa.csv \
                --endpoint "${ENDPOINT}" \
                --model "${SERVED_NAME}" \
                --quantization "${QUANT}" \
                --hardware "${HW}" \
                --output /results \
                --wait-timeout 600 ${QUALITY_EXTRA_ARGS} || TEST_EXIT=\$?

              echo "=== Uploading quality results to S3 ==="
              python3 -c "
              import boto3, os, glob
              s3 = boto3.client('s3', region_name='${AWS_REGION}')
              for f in glob.glob('/results/*.jsonl'):
                  key = '${QUALITY_S3_PREFIX}' + os.path.basename(f)
                  s3.upload_file(f, '${RESULTS_BUCKET}', key)
                  print('Uploaded: s3://${RESULTS_BUCKET}/' + key)
              " || true
              exit \${TEST_EXIT}
          env:
            - name: PYTHONUNBUFFERED
              value: "1"
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
            name: ${SCRIPT_CM}
        - name: config
          configMap:
            name: ${CONFIG_CM}
        - name: results
          emptyDir:
            sizeLimit: 256Mi
        - name: tmp
          emptyDir:
            sizeLimit: 2Gi
EOF

JOB_SUBMITTED="true"
# Make the isolated ConfigMaps children of the Job. Kubernetes garbage collection
# deletes them automatically when Job TTL cleanup (or manual Job deletion) runs.
JOB_UID=$(kubectl get job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" \
    -o jsonpath='{.metadata.uid}' 2>/dev/null || true)
if [[ -n "${JOB_UID}" ]]; then
    OWNER_PATCH="{\"metadata\":{\"ownerReferences\":[{\"apiVersion\":\"batch/v1\",\"kind\":\"Job\",\"name\":\"${JOB_NAME}\",\"uid\":\"${JOB_UID}\",\"controller\":false,\"blockOwnerDeletion\":false}]}}"
    for CM in "${SCRIPT_CM}" "${CONFIG_CM}"; do
        if ! kubectl patch configmap "${CM}" -n "${BENCHMARK_NAMESPACE}" \
            --type=merge -p "${OWNER_PATCH}" >/dev/null; then
            log_warn "Could not attach ownerReference to ConfigMap ${CM}."
        fi
    done
else
    log_warn "Could not read Job UID; ConfigMaps may require manual cleanup."
fi
trap - EXIT

log_info "Quality Job submitted: ${JOB_NAME}"
echo "  Status : kubectl get job ${JOB_NAME} -n ${BENCHMARK_NAMESPACE} -w"
echo "  Logs   : kubectl logs -n ${BENCHMARK_NAMESPACE} -l job-name=${JOB_NAME} -f --tail=100"
echo "  Results: s3://${RESULTS_BUCKET}/${QUALITY_S3_PREFIX}"

if [[ "${DETACH}" == "true" ]]; then
    log_info "Detached. The quality Job continues in-cluster."
    exit 0
fi

log_info "Looking for a quality pod (best effort; Job status remains authoritative)..."
POD=""
for i in $(seq 1 60); do
    POD=$(kubectl get pods -l "job-name=${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    [[ -n "${POD}" ]] && break
    echo -n "."; sleep 5
done
echo ""

if [[ -n "${POD}" ]]; then
    log_info "Following ${POD} (Ctrl+C detaches; the Job continues in-cluster)..."
    kubectl logs "${POD}" -n "${BENCHMARK_NAMESPACE}" -f --all-containers=true 2>/dev/null || true
else
    log_warn "No pod visible yet; continuing to wait on Job ${JOB_NAME} instead of returning early."
fi

# Unless --detach was explicit, do not claim success until Kubernetes reports a
# terminal Job condition. A broken log stream or slow scheduler must not turn a
# running Job into a false-successful CLI result. The 10-minute settlement window
# allows the Job controller to publish Failed after activeDeadlineSeconds.
DEADLINE=$((SECONDS + JOB_DEADLINE + 600))
while [[ ${SECONDS} -lt ${DEADLINE} ]]; do
    FAILED=$(kubectl get job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
    SUCCEEDED=$(kubectl get job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "")
    if [[ "${FAILED}" == "True" ]]; then
        log_error "Quality Job failed. Inspect: kubectl describe job ${JOB_NAME} -n ${BENCHMARK_NAMESPACE}"
        exit 1
    fi
    if [[ "${SUCCEEDED}" == "1" ]]; then
        log_info "Quality evaluation completed successfully."
        aws s3 ls "s3://${RESULTS_BUCKET}/${QUALITY_S3_PREFIX}" \
            --recursive --human-readable --region "${AWS_REGION}" 2>/dev/null || \
            log_warn "Job succeeded but quality results were not found in S3."
        exit 0
    fi
    sleep 10
done

log_error "Timed out waiting for quality Job ${JOB_NAME} to complete."
exit 1
