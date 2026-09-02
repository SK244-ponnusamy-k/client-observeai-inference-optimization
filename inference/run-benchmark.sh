#!/bin/bash
# ==============================================================================
# inference/run-benchmark.sh
#
# One command to run a benchmark Job inside the cluster and upload results to S3.
#
# Usage:
#   cd llm-inference-framework
#   bash inference/run-benchmark.sh                          # defaults to gpt-oss-20b realtime
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2";   shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        *) log_error "Unknown: $1"; exit 1 ;;
    esac
done

# Resolve manifest and endpoint from model name
case "${MODEL}" in
    gpt-oss-20b)
        MANIFEST="configs/manifests/gpt-oss-20b-baseline.yaml"
        SVC="oai-infopt-vllm-gpt-oss-20b"
        ;;
    qwen-0.5b|qwen)
        MANIFEST="configs/manifests/qwen-2.5-0.5b-baseline.yaml"
        SVC="oai-infopt-vllm-qwen-0.5b"
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
echo "══════════════════════════════════════════════════"
echo "  BENCHMARK JOB"
echo "  Model   : ${MODEL}"
echo "  Profile : ${PROFILE}"
echo "  Endpoint: ${ENDPOINT}"
echo "  Job     : ${JOB_NAME}"
echo "  Results : s3://${RESULTS_BUCKET}/results/${TIMESTAMP}/"
echo "══════════════════════════════════════════════════"
echo ""

# Step 1 — Create ConfigMaps
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

log_info "ConfigMaps ready."

# Step 2 — Submit Job
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
              pip install --quiet --no-cache-dir openai==1.35.14 pyyaml==6.0.1 aiohttp==3.9.5 boto3==1.34.0
              python3 /app/load-test.py \
                --manifest /configs/manifests/manifest.yaml \
                --profile  /configs/profiles/profile.yaml \
                --endpoint "${ENDPOINT}" \
                --output   /results \
                --wait-timeout 300
              echo "=== Uploading results to S3 ==="
              python3 -c "
              import boto3, os, glob
              s3 = boto3.client('s3', region_name='${AWS_REGION}')
              for f in glob.glob('/results/*.jsonl'):
                  key = 'results/${TIMESTAMP}/${PROFILE}/' + os.path.basename(f)
                  s3.upload_file(f, '${RESULTS_BUCKET}', key)
                  print('Uploaded: s3://${RESULTS_BUCKET}/' + key)
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

# Step 3 — Wait for pod and stream logs
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
    log_info "Streaming logs..."
    kubectl logs "${POD}" -n "${BENCHMARK_NAMESPACE}" -f 2>/dev/null || true
fi

# Step 4 — Wait for completion
log_info "Waiting for job completion..."
kubectl wait job "${JOB_NAME}" \
    --for=condition=complete \
    --timeout=7200s \
    -n "${BENCHMARK_NAMESPACE}" && STATUS="PASSED" || STATUS="FAILED"

echo ""
echo "══════════════════════════════════════════════════"
echo "  BENCHMARK ${STATUS}"
echo "  Results: s3://${RESULTS_BUCKET}/results/${TIMESTAMP}/${PROFILE}/"
echo "══════════════════════════════════════════════════"

# Step 5 — Show S3 results
log_info "S3 results:"
aws s3 ls "s3://${RESULTS_BUCKET}/results/${TIMESTAMP}/" \
    --recursive --human-readable --region "${AWS_REGION}" 2>/dev/null || \
    log_warn "No results found in S3 yet"

[[ "${STATUS}" == "PASSED" ]] || exit 1
