#!/bin/bash
# ==============================================================================
# inference/run-benchmark.sh
#
# One command to run a benchmark Job inside the cluster and upload results to S3.
#
# The Job runs the orchestrator (inference/load-test.py), which shells out to
# `vllm bench serve`. That CLI ships in the vLLM DLC image, so the Job runs from
# ${VLLM_IMAGE} (CPU-only) — NOT python:3.11-slim. boto3 is pip-installed to /tmp
# for the S3 upload (the DLC already has vllm + tokenizers + pyyaml).
#
# Usage:
#   bash inference/run-benchmark.sh                                    # gpt-oss-20b, g6e, realtime
#   bash inference/run-benchmark.sh --model gpt-oss-20b --profile batch
#   bash inference/run-benchmark.sh --model gpt-oss-20b --hw g5        # pick the matrix cell
#   bash inference/run-benchmark.sh --model gpt-oss-20b --profile realtime --hw g6
#   bash inference/run-benchmark.sh --manifest configs/manifests/gpt-oss-20b-g6e-mxfp4.yaml
#
# NOTE: run once per profile. Deploy the model on the SAME --hw first:
#   bash vllm/models/gpt-oss-20b/deploy.sh --hw g6e.2xlarge
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

# The benchmark Job needs the vLLM CLI → run from the pinned DLC image.
: "${VLLM_IMAGE:?ERROR: VLLM_IMAGE is not set. Ensure config/config.env exports it with a pinned digest.}"
: "${RESULTS_BUCKET:?ERROR: RESULTS_BUCKET is not set in config/config.env.}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

# Defaults
MODEL="gpt-oss-20b"
PROFILE="realtime"
HW="g6e"                 # matrix cell suffix: g5 | g6 | g6e
MANIFEST_OVERRIDE=""     # optional explicit manifest path

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL="$2";    shift 2 ;;
        --profile)  PROFILE="$2";  shift 2 ;;
        --hw)       HW="$2";       shift 2 ;;
        --manifest) MANIFEST_OVERRIDE="$2"; shift 2 ;;
        *) log_error "Unknown: $1"; exit 1 ;;
    esac
done

# Resolve service name + default quantization (→ manifest cell) from model name.
case "${MODEL}" in
    gpt-oss-20b)
        SVC="oai-infopt-vllm-gpt-oss-20b"; QUANT="mxfp4" ;;
    qwen3.5-4b|qwen3-5-4b)
        SVC="oai-infopt-vllm-qwen3-5-4b"; QUANT="bf16" ;;
    gemma-4-26b-a4b|gemma-4-26b-a4b-it|gemma)
        SVC="oai-infopt-vllm-gemma-4-26b-a4b"; QUANT="w4a16" ;;
    qwen-0.5b|qwen-0-5b|qwen)
        SVC="oai-infopt-vllm-qwen-0-5b"; QUANT="" ;;   # smoke uses its own baseline manifest
    *)
        log_error "Unknown model: ${MODEL}."
        log_error "Use: gpt-oss-20b | qwen3.5-4b | gemma-4-26b-a4b | qwen-0.5b"
        exit 1
        ;;
esac

# Trainium (Neuron, single-instance) cells: separate service, quant by generation.
#   trn3 → MXFP4 (native), trn2 → BF16. Manifest cell: <model>-<hw>-<quant>.yaml
if [[ "${HW}" == trn* ]]; then
    SVC="${SVC}-neuron"
    case "${HW}" in
        trn3*) QUANT="mxfp4" ;;
        trn2*) QUANT="bf16" ;;
    esac
fi

# Resolve the manifest: explicit override > smoke baseline > matrix cell <model>-<hw>-<quant>.
if [[ -n "${MANIFEST_OVERRIDE}" ]]; then
    MANIFEST="${MANIFEST_OVERRIDE}"
elif [[ "${MODEL}" == qwen-0.5b || "${MODEL}" == qwen-0-5b || "${MODEL}" == qwen ]]; then
    MANIFEST="configs/manifests/qwen-2.5-0.5b-baseline.yaml"
else
    MANIFEST="configs/manifests/${MODEL}-${HW}-${QUANT}.yaml"
fi

PROFILE_FILE="configs/workload_profiles/${PROFILE}_v1.yaml"

# Fail fast on missing config files (better than a confusing ConfigMap error).
[[ -f "${FRAMEWORK_ROOT}/${MANIFEST}" ]] || { log_error "Manifest not found: ${MANIFEST}"; exit 1; }
[[ -f "${FRAMEWORK_ROOT}/${PROFILE_FILE}" ]] || { log_error "Profile not found: ${PROFILE_FILE}"; exit 1; }

ENDPOINT="http://${SVC}:8000"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="oai-infopt-bench-${MODEL//\./-}-${PROFILE}-${TIMESTAMP}"

echo ""
echo "======================================================"
echo "  BENCHMARK JOB"
echo "  Model    : ${MODEL}"
echo "  Hardware : ${HW}"
echo "  Manifest : ${MANIFEST}"
echo "  Profile  : ${PROFILE}"
echo "  Endpoint : ${ENDPOINT}"
echo "  Image    : ${VLLM_IMAGE}"
echo "  Job      : ${JOB_NAME}"
echo "  Results  : s3://${RESULTS_BUCKET}/results/${TIMESTAMP}/${PROFILE}/"
echo "======================================================"
echo ""

cd "${FRAMEWORK_ROOT}"

# ==============================================================================
# Step 1 — Create / update ConfigMaps (script + manifest + profile)
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

log_info "ConfigMaps ready. (Inputs come from vllm bench serve's built-in 'random' dataset — no external dataset needed.)"

# ==============================================================================
# Step 2 — Submit Job (runs from the vLLM DLC image, CPU-only)
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
  activeDeadlineSeconds: 10800
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ${JOB_NAME}
        app.kubernetes.io/component: benchmark-runner
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
      # Do NOT co-locate the runner on the GPU inference node (latency contamination).
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
          image: ${VLLM_IMAGE}
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -euo pipefail
              export HOME=/tmp
              # boto3 for the S3 upload (not in the DLC). vllm/pyyaml/tokenizers already present.
              export PYTHONPATH="/tmp/pip-packages:\${PYTHONPATH:-}"
              pip install --quiet --no-cache-dir --target=/tmp/pip-packages boto3==1.34.0 || true

              TEST_EXIT=0
              python3 /app/load-test.py \
                --manifest /configs/manifests/manifest.yaml \
                --profile  /configs/profiles/profile.yaml \
                --endpoint "${ENDPOINT}" \
                --output   /results \
                --wait-timeout 600 || TEST_EXIT=\$?

              echo "=== Uploading results to S3 ==="
              python3 -c "
              import boto3, os, glob
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
            - name: RESULTS_BUCKET
              value: "${RESULTS_BUCKET}"
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
              cpu: "2"
              memory: "4Gi"
            limits:
              cpu: "4"
              memory: "8Gi"
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
            sizeLimit: 4Gi
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
JOB_TIMEOUT=10800   # matches activeDeadlineSeconds
JOB_DONE="false"
DEADLINE=$((SECONDS + JOB_TIMEOUT))
while [[ ${SECONDS} -lt ${DEADLINE} ]]; do
    JOB_COMPLETE=$(kubectl get job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "")
    JOB_FAILED=$(kubectl get job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
    JOB_SUCCEEDED=$(kubectl get job "${JOB_NAME}" -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "")

    if [[ "${JOB_COMPLETE}" == "True" || "${JOB_SUCCEEDED}" == "1" ]]; then
        STATUS="PASSED"; JOB_DONE="true"; break
    elif [[ "${JOB_FAILED}" == "True" ]]; then
        # Failed job exit=1 can mean SLO violation (results still valid) or a real error.
        STATUS="FAILED_OR_SLO"; JOB_DONE="true"; break
    fi
    sleep 10
done
[[ "${JOB_DONE}" != "true" ]] && STATUS="TIMEOUT"

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
