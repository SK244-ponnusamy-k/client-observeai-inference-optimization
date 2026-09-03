#!/bin/bash
# ==============================================================================
# vllm/models/post-deploy-validate.sh
#
# Automatic post-deploy validation pipeline.
# Called by deploy.sh after the model is Ready.
#
# What this does:
#   1. Waits for vLLM /health to confirm model is serving
#   2. Creates benchmark script ConfigMap (load-test.py)
#   3. Creates manifest + profile ConfigMaps for the run
#   4. Runs realtime benchmark Job (concurrency 1/2/4)
#   5. Runs batch benchmark Job (concurrency 16/32/64/128)
#   6. Waits for both jobs to complete
#   7. Prints pass/fail summary with key metrics
#   8. Uploads results to S3
#   9. Posts Grafana annotation marking the validation run
#
# Called by:
#   bash vllm/models/gpt-oss-20b/deploy.sh   (--validate flag)
#   bash vllm/models/qwen-2.5-0.5b/deploy.sh (--validate flag)
#
# Can also run standalone:
#   cd llm-inference-framework
#   bash vllm/models/post-deploy-validate.sh \
#     --model gpt-oss-20b \
#     --manifest configs/manifests/gpt-oss-20b-baseline.yaml \
#     --endpoint http://oai-infopt-vllm-gpt-oss-20b:8000
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }
log_step()  { echo ""; echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; \
              echo -e "${GREEN}  $1${NC}"; \
              echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ==============================================================================
# Parse arguments
# ==============================================================================
MODEL_NAME=""
MANIFEST_PATH=""
ENDPOINT=""
SKIP_BATCH="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)     MODEL_NAME="$2";    shift 2 ;;
        --manifest)  MANIFEST_PATH="$2"; shift 2 ;;
        --endpoint)  ENDPOINT="$2";      shift 2 ;;
        --skip-batch) SKIP_BATCH="true"; shift ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL_NAME}" || -z "${MANIFEST_PATH}" || -z "${ENDPOINT}" ]]; then
    log_error "Usage: $0 --model <name> --manifest <path> --endpoint <url>"
    log_error "Example: $0 --model gpt-oss-20b --manifest configs/manifests/gpt-oss-20b-baseline.yaml --endpoint http://oai-infopt-vllm-gpt-oss-20b:8000"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
REALTIME_JOB="oai-infopt-validate-realtime-${TIMESTAMP}"
BATCH_JOB="oai-infopt-validate-batch-${TIMESTAMP}"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  POST-DEPLOY VALIDATION"
echo "══════════════════════════════════════════════════════════════"
echo "  Model    : ${MODEL_NAME}"
echo "  Manifest : ${MANIFEST_PATH}"
echo "  Endpoint : ${ENDPOINT}"
echo "  Namespace: ${BENCHMARK_NAMESPACE}"
echo "══════════════════════════════════════════════════════════════"

# ==============================================================================
# Step 1 — Wait for /health
# ==============================================================================
log_step "Step 1 — Waiting for model health endpoint"

# Port-forward to reach the service from within the script
LOCAL_PORT=18080
SVC_NAME="oai-infopt-vllm-${MODEL_NAME}"

log_info "Port-forwarding ${SVC_NAME}:8000 → localhost:${LOCAL_PORT}..."
kubectl port-forward "svc/${SVC_NAME}" "${LOCAL_PORT}:8000" \
    -n "${BENCHMARK_NAMESPACE}" &
PF_PID=$!
trap "kill ${PF_PID} 2>/dev/null || true" EXIT

sleep 3

HEALTH_TIMEOUT=300
HEALTH_START=$(date +%s)
while true; do
    if curl -sf "http://localhost:${LOCAL_PORT}/health" >/dev/null 2>&1; then
        log_info "Model is healthy and serving."
        break
    fi
    NOW=$(date +%s)
    ELAPSED=$((NOW - HEALTH_START))
    if [[ ${ELAPSED} -gt ${HEALTH_TIMEOUT} ]]; then
        log_error "Model health check timed out after ${HEALTH_TIMEOUT}s"
        exit 1
    fi
    echo -n "."
    sleep 5
done
echo ""

# Quick smoke test — list models
MODELS_RESPONSE=$(curl -sf "http://localhost:${LOCAL_PORT}/v1/models" 2>/dev/null || echo "{}")
log_info "Models response: ${MODELS_RESPONSE}"

# ==============================================================================
# Step 2 — Create benchmark ConfigMaps
# ==============================================================================
log_step "Step 2 — Creating benchmark ConfigMaps"

# load-test.py script
log_info "Creating benchmark script ConfigMap..."
kubectl create configmap oai-infopt-benchmark-script \
    --from-file=load-test.py="${FRAMEWORK_ROOT}/inference/load-test.py" \
    --namespace "${BENCHMARK_NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -

# Manifest ConfigMap
log_info "Creating manifest ConfigMap..."
kubectl create configmap oai-infopt-benchmark-manifest \
    --from-file=manifest.yaml="${FRAMEWORK_ROOT}/${MANIFEST_PATH}" \
    --namespace "${BENCHMARK_NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -

# Realtime profile ConfigMap
log_info "Creating realtime profile ConfigMap..."
kubectl create configmap oai-infopt-benchmark-profile-realtime \
    --from-file=profile.yaml="${FRAMEWORK_ROOT}/configs/workload_profiles/realtime_v1.yaml" \
    --namespace "${BENCHMARK_NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -

# Batch profile ConfigMap
log_info "Creating batch profile ConfigMap..."
kubectl create configmap oai-infopt-benchmark-profile-batch \
    --from-file=profile.yaml="${FRAMEWORK_ROOT}/configs/workload_profiles/batch_v1.yaml" \
    --namespace "${BENCHMARK_NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -

log_info "ConfigMaps created."

# ==============================================================================
# Step 3 — Submit realtime benchmark Job
# ==============================================================================
log_step "Step 3 — Submitting realtime benchmark Job"

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${REALTIME_JOB}
  namespace: ${BENCHMARK_NAMESPACE}
  labels:
    app.kubernetes.io/component: benchmark-runner
    project: observeai-inference-optimization
    validation-run: "${TIMESTAMP}"
    model: "${MODEL_NAME}"
    profile: realtime
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 3600
  ttlSecondsAfterFinished: 7200
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
              export HOME=/tmp
              export PIP_USER=yes
              export PYTHONUSERBASE=/tmp/pip-user
              export PYTHONPATH=/tmp/pip-user/lib/python3.11/site-packages
              pip install --quiet --no-cache-dir --user \
                "openai==1.57.0" \
                pyyaml==6.0.1 \
                boto3==1.34.0 \
                "aiohttp==3.10.0"
              python3 /app/load-test.py \
                --manifest /configs/manifests/manifest.yaml \
                --profile  /configs/profiles/profile.yaml \
                --endpoint "${ENDPOINT}" \
                --output   /results \
                --wait-timeout 300
              echo "=== REALTIME RESULTS ==="
              cat /results/*.jsonl 2>/dev/null || echo "No results file found"
              # Upload to S3
              python3 -c "
              import boto3, os, glob
              s3 = boto3.client('s3', region_name='${AWS_REGION}')
              for f in glob.glob('/results/*.jsonl'):
                  key = 'results/validation/${TIMESTAMP}/realtime/' + os.path.basename(f)
                  s3.upload_file(f, '${RESULTS_BUCKET}', key)
                  print(f'Uploaded: s3://${RESULTS_BUCKET}/' + key)
              " 2>/dev/null || echo "S3 upload skipped"
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
            name: oai-infopt-benchmark-profile-realtime
        - name: results
          emptyDir:
            sizeLimit: 1Gi
        - name: tmp
          emptyDir:
            sizeLimit: 512Mi
EOF

log_info "Realtime benchmark Job submitted: ${REALTIME_JOB}"

# ==============================================================================
# Step 4 — Submit batch benchmark Job (unless --skip-batch)
# ==============================================================================
if [[ "${SKIP_BATCH}" == "false" ]]; then
    log_step "Step 4 — Submitting batch benchmark Job"

    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${BATCH_JOB}
  namespace: ${BENCHMARK_NAMESPACE}
  labels:
    app.kubernetes.io/component: benchmark-runner
    project: observeai-inference-optimization
    validation-run: "${TIMESTAMP}"
    model: "${MODEL_NAME}"
    profile: batch
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 7200
  ttlSecondsAfterFinished: 7200
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
              export HOME=/tmp
              export PIP_USER=yes
              export PYTHONUSERBASE=/tmp/pip-user
              export PYTHONPATH=/tmp/pip-user/lib/python3.11/site-packages
              pip install --quiet --no-cache-dir --user \
                "openai==1.57.0" \
                pyyaml==6.0.1 \
                boto3==1.34.0 \
                "aiohttp==3.10.0"
              python3 /app/load-test.py \
                --manifest /configs/manifests/manifest.yaml \
                --profile  /configs/profiles/profile.yaml \
                --endpoint "${ENDPOINT}" \
                --output   /results \
                --wait-timeout 300
              echo "=== BATCH RESULTS ==="
              cat /results/*.jsonl 2>/dev/null || echo "No results file found"
              python3 -c "
              import boto3, os, glob
              s3 = boto3.client('s3', region_name='${AWS_REGION}')
              for f in glob.glob('/results/*.jsonl'):
                  key = 'results/validation/${TIMESTAMP}/batch/' + os.path.basename(f)
                  s3.upload_file(f, '${RESULTS_BUCKET}', key)
                  print(f'Uploaded: s3://${RESULTS_BUCKET}/' + key)
              " 2>/dev/null || echo "S3 upload skipped"
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
            name: oai-infopt-benchmark-profile-batch
        - name: results
          emptyDir:
            sizeLimit: 1Gi
        - name: tmp
          emptyDir:
            sizeLimit: 512Mi
EOF

    log_info "Batch benchmark Job submitted: ${BATCH_JOB}"
else
    log_warn "Batch benchmark skipped (--skip-batch flag set)."
fi

# ==============================================================================
# Step 5 — Wait for realtime job + stream logs
# ==============================================================================
log_step "Step 5 — Waiting for realtime benchmark (concurrency 1/2/4)"

log_info "Waiting for realtime job pod to start..."
for i in $(seq 1 30); do
    REALTIME_POD=$(kubectl get pods \
        -l "job-name=${REALTIME_JOB}" \
        -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    [[ -n "${REALTIME_POD}" ]] && break
    echo -n "."
    sleep 5
done
echo ""
log_info "Realtime pod: ${REALTIME_POD:-not found}"

if [[ -n "${REALTIME_POD:-}" ]]; then
    log_info "Streaming realtime benchmark logs..."
    kubectl logs "${REALTIME_POD}" -n "${BENCHMARK_NAMESPACE}" -f 2>/dev/null || true
fi

log_info "Waiting for realtime job completion (timeout: 60min)..."
kubectl wait job "${REALTIME_JOB}" \
    --for=condition=complete \
    --timeout=3600s \
    -n "${BENCHMARK_NAMESPACE}" && REALTIME_STATUS="PASSED" || REALTIME_STATUS="FAILED"

log_info "Realtime benchmark: ${REALTIME_STATUS}"

# ==============================================================================
# Step 6 — Wait for batch job (if running)
# ==============================================================================
BATCH_STATUS="SKIPPED"
if [[ "${SKIP_BATCH}" == "false" ]]; then
    log_step "Step 6 — Waiting for batch benchmark (concurrency 16/32/64/128)"

    log_info "Waiting for batch job pod to start..."
    for i in $(seq 1 30); do
        BATCH_POD=$(kubectl get pods \
            -l "job-name=${BATCH_JOB}" \
            -n "${BENCHMARK_NAMESPACE}" \
            -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
        [[ -n "${BATCH_POD}" ]] && break
        echo -n "."
        sleep 5
    done
    echo ""
    log_info "Batch pod: ${BATCH_POD:-not found}"

    if [[ -n "${BATCH_POD:-}" ]]; then
        log_info "Streaming batch benchmark logs..."
        kubectl logs "${BATCH_POD}" -n "${BENCHMARK_NAMESPACE}" -f 2>/dev/null || true
    fi

    log_info "Waiting for batch job completion (timeout: 2h)..."
    kubectl wait job "${BATCH_JOB}" \
        --for=condition=complete \
        --timeout=7200s \
        -n "${BENCHMARK_NAMESPACE}" && BATCH_STATUS="PASSED" || BATCH_STATUS="FAILED"

    log_info "Batch benchmark: ${BATCH_STATUS}"
fi

# ==============================================================================
# Step 7 — Post Grafana annotation marking this validation run
# ==============================================================================
log_step "Step 7 — Posting Grafana annotation"

# Port-forward Grafana to post annotation
GRAFANA_SVC=$(kubectl get svc -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "kube-prometheus-stack-grafana")

GRAFANA_PASS=$(kubectl get secret kube-prometheus-stack-grafana \
    -n monitoring \
    -o jsonpath="{.data.admin-password}" 2>/dev/null | base64 -d 2>/dev/null || echo "")

if [[ -n "${GRAFANA_PASS}" ]]; then
    kubectl port-forward "svc/${GRAFANA_SVC}" 13000:80 -n monitoring &
    GRAFANA_PF_PID=$!
    sleep 3

    OVERALL_STATUS="PASSED"
    [[ "${REALTIME_STATUS}" == "FAILED" || "${BATCH_STATUS}" == "FAILED" ]] && OVERALL_STATUS="FAILED"

    curl -sf -X POST "http://localhost:13000/api/annotations" \
        -H "Content-Type: application/json" \
        -u "admin:${GRAFANA_PASS}" \
        -d "{
            \"text\": \"Validation run: ${MODEL_NAME} | realtime=${REALTIME_STATUS} | batch=${BATCH_STATUS} | ts=${TIMESTAMP}\",
            \"tags\": [\"validation\", \"${MODEL_NAME}\", \"${OVERALL_STATUS}\"],
            \"time\": $(date +%s)000
        }" >/dev/null 2>&1 && log_info "Grafana annotation posted." || log_warn "Grafana annotation failed (non-critical)."

    kill ${GRAFANA_PF_PID} 2>/dev/null || true
fi

# ==============================================================================
# Done — print summary
# ==============================================================================
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  VALIDATION COMPLETE — ${MODEL_NAME}"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  Realtime benchmark : ${REALTIME_STATUS}"
echo "  Batch benchmark    : ${BATCH_STATUS}"
echo ""
echo "  Results in S3:"
echo "    aws s3 ls s3://${RESULTS_BUCKET}/results/validation/${TIMESTAMP}/"
echo ""
echo "  Grafana dashboard:"
echo "    http://localhost:3000 → vLLM + GPU Overview"
echo "    (look for annotation marker at ${TIMESTAMP})"
echo ""

# Exit non-zero if any benchmark failed
if [[ "${REALTIME_STATUS}" == "FAILED" || "${BATCH_STATUS}" == "FAILED" ]]; then
    log_error "One or more benchmarks failed — check logs above."
    exit 1
fi

log_info "All validations passed."
