#!/bin/bash
# ==============================================================================
# scripts/deploy.sh
#
# Generic vLLM deploy — reads everything from models/<model-id>.yaml.
# No per-model scripts needed. No Python dependency.
#
# Usage:
#   bash scripts/deploy.sh --model gpt-oss-20b
#   bash scripts/deploy.sh --model qwen-2.5-0.5b
#   bash scripts/deploy.sh --model qwen3-35b-nvfp4
#   bash scripts/deploy.sh --model gpt-oss-20b --validate
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
# Reads a top-level scalar value from a YAML file.
yaml_field() {
    local file="$1" key="$2"
    grep -m1 "^${key}:" "${file}" | sed "s/^${key}:[[:space:]]*//" | sed 's/[[:space:]]*#.*//' | tr -d "'\"" | xargs
}

# Reads a serving sub-key (indented under "serving:")
yaml_serving() {
    local file="$1" key="$2"
    awk "/^serving:/{found=1} found && /^  ${key}:/{gsub(/^  ${key}:[[:space:]]*/,\"\"); sub(/[[:space:]]*#.*/,\"\"); gsub(/['\"]*/,\"\"); print; exit}" "${file}" | xargs
}

# Reads a resources sub-key
yaml_resources() {
    local file="$1" key="$2"
    awk "/^resources:/{found=1} found && /^  ${key}:/{gsub(/^  ${key}:[[:space:]]*/,\"\"); sub(/[[:space:]]*#.*/,\"\"); gsub(/['\"]*/,\"\"); print; exit}" "${file}" | xargs
}

# Reads a probes sub-key
yaml_probes() {
    local file="$1" key="$2"
    awk "/^probes:/{found=1} found && /^  ${key}:/{gsub(/^  ${key}:[[:space:]]*/,\"\"); sub(/[[:space:]]*#.*/,\"\"); gsub(/['\"]*/,\"\"); print; exit}" "${file}" | xargs
}

# ── Parse args ────────────────────────────────────────────────────────────────
MODEL=""
VALIDATE="false"
RUN_BENCHMARK="false"
PROFILE="realtime"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)     MODEL="$2";          shift 2 ;;
        --validate)  VALIDATE="true";     shift ;;
        --benchmark) RUN_BENCHMARK="true"; shift ;;
        --profile)   PROFILE="$2";        shift 2 ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    log_error "Usage: bash scripts/deploy.sh --model <model-id>"
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
        log_error "Available: $(ls "${FRAMEWORK_ROOT}/models/"*.yaml | xargs -I{} basename {} .yaml | tr '\n' ' ')"
        exit 1
    }
fi

# ── Load all fields from model YAML ──────────────────────────────────────────
MODEL_ID=$(yaml_field "${MODEL_FILE}" "model_id")
MODEL_HF_ID=$(yaml_field "${MODEL_FILE}" "hf_id")
MODEL_S3_FOLDER=$(yaml_field "${MODEL_FILE}" "s3_folder")
MODEL_SERVED_NAME=$(yaml_field "${MODEL_FILE}" "served_name")
PORT_FORWARD_PORT=$(yaml_field "${MODEL_FILE}" "port_forward_port")
BENCHMARK_MANIFEST=$(yaml_field "${MODEL_FILE}" "benchmark_manifest")

GPU_MEM_UTIL=$(yaml_serving "${MODEL_FILE}" "gpu_memory_utilization")
MAX_MODEL_LEN=$(yaml_serving "${MODEL_FILE}" "max_model_len")
MAX_NUM_SEQS=$(yaml_serving "${MODEL_FILE}" "max_num_seqs")
MAX_NUM_BATCHED_TOKENS=$(yaml_serving "${MODEL_FILE}" "max_num_batched_tokens")
QUANTIZATION=$(yaml_serving "${MODEL_FILE}" "quantization")
LOAD_FORMAT=$(yaml_serving "${MODEL_FILE}" "load_format")

CPU_REQUEST=$(yaml_resources "${MODEL_FILE}" "cpu_request")
CPU_LIMIT=$(yaml_resources "${MODEL_FILE}" "cpu_limit")
MEM_REQUEST=$(yaml_resources "${MODEL_FILE}" "memory_request")
MEM_LIMIT=$(yaml_resources "${MODEL_FILE}" "memory_limit")
GPU_COUNT=$(yaml_resources "${MODEL_FILE}" "gpu")
PVC_STORAGE=$(yaml_resources "${MODEL_FILE}" "pvc_storage")
TMP_SIZE=$(yaml_resources "${MODEL_FILE}" "tmp_size")
DSHM_SIZE=$(yaml_resources "${MODEL_FILE}" "dshm_size")

READINESS_DELAY=$(yaml_probes "${MODEL_FILE}" "readiness_initial_delay_s")
LIVENESS_DELAY=$(yaml_probes "${MODEL_FILE}" "liveness_initial_delay_s")
GRACE_PERIOD=$(yaml_probes "${MODEL_FILE}" "termination_grace_period_s")

# Validate required fields
if [[ -z "${MODEL_ID}" ]]; then
    log_error "Could not read model_id from ${MODEL_FILE}"; exit 1
fi

DEPLOYMENT_NAME="oai-infopt-vllm-${MODEL_ID}"
SERVICE_NAME="oai-infopt-vllm-${MODEL_ID}"
PVC_NAME="oai-infopt-metadata-${MODEL_ID}"

: "${VLLM_IMAGE:?ERROR: VLLM_IMAGE is not set. Ensure config/config.env exports VLLM_IMAGE.}"

echo ""
echo "══════════════════════════════════════════════════"
echo "  DEPLOYING — ${MODEL_HF_ID}"
echo "  Deployment : ${DEPLOYMENT_NAME}"
echo "  Service    : ${SERVICE_NAME}:8000"
echo "  Namespace  : ${BENCHMARK_NAMESPACE}"
echo "  Quantize   : ${QUANTIZATION:-none}"
echo "  GPU mem    : ${GPU_MEM_UTIL}"
echo "══════════════════════════════════════════════════"
echo ""

# ── Check model in S3 ─────────────────────────────────────────────────────────
log_info "Checking model in S3..."
if ! aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_S3_FOLDER}/" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log_error "Model not found: s3://${MODEL_BUCKET}/${MODEL_S3_FOLDER}/"
    log_error "Download it first: bash scripts/download.sh --model ${MODEL}"
    exit 1
fi
log_info "Model found in S3."

# ── Apply PVC ─────────────────────────────────────────────────────────────────
log_info "Applying PVC: ${PVC_NAME}..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC_NAME}
  namespace: ${BENCHMARK_NAMESPACE}
  labels:
    app.kubernetes.io/component: metadata-cache
    project: observeai-inference-optimization
    model: ${MODEL_ID}
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: gp3
  resources:
    requests:
      storage: ${PVC_STORAGE}
EOF

# ── Apply Service ─────────────────────────────────────────────────────────────
log_info "Applying service: ${SERVICE_NAME}..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: ${SERVICE_NAME}
  namespace: ${BENCHMARK_NAMESPACE}
  labels:
    app.kubernetes.io/name: ${SERVICE_NAME}
    app.kubernetes.io/component: inference-server
    project: observeai-inference-optimization
    model: ${MODEL_ID}
spec:
  type: ClusterIP
  selector:
    app: ${DEPLOYMENT_NAME}
  ports:
    - name: http
      port: 8000
      targetPort: 8000
      protocol: TCP
EOF

# ── Build vLLM args list ──────────────────────────────────────────────────────
ARGS_YAML="            - \"--model=s3://${MODEL_BUCKET}/${MODEL_S3_FOLDER}/\""
ARGS_YAML+=$'\n'"            - \"--load-format=${LOAD_FORMAT}\""
ARGS_YAML+=$'\n'"            - \"--host=0.0.0.0\""
ARGS_YAML+=$'\n'"            - \"--port=8000\""
ARGS_YAML+=$'\n'"            - \"--served-model-name=${MODEL_SERVED_NAME}\""
ARGS_YAML+=$'\n'"            - \"--gpu-memory-utilization=${GPU_MEM_UTIL}\""
ARGS_YAML+=$'\n'"            - \"--max-model-len=${MAX_MODEL_LEN}\""
[[ -n "${MAX_NUM_SEQS}" ]]           && ARGS_YAML+=$'\n'"            - \"--max-num-seqs=${MAX_NUM_SEQS}\""
[[ -n "${MAX_NUM_BATCHED_TOKENS}" ]] && ARGS_YAML+=$'\n'"            - \"--max-num-batched-tokens=${MAX_NUM_BATCHED_TOKENS}\""
[[ -n "${QUANTIZATION}" ]]           && ARGS_YAML+=$'\n'"            - \"--quantization=${QUANTIZATION}\""

# Append extra_args lines from YAML (lines under "  extra_args:" starting with "    -")
while IFS= read -r line; do
    arg=$(echo "${line}" | sed 's/^[[:space:]]*-[[:space:]]*//' | tr -d '"'"'"'')
    [[ -z "${arg}" ]] && continue
    ARGS_YAML+=$'\n'"            - \"${arg}\""
done < <(awk '/^  extra_args:/{found=1; next} found && /^    -/{print} found && /^  [a-z]/{exit}' "${MODEL_FILE}")

# ── Apply Deployment ──────────────────────────────────────────────────────────
log_info "Applying deployment: ${DEPLOYMENT_NAME}..."
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${DEPLOYMENT_NAME}
  namespace: ${BENCHMARK_NAMESPACE}
  labels:
    app.kubernetes.io/name: ${DEPLOYMENT_NAME}
    app.kubernetes.io/component: inference-server
    app.kubernetes.io/managed-by: kubectl
    project: observeai-inference-optimization
    engagement: shellkode-sow
    environment: sandbox
    model: ${MODEL_ID}
  annotations:
    checkov.io/skip1: "CKV_K8S_43=Image digest pinned in VLLM_IMAGE via config/config.env"
    checkov.io/skip2: "CKV_K8S_14=Image tag pinned; static analysis cannot resolve runtime vars"
    checkov.io/skip3: "CKV_K8S_15=imagePullPolicy IfNotPresent acceptable with digest-pinned image"
    checkov.io/skip4: "CKV_K8S_40=runAsUser 1000 pending vLLM UID compatibility test"
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: ${DEPLOYMENT_NAME}
  template:
    metadata:
      labels:
        app: ${DEPLOYMENT_NAME}
        app.kubernetes.io/name: ${DEPLOYMENT_NAME}
        app.kubernetes.io/component: inference-server
        project: observeai-inference-optimization
        model: ${MODEL_ID}
    spec:
      serviceAccountName: oai-infopt-serving-sa
      automountServiceAccountToken: false
      hostPID: false
      hostIPC: false
      hostNetwork: false
      terminationGracePeriodSeconds: ${GRACE_PERIOD}
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app.kubernetes.io/component: benchmark-runner
                topologyKey: kubernetes.io/hostname
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: vllm
          image: ${VLLM_IMAGE}
          imagePullPolicy: IfNotPresent
          args:
${ARGS_YAML}
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
          envFrom:
            - configMapRef:
                name: oai-infopt-vllm-config
          env:
            - name: HOME
              value: "/tmp"
            - name: USER
              value: "vllm"
            - name: TORCHINDUCTOR_CACHE_DIR
              value: "/tmp/torchinductor_cache"
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            runAsUser: 1000
            runAsGroup: 1000
            capabilities:
              drop: ["ALL"]
            seccompProfile:
              type: RuntimeDefault
          resources:
            requests:
              cpu: "${CPU_REQUEST}"
              memory: "${MEM_REQUEST}"
              nvidia.com/gpu: "${GPU_COUNT}"
            limits:
              cpu: "${CPU_LIMIT}"
              memory: "${MEM_LIMIT}"
              nvidia.com/gpu: "${GPU_COUNT}"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: ${READINESS_DELAY}
            periodSeconds: 10
            failureThreshold: 36
            timeoutSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: ${LIVENESS_DELAY}
            periodSeconds: 30
            failureThreshold: 10
            timeoutSeconds: 5
          volumeMounts:
            - name: metadata-cache
              mountPath: /metadata
            - name: dshm
              mountPath: /dev/shm
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: metadata-cache
          persistentVolumeClaim:
            claimName: ${PVC_NAME}
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: ${DSHM_SIZE}
        - name: tmp
          emptyDir:
            sizeLimit: ${TMP_SIZE}
EOF

# ── Wait for rollout ──────────────────────────────────────────────────────────
log_info "Waiting for deployment (model loading takes several minutes)..."
kubectl rollout status deployment/"${DEPLOYMENT_NAME}" \
    -n "${BENCHMARK_NAMESPACE}" \
    --timeout=900s || {
    log_warn "Not ready yet — check: kubectl logs deployment/${DEPLOYMENT_NAME} -n ${BENCHMARK_NAMESPACE} | tail -50"
}

echo ""
echo "══════════════════════════════════════════════════"
echo "  DEPLOYED — ${MODEL_ID}"
echo "══════════════════════════════════════════════════"
kubectl get pods -l "app=${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}"
echo ""
echo "  Port-forward:"
echo "    kubectl port-forward svc/${SERVICE_NAME} ${PORT_FORWARD_PORT}:8000 -n ${BENCHMARK_NAMESPACE} &"
echo "    curl http://localhost:${PORT_FORWARD_PORT}/health"
echo ""
echo "  Benchmark:"
echo "    bash inference/run-benchmark.sh --model ${MODEL} --profile ${PROFILE}"
echo "══════════════════════════════════════════════════"

# ── Optional automatic benchmark trigger ─────────────────────────────────────
if [[ "${RUN_BENCHMARK}" == "true" ]]; then
    log_info "Deployment ready — automatically launching benchmark (${PROFILE} profile)..."
    bash "${FRAMEWORK_ROOT}/inference/run-benchmark.sh" --model "${MODEL}" --profile "${PROFILE}"
fi

if [[ "${VALIDATE}" == "true" ]]; then
    log_info "Running post-deploy validation pipeline..."
    bash "${FRAMEWORK_ROOT}/scripts/post-deploy-validate.sh" \
        --model "${MODEL_SERVED_NAME}" \
        --manifest "${BENCHMARK_MANIFEST}" \
        --endpoint "http://${SERVICE_NAME}:8000"
fi
