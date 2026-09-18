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
PROFILE="both"           # realtime | batch | both — 'both' runs the two profiles in ONE job/container
HW="g6e"                 # matrix cell suffix: g5 | g6 | g6e
MANIFEST_OVERRIDE=""     # optional explicit manifest path
DATASET_OVERRIDE=""      # optional explicit dataset path / S3 key
IMAGE_OVERRIDE=""        # optional explicit benchmark runner image
SVC_OVERRIDE=""          # optional explicit vLLM service name (for --tag parallel deploys)
TAG=""                   # optional deploy tag — isolates the S3 results subfolder
INSTANCE_TYPE=""         # optional FULL instance type (e.g. g7e.24xlarge) for dynamic cost/label

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL="$2";    shift 2 ;;
        --profile)  PROFILE="$2";  shift 2 ;;
        --hw)       HW="$2";       shift 2 ;;
        --manifest) MANIFEST_OVERRIDE="$2"; shift 2 ;;
        --dataset)  DATASET_OVERRIDE="$2";  shift 2 ;;
        --image)    IMAGE_OVERRIDE="$2";    shift 2 ;;
        --svc)      SVC_OVERRIDE="$2";      shift 2 ;;
        --tag)      TAG="$2";               shift 2 ;;
        --instance) INSTANCE_TYPE="$2";     shift 2 ;;
        *) log_error "Unknown: $1"; exit 1 ;;
    esac
done

# Full instance type the model was deployed on (e.g. g7e.24xlarge). Passed into
# the benchmark pod as OAI_INSTANCE_TYPE so load-test.py records the REAL instance
# and resolves its cost from the EC2 price book dynamically — no per-instance
# manifest edits. Falls back to the manifest's serving.instance_type when unset.
: "${INSTANCE_TYPE:=}"

BENCHMARK_RUNNER_IMAGE="${IMAGE_OVERRIDE:-${BENCHMARK_IMAGE:-$VLLM_IMAGE}}"

# Resolve service name + default quantization (→ manifest cell) from model name.
case "${MODEL}" in
    gpt-oss-20b)
        SVC="oai-infopt-vllm-gpt-oss-20b"; QUANT="mxfp4" ;;
    qwen3.5-4b|qwen3-5-4b)
        SVC="oai-infopt-vllm-qwen3-5-4b"; QUANT="bf16" ;;
    gemma-4-26b-a4b|gemma-4-26b-a4b-it|gemma)
        SVC="oai-infopt-vllm-gemma-4-26b-a4b"; QUANT="w4a16" ;;
    gemma-4-31b|gemma-4-31b-it)
        SVC="oai-infopt-vllm-gemma-4-31b"; QUANT="" ;;  # single baseline manifest (TP=4), not a matrix cell
    qwen-0.5b|qwen-0-5b|qwen)
        SVC="oai-infopt-vllm-qwen-0-5b"; QUANT="" ;;   # smoke uses its own baseline manifest
    *)
        # Generic fallback for catalog-onboarded models (oai model generate).
        # The service name always follows the convention oai-infopt-vllm-<id>, and
        # the quantization is carried in the manifest filename, so an explicit
        # --manifest (which oai benchmark always passes) makes this fully generic.
        SVC="oai-infopt-vllm-${MODEL}"
        QUANT="${QUANT:-}"
        if [[ -z "${MANIFEST_OVERRIDE}" ]]; then
            log_warn "Unknown model '${MODEL}' — relying on --manifest / <model>-<hw>-<quant> convention."
        fi
        ;;
esac

# --svc overrides the resolved service name (used by tagged parallel deploys,
# e.g. --svc oai-infopt-vllm-gpt-oss-20b-g7e). Default keeps the base name.
if [[ -n "${SVC_OVERRIDE}" ]]; then
    SVC="${SVC_OVERRIDE}"
fi

# Resolve the manifest: explicit override > single-manifest baselines > matrix cell <model>-<hw>-<quant>.
# qwen-0.5b (smoke) and gemma-4-31b (dense TP=4) each ship ONE manifest, not a
# per-hardware matrix, so they bypass the <model>-<hw>-<quant> naming.
if [[ -n "${MANIFEST_OVERRIDE}" ]]; then
    MANIFEST="${MANIFEST_OVERRIDE}"
elif [[ "${MODEL}" == qwen-0.5b || "${MODEL}" == qwen-0-5b || "${MODEL}" == qwen ]]; then
    MANIFEST="configs/manifests/qwen-2.5-0.5b-baseline.yaml"
elif [[ "${MODEL}" == gemma-4-31b || "${MODEL}" == gemma-4-31b-it ]]; then
    MANIFEST="configs/manifests/gemma-4-31b-baseline.yaml"
else
    MANIFEST="configs/manifests/${MODEL}-${HW}-${QUANT}.yaml"
fi

# Resolve which profile(s) to run. 'both' runs realtime + batch in ONE job so the
# container is created once (image pulled once) and both run back-to-back inside it.
case "${PROFILE}" in
    both)     PROFILE_LIST=("realtime" "batch") ;;
    realtime) PROFILE_LIST=("realtime") ;;
    batch)    PROFILE_LIST=("batch") ;;
    *) log_error "Invalid --profile '${PROFILE}'. Use: realtime | batch | both"; exit 1 ;;
esac

# Fail fast on missing config files (better than a confusing ConfigMap error).
[[ -f "${FRAMEWORK_ROOT}/${MANIFEST}" ]] || { log_error "Manifest not found: ${MANIFEST}"; exit 1; }
for P in "${PROFILE_LIST[@]}"; do
    [[ -f "${FRAMEWORK_ROOT}/configs/workload_profiles/${P}_v1.yaml" ]] || \
        { log_error "Profile not found: configs/workload_profiles/${P}_v1.yaml"; exit 1; }
done

ENDPOINT="http://${SVC}:8000"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Optional tag isolates parallel runs of the SAME model on different instances.
# It is folded into the Job name (unique k8s object) and the S3 result key
# (so two concurrent runs don't overwrite each other's results).
TAG_SEG=""
if [[ -n "${TAG}" ]]; then
    TAG_SEG="-${TAG}"
fi
# Base name shared by all per-profile Jobs. Each profile becomes its OWN Job
# object: <base>-realtime and <base>-batch. They are separate k8s Jobs so a
# batch failure never discards the completed realtime results (and vice-versa).
JOB_NAME_BASE="oai-infopt-bench-${MODEL//\./-}${TAG_SEG}-${TIMESTAMP}"
# S3 result prefix segment: results/<ts>/<tag>/<profile>/ when tagged, else results/<ts>/<profile>/
S3_TAG_SEG=""
if [[ -n "${TAG}" ]]; then
    S3_TAG_SEG="${TAG}/"
fi
# In-cluster ordering gate (no k8s RBAC needed): the realtime Job writes this
# marker to S3 when it finishes; the batch Job's init container blocks until the
# marker appears, so batch only starts loading the GPU AFTER realtime is done.
# Both Jobs are submitted up front, so closing the terminal / Ctrl+C is safe.
S3_MARKER_KEY="results/${TIMESTAMP}/${S3_TAG_SEG}_markers/realtime.done"

# Per-profile hard caps. Batch (high-concurrency sweep, up to 10k prompts) is the
# long pole, so it gets a bigger deadline than the latency-bound realtime run.
REALTIME_DEADLINE=5400    # 1.5 h
BATCH_DEADLINE=14400      # 4 h

echo ""
echo "======================================================"
echo "  BENCHMARK JOB(S)"
echo "  Model    : ${MODEL}"
echo "  Hardware : ${HW}"
echo "  Manifest : ${MANIFEST}"
echo "  Profile  : ${PROFILE}"
echo "  Endpoint : ${ENDPOINT}"
echo "  Image    : ${BENCHMARK_RUNNER_IMAGE}"
echo "  Job base : ${JOB_NAME_BASE}"
echo "  Profiles : ${PROFILE_LIST[*]}  (one SEPARATE Job each, run sequentially)"
echo "  Tag      : ${TAG:-<none>}"
echo "  Results  : s3://${RESULTS_BUCKET}/results/${TIMESTAMP}/${S3_TAG_SEG}<profile>/"
echo "======================================================"
echo ""

cd "${FRAMEWORK_ROOT}"

# Handle custom dataset S3 sync (upload local dataset if missing from S3, else skip upload)
if [[ -n "${DATASET_OVERRIDE:-}" ]]; then
    DS_BASENAME=$(basename "${DATASET_OVERRIDE}")
    S3_KEY="datasets/${DS_BASENAME}"

    if aws s3 ls "s3://${RESULTS_BUCKET}/${S3_KEY}" --region "${AWS_REGION}" >/dev/null 2>&1; then
        log_info "Dataset found in S3: s3://${RESULTS_BUCKET}/${S3_KEY} (Skipping upload)."
    else
        LOCAL_FILE=""
        if [[ -f "${DATASET_OVERRIDE}" ]]; then
            LOCAL_FILE="${DATASET_OVERRIDE}"
        elif [[ -f "${FRAMEWORK_ROOT}/configs/workload_profiles/datasets/${DS_BASENAME}" ]]; then
            LOCAL_FILE="${FRAMEWORK_ROOT}/configs/workload_profiles/datasets/${DS_BASENAME}"
        fi

        if [[ -n "${LOCAL_FILE}" && -f "${LOCAL_FILE}" ]]; then
            log_info "Dataset not found in S3. Uploading '${LOCAL_FILE}' -> s3://${RESULTS_BUCKET}/${S3_KEY}..."
            aws s3 cp "${LOCAL_FILE}" "s3://${RESULTS_BUCKET}/${S3_KEY}" --region "${AWS_REGION}"
            log_info "Dataset upload complete."
        else
            log_warn "Dataset '${DATASET_OVERRIDE}' not found locally or in S3. Pod will attempt download."
        fi
    fi
    DATASET_OVERRIDE="${S3_KEY}"
fi

# ==============================================================================
# Step 1 — Create / update ConfigMaps (script + manifest + profile isolated per Job)
# ==============================================================================
log_info "Creating ConfigMaps..."

SCRIPT_CM="${JOB_NAME_BASE}-script"
MANIFEST_CM="${JOB_NAME_BASE}-manifest"

kubectl create configmap "${SCRIPT_CM}" \
    --from-file=load-test.py="${FRAMEWORK_ROOT}/inference/load-test.py" \
    -n "${BENCHMARK_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null

kubectl create configmap "${MANIFEST_CM}" \
    --from-file=manifest.yaml="${FRAMEWORK_ROOT}/${MANIFEST}" \
    -n "${BENCHMARK_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null

# One ConfigMap per profile, each holding its own profile.yaml. Mounted at
# /configs/profiles/profile.yaml inside that profile's own Job container.
for P in "${PROFILE_LIST[@]}"; do
    kubectl create configmap "${JOB_NAME_BASE}-profile-${P}" \
        --from-file=profile.yaml="${FRAMEWORK_ROOT}/configs/workload_profiles/${P}_v1.yaml" \
        -n "${BENCHMARK_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
done

if [[ -n "${DATASET_OVERRIDE:-}" ]]; then
    log_info "ConfigMaps ready. (Using custom dataset from S3: s3://${RESULTS_BUCKET}/${DATASET_OVERRIDE})"
else
    log_info "ConfigMaps ready. (Inputs come from vllm bench serve's built-in 'random' dataset)."
fi

DATASET_JOB_ARG=""
if [[ -n "${DATASET_OVERRIDE:-}" ]]; then
    DATASET_JOB_ARG="--dataset ${DATASET_OVERRIDE}"
fi

# ==============================================================================
# Step 2 — Submit ONE Job per profile (each its own k8s object, run sequentially)
#
# submit_profile_job <profile> — renders and applies a single-profile Job.
#   * realtime : runs immediately; on success writes an S3 "done" marker.
#   * batch    : an init container polls S3 for the realtime marker and blocks
#                until it appears, so batch never loads the GPU while realtime is
#                still measuring latency. If realtime is NOT in this run, the gate
#                is skipped so batch starts right away.
# Every Job runs from the vLLM DLC image, CPU-only, off the GPU node.
# ==============================================================================
submit_profile_job() {
    local P="$1"
    local JOB_NAME="${JOB_NAME_BASE}-${P}"
    local DEADLINE_S; local GATE_ON_REALTIME="false"
    if [[ "${P}" == "batch" ]]; then
        DEADLINE_S="${BATCH_DEADLINE}"
        # Only gate batch on realtime when realtime is actually part of this run.
        for _p in "${PROFILE_LIST[@]}"; do [[ "${_p}" == "realtime" ]] && GATE_ON_REALTIME="true"; done
    else
        DEADLINE_S="${REALTIME_DEADLINE}"
    fi

    # --- init container: S3 marker gate (batch only, when realtime is in the run) ---
    local INIT_CONTAINER=""
    if [[ "${GATE_ON_REALTIME}" == "true" ]]; then
        INIT_CONTAINER=$(cat <<INITEOF
      initContainers:
        - name: wait-for-realtime
          image: ${BENCHMARK_RUNNER_IMAGE}
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -uo pipefail
              export HOME=/tmp
              export PYTHONPATH="/tmp/pip-packages:\${PYTHONPATH:-}"
              python3 -c "import boto3" 2>/dev/null || \\
                pip install --quiet --no-cache-dir --target=/tmp/pip-packages boto3==1.34.0 || true
              echo "Gate: waiting for realtime completion marker s3://${RESULTS_BUCKET}/${S3_MARKER_KEY}"
              python3 -c "
              import boto3, os, sys, time
              s3 = boto3.client('s3', region_name='${AWS_REGION}')
              deadline = time.time() + ${REALTIME_DEADLINE} + 600
              while time.time() < deadline:
                  try:
                      s3.head_object(Bucket='${RESULTS_BUCKET}', Key='${S3_MARKER_KEY}')
                      print('Realtime marker found — starting batch.'); sys.exit(0)
                  except Exception:
                      print('...realtime not done yet; sleeping 20s'); time.sleep(20)
              print('Gate timed out waiting for realtime — starting batch anyway.'); sys.exit(0)
              "
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

    # --- main container: run load-test for THIS profile, upload, mark done ------
    # realtime writes the S3 marker after a successful (or SLO-only) run so the
    # gated batch Job can proceed. batch writes no marker.
    local MARK_STEP=""
    if [[ "${P}" == "realtime" ]]; then
        MARK_STEP=$(cat <<MARKEOF
              echo "=== Writing realtime completion marker ==="
              python3 -c "
              import boto3
              s3 = boto3.client('s3', region_name='${AWS_REGION}')
              s3.put_object(Bucket='${RESULTS_BUCKET}', Key='${S3_MARKER_KEY}', Body=b'done')
              print('Marker written: s3://${RESULTS_BUCKET}/${S3_MARKER_KEY}')
              " || echo "WARN: failed to write realtime marker (batch gate will time out and proceed)"
MARKEOF
)
    fi

    log_info "Submitting ${P} Job: ${JOB_NAME} (deadline ${DEADLINE_S}s, gate=${GATE_ON_REALTIME})..."

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
    profile: "${P}"
spec:
  backoffLimit: 0
  activeDeadlineSeconds: ${DEADLINE_S}
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ${JOB_NAME}
        app.kubernetes.io/component: benchmark-runner
        project: observeai-inference-optimization
        model: "${MODEL}"
        profile: "${P}"
      annotations:
        # Prevent Karpenter from consolidating/evicting the node mid-benchmark.
        # A single eviction is terminal here (backoffLimit: 0), so the whole
        # sweep is lost. This pins the node for the lifetime of the run.
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
      # Do NOT co-locate the runner on the GPU inference node (latency contamination).
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchLabels:
                  app.kubernetes.io/component: inference-server
              topologyKey: kubernetes.io/hostname
      tolerations: []
${INIT_CONTAINER}
      containers:
        - name: benchmark
          image: ${BENCHMARK_RUNNER_IMAGE}
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -uo pipefail
              export HOME=/tmp
              export PYTHONPATH="/tmp/pip-packages:\${PYTHONPATH:-}"
              # Install required runner packages if not already present in the image
              python3 -c "import boto3, vllm, yaml" 2>/dev/null || \\
                pip install --quiet --no-cache-dir --target=/tmp/pip-packages boto3==1.34.0 pyyaml vllm || true

              TEST_EXIT=0
              echo "=== PROFILE: ${P} ==="
              # Per-level S3 upload target — load-test.py uploads the cumulative
              # JSONL after EVERY concurrency level, so a failure at a high level
              # still leaves the earlier levels in S3.
              export RESULTS_S3_PREFIX="results/${TIMESTAMP}/${S3_TAG_SEG}${P}/"
              python3 /app/load-test.py \\
                --manifest /configs/manifests/manifest.yaml \\
                --profile  /configs/profiles/profile.yaml \\
                --endpoint "${ENDPOINT}" \\
                --output   /results/${P} \\
                --wait-timeout 600 ${DATASET_JOB_ARG} || TEST_EXIT=\$?
              echo "=== Uploading ${P} results to S3 ==="
              python3 -c "
              import boto3, os, glob
              s3 = boto3.client('s3', region_name='${AWS_REGION}')
              for f in glob.glob('/results/${P}/*.jsonl'):
                  key = 'results/${TIMESTAMP}/${S3_TAG_SEG}${P}/' + os.path.basename(f)
                  s3.upload_file(f, '${RESULTS_BUCKET}', key)
                  print('Uploaded: s3://${RESULTS_BUCKET}/' + key)
              " || true
${MARK_STEP}
              exit \${TEST_EXIT}
          env:
            - name: AWS_DEFAULT_REGION
              value: "${AWS_REGION}"
            - name: RESULTS_BUCKET
              value: "${RESULTS_BUCKET}"
            - name: DCGM_METRICS_URL
              value: "http://dcgm-exporter.monitoring.svc.cluster.local:9400/metrics"
            # Full deployed instance type (e.g. g7e.24xlarge). load-test.py records
            # this and looks up its $/hr from the EC2 price book — dynamic cost,
            # no per-instance manifest edits. Empty → falls back to the manifest.
            - name: OAI_INSTANCE_TYPE
              value: "${INSTANCE_TYPE}"
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
            - name: profile-${P}
              mountPath: /configs/profiles
              readOnly: true
            - name: results
              mountPath: /results
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: app-code
          configMap:
            name: ${SCRIPT_CM}
        - name: manifest
          configMap:
            name: ${MANIFEST_CM}
        - name: profile-${P}
          configMap:
            name: ${JOB_NAME_BASE}-profile-${P}
        - name: results
          emptyDir:
            sizeLimit: 1Gi
        - name: tmp
          emptyDir:
            sizeLimit: 4Gi
EOF
    log_info "${P} Job submitted: ${JOB_NAME}"
}

# Submit realtime FIRST so its marker-writing Job exists before batch's gate
# starts polling. Order within PROFILE_LIST already puts realtime before batch.
SUBMITTED_JOBS=()
for P in "${PROFILE_LIST[@]}"; do
    submit_profile_job "${P}"
    SUBMITTED_JOBS+=("${JOB_NAME_BASE}-${P}")
done

# ==============================================================================
# Step 3 — Detach. All Jobs are now running server-side; the terminal is free.
# The batch Job self-gates on realtime via the S3 marker, so ordering holds even
# if this script exits right now (Ctrl+C or closing the shell is safe).
# ==============================================================================
echo ""
echo "======================================================"
echo "  BENCHMARK JOB(S) SUBMITTED — running in-cluster"
echo "  Profiles : ${PROFILE_LIST[*]}  (separate Jobs, batch gated on realtime)"
for J in "${SUBMITTED_JOBS[@]}"; do
    echo "    • ${J}"
done
echo "  Results  : s3://${RESULTS_BUCKET}/results/${TIMESTAMP}/"
echo "======================================================"
echo ""
echo "  These Jobs run on the cluster independently of this terminal."
echo "  Closing the shell or pressing Ctrl+C does NOT stop them."
echo ""
echo "  Follow logs:"
for J in "${SUBMITTED_JOBS[@]}"; do
    echo "    kubectl logs -n ${BENCHMARK_NAMESPACE} -l job-name=${J} -f --tail=50"
done
echo ""
echo "  Job status:"
echo "    kubectl get jobs -n ${BENCHMARK_NAMESPACE} -l model=${MODEL} -w"
echo ""
echo "  Results (as they upload, per concurrency level):"
echo "    aws s3 ls s3://${RESULTS_BUCKET}/results/${TIMESTAMP}/ --recursive --region ${AWS_REGION}"
echo "    aws s3 cp s3://${RESULTS_BUCKET}/results/${TIMESTAMP}/ results/ --recursive --region ${AWS_REGION}"
echo ""
echo "  Grafana  : kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring"
echo ""

# Optional best-effort log tail of the FIRST (realtime) Job so an attended run
# still sees live output. Detaching here (Ctrl+C) leaves all Jobs running.
FIRST_JOB="${SUBMITTED_JOBS[0]}"
log_info "Tailing ${FIRST_JOB} (Ctrl+C detaches — Jobs keep running in cluster)..."
for i in $(seq 1 30); do
    POD=$(kubectl get pods -l "job-name=${FIRST_JOB}" \
        -n "${BENCHMARK_NAMESPACE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    [[ -n "${POD}" ]] && break
    echo -n "."
    sleep 5
done
echo ""
if [[ -n "${POD:-}" ]]; then
    kubectl logs "${POD}" -n "${BENCHMARK_NAMESPACE}" -f 2>/dev/null || true
fi

exit 0
