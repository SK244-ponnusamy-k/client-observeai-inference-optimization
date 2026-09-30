#!/bin/bash
# ==============================================================================
# inference/run-autostop.sh
#
# Submit an in-cluster "auto-stop" cleanup Job that waits for a deploy pipeline
# (realtime -> batch -> quality) to finish, then stops the model by deleting its
# vLLM Deployment / Service / PVC. Once the pod is gone the GPU node goes idle
# and Karpenter reaps it, so billing stops.
#
# Submitted by `oai deploy --auto-stop`. Because it runs entirely in the cluster,
# it works even after the user closes their terminal (the whole point).
#
# Two wait modes:
#   --wait-mode quality-jobs  : wait until the quality Job(s) for this model+tag
#                               reach a terminal state (Complete or Failed).
#                               Used when the pipeline ends with quality.
#   --wait-mode marker        : wait for an S3 completion marker (the benchmark's
#                               final-profile .done key). Used when only
#                               --benchmark was requested (no quality).
#
# Usage (normally invoked by oai, not by hand):
#   bash inference/run-autostop.sh --model qwen3-5-9b-tp1 --tag g6e \
#     --run-timestamp 20260928-105426 --wait-mode quality-jobs \
#     --quality-sheet org --quality-sheet xl
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/config/config.env"

: "${RESULTS_BUCKET:?ERROR: RESULTS_BUCKET not set in config/config.env.}"
: "${BENCHMARK_NAMESPACE:?ERROR: BENCHMARK_NAMESPACE not set in config/config.env.}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]  $(date +'%H:%M:%S')${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]  $(date +'%H:%M:%S')${NC} $1"; }
log_error() { echo -e "${RED}[ERROR] $(date +'%H:%M:%S')${NC} $1"; }

MODEL=""
TAG=""
RUN_TIMESTAMP=""
WAIT_MODE="quality-jobs"
WAIT_FOR_MARKER=""
QUALITY_SHEETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)           MODEL="$2"; shift 2 ;;
        --tag)             TAG="$2"; shift 2 ;;
        --run-timestamp)   RUN_TIMESTAMP="$2"; shift 2 ;;
        --wait-mode)       WAIT_MODE="$2"; shift 2 ;;
        --wait-for-marker) WAIT_FOR_MARKER="$2"; shift 2 ;;
        --quality-sheet)   QUALITY_SHEETS+=("$2"); shift 2 ;;
        *) log_warn "Unknown arg: $1"; shift ;;
    esac
done

: "${MODEL:?ERROR: --model is required.}"

# Kubernetes copies a Job's metadata.name into the auto-injected pod label
# batch.kubernetes.io/job-name (value limit 63 bytes). Keep every name <=63 with
# a stable checksum suffix. Mirrors run-benchmark.sh / run-quality.sh.
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

MODEL_ID_DASHED="${MODEL//\./-}"
TAG_SEG=""
[[ -n "${TAG}" ]] && TAG_SEG="-${TAG}"

# Names the auto-stop Job will delete (must match deploy.sh / stop.sh naming).
DEPLOYMENT_NAME="oai-infopt-vllm-${MODEL_ID_DASHED}${TAG_SEG}"
SERVICE_NAME="oai-infopt-vllm-${MODEL_ID_DASHED}${TAG_SEG}"
PVC_NAME="oai-infopt-metadata-${MODEL_ID_DASHED}${TAG_SEG}"

JOB_NAME=$(k8s_name "oai-infopt-autostop-${MODEL_ID_DASHED}${TAG_SEG}-${RUN_TIMESTAMP}")
JOB_LABEL="oai-infopt-autostop-runner"

# Label selector to find the quality Job(s) for this model+tag.
#
# IMPORTANT: run-quality.sh labels each quality Job with run-tag=<base>-<sheet>
# (e.g. "astest3-org", "astest3-xl") because ORG and XL are separate Jobs that
# must land under separate S3 prefixes. The auto-stop only knows the BASE tag
# (e.g. "astest3"), so an exact run-tag= match NEVER hits -> the counter stayed
# 0/0 forever and the model was never stopped. We therefore select on
# component+model only, then filter returned Jobs by a run-tag PREFIX equal to
# the base tag (label selectors cannot do prefix matching themselves).
RUN_TAG_BASE="${TAG:-untagged}"
QUALITY_SELECTOR="app.kubernetes.io/component=quality-runner,model=${MODEL}"

# How many quality Jobs to expect (one per sheet). Default: both sheets.
EXPECTED_QUALITY_JOBS="${#QUALITY_SHEETS[@]}"
[[ "${EXPECTED_QUALITY_JOBS}" -eq 0 ]] && EXPECTED_QUALITY_JOBS=2

# Overall safety cap so the cleanup Job can never hang forever (13h pipeline
# envelope + 1h headroom). Raised in step with the larger realtime (3h) + batch
# (6h) benchmark caps for reasoning-ON on large/slow models. If the pipeline
# never finishes, the Job stops the model anyway at this deadline to avoid a
# stuck billing node.
JOB_DEADLINE=50400   # 14h

echo ""
echo "======================================================"
echo "  AUTO-STOP CLEANUP JOB"
echo "  Model        : ${MODEL} ${TAG:+(tag=${TAG})}"
echo "  Stops        : deployment/${DEPLOYMENT_NAME}"
echo "  Wait mode    : ${WAIT_MODE}"
if [[ "${WAIT_MODE}" == "marker" ]]; then
    echo "  Waits for    : s3://${RESULTS_BUCKET}/${WAIT_FOR_MARKER}"
else
    echo "  Waits for    : ${EXPECTED_QUALITY_JOBS} quality Job(s) matching [${QUALITY_SELECTOR}]"
fi
echo "  Job          : ${JOB_NAME}"
echo "  Namespace    : ${BENCHMARK_NAMESPACE}"
echo "======================================================"
echo ""

# Ensure the RBAC exists (idempotent). Without it the Job cannot delete the
# Deployment. Apply is safe to re-run.
if ! kubectl get serviceaccount oai-infopt-autostop-sa -n "${BENCHMARK_NAMESPACE}" >/dev/null 2>&1; then
    log_info "Applying auto-stop RBAC (ServiceAccount + Role + RoleBinding)..."
    kubectl apply -f "${FRAMEWORK_ROOT}/cluster/autostop-rbac.yaml" >/dev/null
fi

# Build the wait step for the main container based on the mode.
if [[ "${WAIT_MODE}" == "marker" ]]; then
    : "${WAIT_FOR_MARKER:?ERROR: --wait-for-marker is required for --wait-mode marker.}"
    WAIT_STEP=$(cat <<WAITEOF
              echo "Waiting for benchmark completion marker s3://${RESULTS_BUCKET}/${WAIT_FOR_MARKER}"
              # This python block MUST exit 0 only when the marker is genuinely
              # found (or the deadline is hit). ANY other failure - notably
              # NoCredentialsError when the pod has no AWS identity - must exit
              # non-zero so the bash guard below SKIPS the stop. Previously an
              # uncaught boto3 error crashed python but bash (no set -e) fell
              # through and deleted the model before the benchmark even ran.
              python3 -c "
              import sys, time
              try:
                  import boto3
                  from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError
              except Exception as exc:
                  print('ERROR: boto3 import failed:', exc); sys.exit(3)
              try:
                  s3 = boto3.client('s3', region_name='${AWS_REGION}')
              except Exception as exc:
                  print('ERROR: could not create S3 client:', exc); sys.exit(3)
              deadline = time.time() + ${JOB_DEADLINE}
              while time.time() < deadline:
                  try:
                      s3.get_object(Bucket='${RESULTS_BUCKET}', Key='${WAIT_FOR_MARKER}')
                      print('Benchmark marker found - pipeline complete.'); sys.exit(0)
                  except NoCredentialsError:
                      print('ERROR: no AWS credentials in auto-stop pod; cannot poll S3 marker.')
                      print('Refusing to stop the model blindly. Stop manually: oai stop ${MODEL} ${TAG:+--tag ${TAG}}')
                      sys.exit(4)
                  except ClientError as exc:
                      code = str(exc.response.get('Error', {}).get('Code', ''))
                      if code in ('404', 'NoSuchKey', 'NotFound'):
                          print('...pipeline not done; sleeping 30s'); time.sleep(30); continue
                      if code in ('403', 'AccessDenied'):
                          print('ERROR: access denied reading S3 marker; refusing to stop blindly.'); sys.exit(4)
                      print('ERROR: unexpected S3 error:', code); sys.exit(4)
                  except BotoCoreError as exc:
                      print('ERROR: boto core error polling S3:', exc); sys.exit(4)
              print('Auto-stop deadline reached before marker; stopping anyway.'); sys.exit(0)
              " || { echo "Marker wait failed (exit \$?); NOT stopping the model. Stop manually if needed."; exit 1; }
WAITEOF
)
else
    # quality-jobs mode: wait until the expected number of quality Jobs for this
    # model+tag reach a terminal state (Complete or Failed). kubectl-based.
    WAIT_STEP=$(cat <<WAITEOF
              echo "Waiting for ${EXPECTED_QUALITY_JOBS} quality Job(s) [${QUALITY_SELECTOR}] to finish..."
              deadline=\$(( \$(date +%s) + ${JOB_DEADLINE} ))
              while [[ \$(date +%s) -lt \${deadline} ]]; do
                  # A quality Job is "finished" when it has a Complete or Failed condition.
                  # NOTE: 'grep -c' already prints 0 (and exits 1) when there are
                  # no matches; a '|| echo 0' would append a SECOND 0, producing a
                  # multi-line "0\n0" that breaks the numeric [[ ]] test below.
                  # Sanitize to a single integer with tr/head instead.
                  # Emit one line per quality Job: "<run-tag> <complete-status> <failed-status>".
                  # We then keep only rows whose run-tag equals the base tag OR
                  # starts with "<base>-" (the per-sheet suffix, e.g. astest3-org).
                  # This is the prefix match that a bare label selector can't do,
                  # and it is why the old exact run-tag=<base> selector saw 0/0.
                  rows=\$(kubectl get jobs -n "${BENCHMARK_NAMESPACE}" -l "${QUALITY_SELECTOR}" \\
                      -o jsonpath='{range .items[*]}{.metadata.labels.run-tag}{" "}{.status.conditions[?(@.type=="Complete")].status}{" "}{.status.conditions[?(@.type=="Failed")].status}{"\n"}{end}' 2>/dev/null)
                  total=\$(printf '%s\n' "\${rows}" | awk -v b="${RUN_TAG_BASE}" 'NF && (\$1==b || index(\$1, b"-")==1)' | grep -c .)
                  total=\${total:-0}
                  finished=\$(printf '%s\n' "\${rows}" | awk -v b="${RUN_TAG_BASE}" 'NF && (\$1==b || index(\$1, b"-")==1) && (\$2=="True" || \$3=="True")' | grep -c .)
                  finished=\${finished:-0}
                  echo "...quality Jobs finished \${finished}/\${total} (base-tag=${RUN_TAG_BASE}, expected ${EXPECTED_QUALITY_JOBS}); sleeping 30s"
                  if [[ "\${total}" -ge "${EXPECTED_QUALITY_JOBS}" && "\${finished}" -ge "\${total}" && "\${total}" -gt 0 ]]; then
                      echo "All quality Jobs finished - pipeline complete."; break
                  fi
                  sleep 30
              done
WAITEOF
)
fi

log_info "Submitting auto-stop Job: ${JOB_NAME}..."
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${BENCHMARK_NAMESPACE}
  labels:
    app.kubernetes.io/name: ${JOB_LABEL}
    app.kubernetes.io/component: auto-stop-runner
    project: observeai-inference-optimization
    model: "${MODEL}"
    run-tag: "${RUN_TAG_BASE}"
spec:
  backoffLimit: 1
  activeDeadlineSeconds: $((JOB_DEADLINE + 600))
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ${JOB_LABEL}
        app.kubernetes.io/component: auto-stop-runner
        project: observeai-inference-optimization
        model: "${MODEL}"
        run-tag: "${RUN_TAG_BASE}"
    spec:
      restartPolicy: Never
      serviceAccountName: oai-infopt-autostop-sa
      # This Job DOES need its API token (to watch Jobs + delete the Deployment).
      automountServiceAccountToken: true
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
      # Must NOT land on the GPU node it is about to free.
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchLabels:
                  app.kubernetes.io/component: inference-server
              topologyKey: kubernetes.io/hostname
      containers:
        - name: autostop
          image: ${BENCHMARK_RUNNER_IMAGE:-python:3.11-slim}
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -uo pipefail
              export HOME=/tmp
              export PYTHONPATH="/tmp/pip-packages:\${PYTHONPATH:-}"

              # kubectl (for job watch + delete) and boto3 (for marker mode).
              # The container image (vLLM DLC) has neither kubectl nor curl, so
              # download kubectl using whatever IS present: curl, wget, or python.
              # Without this the kubectl calls silently fail and the job counter
              # stays 0/0 forever (auto-stop never fires).
              KVER="v1.30.0"
              KURL="https://dl.k8s.io/release/\${KVER}/bin/linux/amd64/kubectl"
              if ! command -v kubectl >/dev/null 2>&1; then
                echo "kubectl not found - downloading to /tmp..."
                if command -v curl >/dev/null 2>&1; then
                  curl -sfL "\${KURL}" -o /tmp/kubectl
                elif command -v wget >/dev/null 2>&1; then
                  wget -q "\${KURL}" -O /tmp/kubectl
                else
                  echo "curl/wget missing - fetching kubectl via python urllib..."
                  python3 -c "import urllib.request; urllib.request.urlretrieve('\${KURL}', '/tmp/kubectl')"
                fi
                chmod +x /tmp/kubectl && export PATH="/tmp:\${PATH}"
              fi
              # Fail loudly if kubectl still isn't usable, so the job doesn't spin
              # at 0/0 pretending to wait. It errors out and the model can be
              # stopped manually instead of billing forever.
              if ! kubectl version --client >/dev/null 2>&1; then
                echo "ERROR: kubectl unavailable after install attempts; cannot watch jobs."
                echo "Stop the model manually: oai stop ${MODEL} ${TAG:+--tag ${TAG}}"
                exit 1
              fi
              python3 -c "import boto3" 2>/dev/null || \
                pip install --quiet --no-cache-dir --target=/tmp/pip-packages boto3==1.34.0 || true

              # ---- Wait for the pipeline to finish ----
${WAIT_STEP}

              # ---- Stop the model: delete Deployment / Service / PVC ----
              echo "Stopping model: deleting deployment/${DEPLOYMENT_NAME}, svc/${SERVICE_NAME}, pvc/${PVC_NAME}"
              kubectl delete deployment "${DEPLOYMENT_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true || true
              kubectl delete svc "${SERVICE_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true || true
              kubectl delete pvc "${PVC_NAME}" -n "${BENCHMARK_NAMESPACE}" --ignore-not-found=true || true
              echo "Auto-stop complete. The GPU node will be reclaimed by Karpenter once idle."
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
      volumes:
        - name: tmp
          emptyDir:
            sizeLimit: 1Gi
EOF

log_info "Auto-stop Job submitted: ${JOB_NAME}"
echo "  Status : kubectl get job ${JOB_NAME} -n ${BENCHMARK_NAMESPACE} -w"
echo "  Logs   : kubectl logs -n ${BENCHMARK_NAMESPACE} -l job-name=${JOB_NAME} -f --tail=100"
