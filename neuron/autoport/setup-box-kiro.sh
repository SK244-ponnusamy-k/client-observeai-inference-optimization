#!/bin/bash
# ==============================================================================
# neuron/autoport/setup-box-kiro.sh
#
# One-time setup ON an AutoPort box (launched by launch-box.sh), as the ubuntu
# user. Identical to setup-box.sh EXCEPT step [5/5] installs the Kiro CLI
# instead of Claude Code, so the AutoPort port is driven by `kiro-cli`.
#
#   1. checks the Neuron devices are visible
#   2. finds the DLAMI venv that has vllm_neuron and makes it the default
#   3. checks out the matching vllm-neuron source as the AutoPort workspace and
#      installs it editable (AutoPort writes vllm_neuron/model/<name>/ there)
#   4. syncs the HF weights from S3 (no HuggingFace token needed)
#   5. installs the Kiro CLI + Neuron agentic development package
#
# Kiro CLI (https://kiro.dev/cli/) is installed the same way as Claude Code
# (curl | bash). After install the binary is `kiro-cli`:
#   - interactive : `kiro-cli login` opens a browser to authenticate, then
#                   `kiro-cli chat` starts an interactive session.
#   - headless    : set KIRO_API_KEY (generated at https://app.kiro.dev/) and
#                   Kiro skips the browser login; run prompts non-interactively
#                   with `kiro-cli chat --no-interactive "<prompt>"`.
#
# NOTE on skills: the `deploy-neuron-agentic-development-to-claude` helper that
# setup-box.sh runs installs the AWS Neuron agentic-development SKILLS into
# Claude's layout (~/.claude). That helper is Claude-specific; there is no Kiro
# equivalent shipped by the neuron-agentic-development package today. This script
# still installs the neuron-agentic-development package (the docs/skills payload)
# so it is available on the box, but does NOT run the Claude deploy step.
#
# Usage (idempotent):
#   bash setup-box-kiro.sh --model-name gemma4 --weights-folder gemma-4-31B-it [--bucket B] [--region R]
# ==============================================================================

set -euo pipefail

MODEL_NAME=""; WEIGHTS_FOLDER=""
BUCKET="${MODEL_BUCKET:-shellkode-ai-models}"; REGION="${AWS_REGION:-us-east-2}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name)     MODEL_NAME="$2"; shift 2 ;;
        --weights-folder) WEIGHTS_FOLDER="$2"; shift 2 ;;
        --bucket)         BUCKET="$2"; shift 2 ;;
        --region)         REGION="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "${MODEL_NAME}" && -n "${WEIGHTS_FOLDER}" ]] || { echo "--model-name and --weights-folder are required"; exit 1; }

WORK="${HOME}/autoport/${MODEL_NAME}"
WEIGHTS="${HOME}/models/${WEIGHTS_FOLDER}"
step() { echo ""; echo "==> $*"; }

step "[1/5] Neuron devices"
DEVICES=$(neuron-ls --json-output 2>/dev/null | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))' || echo 0)
[[ "${DEVICES}" -gt 0 ]] || { echo "ERROR: no Neuron devices (neuron-ls). Check the instance type / aws-neuronx-dkms."; exit 1; }
echo "${DEVICES} Neuron devices"

step "[2/5] DLAMI venv with vllm_neuron"
VENV=""
for candidate in /opt/aws_neuronx_venv_* "${HOME}"/opt/aws_neuronx_venv_*; do
    [[ -x "${candidate}/bin/python" ]] || continue
    if "${candidate}/bin/python" -c "import vllm_neuron" 2>/dev/null; then VENV="${candidate}"; break; fi
done
[[ -n "${VENV}" ]] || { echo "ERROR: no venv with vllm_neuron under /opt - is this the vLLM DLAMI?"; exit 1; }
PY="${VENV}/bin/python"
PIP=("${VENV}/bin/pip")
[[ -w "${VENV}/lib" ]] || PIP=(sudo "${VENV}/bin/pip")
INSTALLED=$("${PY}" -c "from importlib.metadata import version; print(version('vllm-neuron'))" 2>/dev/null | tail -n1 || echo unknown)
echo "venv ${VENV}  (vllm-neuron ${INSTALLED}, vllm $("${PY}" -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -n1 || echo '?'))"
grep -q "${VENV}/bin/activate" "${HOME}/.bashrc" || echo "source ${VENV}/bin/activate" >> "${HOME}/.bashrc"

step "[3/5] vllm-neuron source workspace (${WORK})"
REF="release-${INSTALLED}"
mkdir -p "${WORK}"
if [[ ! -d "${WORK}/vllm-neuron/.git" ]]; then
    git clone --branch "${REF}" https://github.com/vllm-project/vllm-neuron.git "${WORK}/vllm-neuron" || {
        echo "ERROR: no branch ${REF} upstream - clone the vllm-neuron release that matches ${INSTALLED} by hand."; exit 1; }
fi
# Editable + --no-deps: the DLAMI's pinned torch/vllm stay exactly as shipped.
"${PIP[@]}" install --quiet --no-deps -e "${WORK}/vllm-neuron"
# Capture ONLY the last stdout line - importing vllm_neuron emits log noise
# (PJRT_DEVICE warning, plugin-activation INFO, port_hold_patch) to stdout, which
# otherwise pollutes the captured path and breaks this assertion.
LOADED=$("${PY}" -c "import vllm_neuron, os; print(os.path.dirname(vllm_neuron.__file__))" 2>/dev/null | tail -n1)
[[ "${LOADED}" == "${WORK}/vllm-neuron/"* ]] || { echo "ERROR: vllm_neuron still loads from ${LOADED}"; exit 1; }
git -C "${WORK}/vllm-neuron" rev-parse HEAD > "${WORK}/BASE_COMMIT"
echo "vllm_neuron now loads from ${LOADED} (base $(cut -c1-12 "${WORK}/BASE_COMMIT"))"

step "[4/5] Weights s3://${BUCKET}/${WEIGHTS_FOLDER}/ -> ${WEIGHTS}"
aws s3 sync "s3://${BUCKET}/${WEIGHTS_FOLDER}/" "${WEIGHTS}/" --region "${REGION}" --only-show-errors
test -f "${WEIGHTS}/config.json" || { echo "ERROR: config.json missing in ${WEIGHTS}"; exit 1; }
du -sh "${WEIGHTS}"
ARCH=$(python3 -c "import json; d=json.load(open('${WEIGHTS}/config.json')); print(d.get('model_type'), d.get('architectures'))")
echo "config: ${ARCH}"
"${PY}" -c "
import json, transformers
mt = json.load(open('${WEIGHTS}/config.json')).get('model_type', '')
try:
    __import__(f'transformers.models.{mt}')
    print(f'transformers {transformers.__version__} has {mt} (CPU reference available)')
except ImportError:
    print(f'WARN: transformers {transformers.__version__} has no {mt} - the CPU reference for logit validation needs a newer transformers; AutoPort will ask.')
" 2>/dev/null | tail -n1

step "[5/5] Kiro CLI + Neuron agentic development package"
if ! command -v kiro-cli >/dev/null 2>&1 && [[ ! -x "${HOME}/.local/bin/kiro-cli" ]]; then
    curl -fsSL https://cli.kiro.dev/install | bash
fi
grep -q 'HOME/.local/bin' "${HOME}/.bashrc" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${HOME}/.bashrc"
# Install the Neuron agentic-development package (docs/skills payload). Unlike the
# Claude flow, there is no `deploy-...-to-kiro` helper, so we do not deploy skills
# into a Kiro-specific layout here.
"${PIP[@]}" install --quiet --upgrade neuron-agentic-development --extra-index-url https://pip.repos.neuron.amazonaws.com

echo ""
echo "=============================================================="
echo "  AutoPort box ready (Kiro CLI)."
echo "  Workspace : ${WORK}/vllm-neuron   (vllm_neuron editable install)"
echo "  Weights   : ${WEIGHTS}"
echo ""
echo "  Next (interactive):"
echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "    cd ${WORK}/vllm-neuron"
echo "    kiro-cli login          # browser auth, once"
echo "    kiro-cli chat           # then paste the prompt"
echo ""
echo "  Or headless (no browser):"
echo "    export KIRO_API_KEY=...   # from https://app.kiro.dev/ account settings"
echo "    kiro-cli chat --no-interactive \"<prompt>\""
echo "=============================================================="
