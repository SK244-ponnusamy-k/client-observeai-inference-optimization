#!/bin/bash
# ==============================================================================
# neuron/autoport/backup-workspace.sh
#
# Snapshots an AutoPort workspace to S3. Runs from cron ON the AutoPort box: the
# box's disk is deleted when its Capacity Block ends, and the port lives only
# there until it is handed off.
#
# Each run saves every change against the base vllm-neuron commit - edited AND
# new files - as:
#   workspace.patch   git apply-able onto BASE_COMMIT (binary-safe)
#   files.tgz         the changed/new files themselves, for browsing
#   diffstat.txt, MANIFEST.json
# to <dest>/latest/ and <dest>/history/<UTC timestamp>/. Unchanged snapshots
# are skipped. The agent's own git index is never touched (throwaway index).
#
# Usage:
#   backup-workspace.sh --workspace ~/autoport/gemma4 --dest s3://bucket/prefix [--only-after ISO8601]
#   --only-after  do nothing before this UTC time (for denser runs near the deadline)
# ==============================================================================

set -euo pipefail

WORKSPACE=""; DEST=""; ONLY_AFTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace)  WORKSPACE="$2"; shift 2 ;;
        --dest)       DEST="${2%/}"; shift 2 ;;
        --only-after) ONLY_AFTER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "${WORKSPACE}" && -n "${DEST}" ]] || { echo "--workspace and --dest are required"; exit 1; }
if [[ -n "${ONLY_AFTER}" && "$(date -u +%s)" -lt "$(date -u -d "${ONLY_AFTER}" +%s)" ]]; then
    exit 0
fi

REPO="${WORKSPACE}/vllm-neuron"
BASE="$(cat "${WORKSPACE}/BASE_COMMIT")"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# Stage everything into a throwaway index (starting from the real one) so new,
# untracked files - the whole ported model package - are part of the diff.
export GIT_INDEX_FILE="${TMP}/index"
cp "${REPO}/.git/index" "${GIT_INDEX_FILE}"
EXCLUDES=(':(exclude,glob)**/__pycache__/**' ':(exclude,glob)**/*.pyc' ':(exclude,glob)*.egg-info/**')
git -C "${REPO}" add -A -- . "${EXCLUDES[@]}"
git -C "${REPO}" diff --cached --binary "${BASE}" > "${TMP}/workspace.patch"
git -C "${REPO}" diff --cached --stat "${BASE}" > "${TMP}/diffstat.txt"
git -C "${REPO}" diff --cached --name-only --diff-filter=d "${BASE}" > "${TMP}/files.txt"
unset GIT_INDEX_FILE

SHA="$(sha256sum "${TMP}/workspace.patch" | cut -c1-64)"
LAST_FILE="${WORKSPACE}/.last-backup-sha"
if [[ -f "${LAST_FILE}" && "$(cat "${LAST_FILE}")" == "${SHA}" ]]; then
    echo "${TS} unchanged (${SHA:0:12}) - skipped"
    exit 0
fi

tar -C "${REPO}" -czf "${TMP}/files.tgz" -T "${TMP}/files.txt"
IMDS_TOKEN="$(curl -sf -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' || true)"
INSTANCE_ID="$(curl -sf -H "X-aws-ec2-metadata-token: ${IMDS_TOKEN}" http://169.254.169.254/latest/meta-data/instance-id || hostname)"
cat > "${TMP}/MANIFEST.json" <<EOF
{
  "saved_at": "${TS}",
  "instance_id": "${INSTANCE_ID}",
  "workspace": "${REPO}",
  "base_commit": "${BASE}",
  "head_commit": "$(git -C "${REPO}" rev-parse HEAD)",
  "files_changed": $(wc -l < "${TMP}/files.txt"),
  "patch_sha256": "${SHA}",
  "restore": "git clone https://github.com/vllm-project/vllm-neuron && git checkout ${BASE} && git apply workspace.patch"
}
EOF

rm -f "${TMP}/files.txt" "${TMP}/index"
aws s3 cp --recursive --only-show-errors "${TMP}/" "${DEST}/history/${TS}/"
aws s3 sync --delete --only-show-errors "${TMP}/" "${DEST}/latest/"
echo "${SHA}" > "${LAST_FILE}"
echo "${TS} saved $(wc -l < "${TMP}/diffstat.txt" | tr -d ' ') diffstat lines (${SHA:0:12}) -> ${DEST}/latest/"
