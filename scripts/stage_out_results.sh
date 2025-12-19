#!/usr/bin/env bash
set -euo pipefail

# Template for copying results out of scratch before the purge removes them.

USER_NAME="${USER:-$(id -un)}"

DEFAULT_SCRATCH_ROOT="/localscratch/${USER_NAME}"
if [[ -d "/raid/${USER_NAME}" ]]; then
  DEFAULT_SCRATCH_ROOT="/raid/${USER_NAME}"
elif [[ -d "/localscratch/${USER_NAME}" ]]; then
  DEFAULT_SCRATCH_ROOT="/localscratch/${USER_NAME}"
fi

RUN_DIR="${1:-${DEFAULT_SCRATCH_ROOT}/dgx-demo/runs}"
DEST="${2:-/path/to/your/persistent/results}"

mkdir -p "${DEST}"
echo "[stage-out] from: ${RUN_DIR}"
echo "[stage-out] to:   ${DEST}"

rsync -a --info=progress2 "${RUN_DIR}/" "${DEST}/"
echo "[stage-out] done"
