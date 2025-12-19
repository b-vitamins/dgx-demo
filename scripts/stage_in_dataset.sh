#!/usr/bin/env bash
set -euo pipefail

# Template for large datasets (stage-in to fast scratch).
# Replace PERSISTENT_DATA with wherever your real dataset lives (project storage, shared FS, etc).

PERSISTENT_DATA="${1:-/path/to/your/persistent/dataset}"

USER_NAME="${USER:-$(id -un)}"

DEFAULT_SCRATCH_ROOT="/localscratch/${USER_NAME}"
if [[ -d "/raid/${USER_NAME}" ]]; then
  DEFAULT_SCRATCH_ROOT="/raid/${USER_NAME}"
elif [[ -d "/localscratch/${USER_NAME}" ]]; then
  DEFAULT_SCRATCH_ROOT="/localscratch/${USER_NAME}"
fi

SCRATCH_DATA="${2:-${DEFAULT_SCRATCH_ROOT}/datasets/mydataset}"

mkdir -p "${SCRATCH_DATA}"

echo "[stage-in] from: ${PERSISTENT_DATA}"
echo "[stage-in] to:   ${SCRATCH_DATA}"

# rsync is the sane default. Use --delete only if you're sure.
rsync -a --info=progress2 "${PERSISTENT_DATA}/" "${SCRATCH_DATA}/"
echo "[stage-in] done"
