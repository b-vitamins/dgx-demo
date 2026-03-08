#!/usr/bin/env bash
set -euo pipefail

echo "[sanity] user:"
id

echo "[sanity] working directory:"
pwd

echo "[sanity] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[sanity] visible cpu count: $(getconf _NPROCESSORS_ONLN)"

tmp_file=".sanity_write_test_${SLURM_JOB_ID:-$$}"
printf 'ok\n' > "${tmp_file}"
rm -f "${tmp_file}"
echo "[sanity] write test: ok"

echo "[sanity] python:"
python -V

echo "[sanity] torch:"
python - <<'PY'
import os
import sys
import torch
print("python executable:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("cuda visible devices env:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    print("gpu name:", torch.cuda.get_device_name(0))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[sanity] nvidia-smi -L:"
  nvidia-smi -L || true
fi
