#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

USER_NAME="${USER:-$(id -un)}"
IMAGE="${USER_NAME}/dgx-demo:torch"
CLUSTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cluster)
      CLUSTER="${2:-}"
      shift 2
      ;;
    --image)
      IMAGE="${2:-}"
      shift 2
      ;;
    *)
      echo "[preflight] unknown argument: $1" >&2
      echo "usage: $0 [--cluster dgx1|dgxh100] [--image <docker-image>]" >&2
      exit 2
      ;;
  esac
done

HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"

detect_cluster() {
  if [[ -n "${CLUSTER}" ]]; then
    echo "${CLUSTER}"
    return
  fi

  if [[ "${HOST_SHORT}" == dgxh100* ]] || [[ -d "/raid/${USER_NAME}" ]]; then
    echo "dgxh100"
    return
  fi

  if [[ "${HOST_SHORT}" == nvidia-dgx* ]] || [[ -d "/localscratch/${USER_NAME}" ]]; then
    echo "dgx1"
    return
  fi

  echo "unknown"
}

CLUSTER="$(detect_cluster)"

case "${CLUSTER}" in
  dgx1)
    SCRATCH_ROOT="/localscratch/${USER_NAME}"
    SLURM_DIR="slurm/dgx1"
    REQUIRED_QUEUES=("q_1day-1G" "q_1day-4G")
    ;;
  dgxh100)
    SCRATCH_ROOT="/raid/${USER_NAME}"
    SLURM_DIR="slurm/dgxh100"
    REQUIRED_QUEUES=("q_12hour-1G" "q_1day-2G")
    ;;
  *)
    SCRATCH_ROOT=""
    SLURM_DIR=""
    REQUIRED_QUEUES=()
    ;;
esac

failures=0
warnings=0

pass() {
  echo "[preflight] PASS: $1"
}

warn() {
  echo "[preflight] WARN: $1"
  warnings=$((warnings + 1))
}

fail() {
  echo "[preflight] FAIL: $1"
  failures=$((failures + 1))
}

check_cmd() {
  local cmd="$1"
  if command -v "${cmd}" >/dev/null 2>&1; then
    pass "found ${cmd}: $(command -v "${cmd}")"
  else
    fail "missing ${cmd}"
  fi
}

echo "[preflight] host=${HOST_SHORT}"
echo "[preflight] repo=${REPO_ROOT}"
echo "[preflight] cluster=${CLUSTER}"
echo "[preflight] image=${IMAGE}"

if [[ "${CLUSTER}" == "unknown" ]]; then
  fail "could not detect cluster from hostname or scratch paths; rerun with --cluster dgx1 or --cluster dgxh100"
else
  pass "using ${SLURM_DIR} and scratch root ${SCRATCH_ROOT}"
fi

for cmd in sbatch squeue sinfo docker; do
  check_cmd "${cmd}"
done

if [[ -n "${SCRATCH_ROOT}" ]]; then
  if [[ -d "${SCRATCH_ROOT}" ]]; then
    pass "scratch root exists: ${SCRATCH_ROOT}"
  else
    fail "scratch root not found: ${SCRATCH_ROOT}"
  fi

  case "${REPO_ROOT}" in
    "${SCRATCH_ROOT}"/*)
      pass "repo is on scratch: ${REPO_ROOT}"
      ;;
    *)
      warn "repo is not under ${SCRATCH_ROOT}; SERC guidance is to work from scratch"
      ;;
  esac
fi

if [[ -n "${SLURM_DIR}" ]]; then
  if [[ -d "${REPO_ROOT}/${SLURM_DIR}" ]]; then
    pass "cluster scripts present: ${SLURM_DIR}"
  else
    fail "missing cluster script directory: ${SLURM_DIR}"
  fi
fi

if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  pass "docker image exists: ${IMAGE}"
else
  warn "docker image not found: ${IMAGE}"
fi

if command -v sinfo >/dev/null 2>&1; then
  PARTITIONS="$(sinfo -h -o '%P' 2>/dev/null | tr ',' '\n' | sed 's/*$//' | sort -u || true)"
  if [[ -n "${PARTITIONS}" ]]; then
    pass "queried partitions with sinfo"
    for queue in "${REQUIRED_QUEUES[@]}"; do
      if printf '%s\n' "${PARTITIONS}" | grep -qx "${queue}"; then
        pass "partition available: ${queue}"
      else
        warn "expected partition not visible in sinfo: ${queue}"
      fi
    done
  else
    warn "sinfo returned no partitions; are you on the login node?"
  fi
fi

if [[ -n "${SLURM_DIR}" ]]; then
  echo "[preflight] next step: sbatch ${SLURM_DIR}/00_test_container_1gpu.sbatch"
fi

if (( failures > 0 )); then
  echo "[preflight] completed with ${failures} failure(s) and ${warnings} warning(s)"
  exit 1
fi

echo "[preflight] completed with ${warnings} warning(s)"
