#!/usr/bin/env bash
set -euo pipefail

# Submit N chained 12h segments of the same long run.
# Usage:
#   ./slurm/dgx1/chain_submit.sh myrun 3
#
# Result:
#   N jobs submitted with dependencies; each resumes from checkpoints in runs/<RUN_ID>.

RUN_ID="${1:-longrun_demo}"
N="${2:-3}"

echo "[chain] RUN_ID=${RUN_ID} segments=${N}"

jid_prev=""
for i in $(seq 1 "${N}"); do
  if [[ -z "${jid_prev}" ]]; then
    jid_prev=$(sbatch --export=ALL,RUN_ID="${RUN_ID}" slurm/dgx1/02_train_1gpu_12h_chain.sbatch | awk '{print $4}')
    echo "[chain] segment ${i} -> job ${jid_prev}"
  else
    jid=$(sbatch --dependency=afterany:"${jid_prev}" --export=ALL,RUN_ID="${RUN_ID}" slurm/dgx1/02_train_1gpu_12h_chain.sbatch | awk '{print $4}')
    echo "[chain] segment ${i} -> job ${jid} (after ${jid_prev})"
    jid_prev="${jid}"
  fi
done
