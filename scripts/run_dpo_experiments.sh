#!/bin/bash
# Submit all four DPO optimizer experiments as independent SLURM jobs.
#
# Run from repo root:
#   bash scripts/run_dpo_experiments.sh
#
# Optional overrides (applied to every job):
#   EPOCHS=5 bash scripts/run_dpo_experiments.sh
#   CONDA_ENV=myenv bash scripts/run_dpo_experiments.sh

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-cs651}"
EPOCHS="${EPOCHS:-}"

BASE_ARGS=(scripts/train_dpo.sh)

submit() {
    local config="$1"
    local extra_export="${2:-}"
    local export_str="ALL,CONFIG=${config},CONDA_ENV=${CONDA_ENV}"
    [ -n "${EPOCHS}" ]      && export_str="${export_str},EPOCHS=${EPOCHS}"
    [ -n "${extra_export}" ] && export_str="${export_str},${extra_export}"
    local job_id
    job_id=$(sbatch --export="${export_str}" "${BASE_ARGS[@]}" | awk '{print $NF}')
    echo "Submitted ${config}  →  job ${job_id}"
}

echo "=== Submitting DPO optimizer comparison (Qwen2.5-0.5B) ==="
# submit configs/dpo_qwen05b.yaml          # AdamW (baseline)
submit configs/dpo_qwen05b_sgd.yaml
submit configs/dpo_qwen05b_rmsprop.yaml
submit configs/dpo_qwen05b_lion.yaml

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       scripts/jobs/<job_id>.out"
echo "Once all done: bash scripts/plot_dpo.sh"
