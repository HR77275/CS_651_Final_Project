#!/bin/bash
# Submit 3 SGD lr sweep jobs in parallel.
# Run from repo root:  bash scripts/run_sgd_sweep.sh

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-cs651}"

submit() {
    local config="$1"
    local job_id
    job_id=$(sbatch --export="ALL,CONFIG=${config},CONDA_ENV=${CONDA_ENV}" scripts/train_dpo.sh | awk '{print $NF}')
    echo "Submitted ${config}  →  job ${job_id}"
}

echo "=== SGD learning rate sweep (Qwen2.5-0.5B) ==="
submit configs/dpo_qwen05b_sgd_lr1e4.yaml
submit configs/dpo_qwen05b_sgd_lr1e5.yaml
submit configs/dpo_qwen05b_sgd_lr1e6.yaml

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     scripts/jobs/<job_id>.out"
