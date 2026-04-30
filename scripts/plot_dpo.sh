#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -p cpu
#SBATCH -t 0-00:30:00
#SBATCH -o ./scripts/jobs/%j.out

# Generate DPO optimizer comparison plots from completed training runs.
#
# Run from repo root:
#   sbatch scripts/plot_dpo.sh
#
# Overrides:
#   sbatch --export=ALL,OUTPUTS_DIR=outputs,PLOT_DIR=outputs/plots scripts/plot_dpo.sh

if [ ! -d "./scripts/jobs" ]; then
    mkdir -p ./scripts/jobs
fi

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONUNBUFFERED=1

module load conda/latest
CONDA_ENV="${CONDA_ENV:-cs651}"
conda activate "${CONDA_ENV}"

PLOT_DIR="${PLOT_DIR:-outputs/dpo_comparison_plots}"
REPORT_DIR="${REPORT_DIR:-docs/report_figures_dpo}"

# Canonical 4-optimizer runs (one best LR per optimizer)
RUN_DIRS=(
    "outputs/dpo_qwen05b_ultrafeedback"   # AdamW
    "outputs/dpo_qwen05b_rmsprop"          # RMSProp
    "outputs/dpo_qwen05b_lion"             # Lion
    "outputs/dpo_qwen05b_sgd_lr1e5"        # SGD (best lr from sweep)
)

echo "[plot_dpo] PLOT_DIR=${PLOT_DIR}"
echo "[plot_dpo] REPORT_DIR=${REPORT_DIR}"
echo "[plot_dpo] Runs: ${RUN_DIRS[*]}"

python -u scripts/plot_dpo_comparison.py \
    --run-dirs "${RUN_DIRS[@]}" \
    --output-dir "${PLOT_DIR}" \
    --report-dir "${REPORT_DIR}"

echo "[plot_dpo] Done. Plots saved to ${PLOT_DIR} and mirrored to ${REPORT_DIR}"
