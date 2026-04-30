#!/bin/bash
#SBATCH -c 4                          # Number of Cores per Task
#SBATCH --mem=32G                     # Requested Memory
#SBATCH -p superpod-a100              # Partition
#SBATCH --gres=gpu:1                  # Number of GPUs
#SBATCH -t 2-00:00:00                 # Job time limit
#SBATCH -o ./scripts/jobs/%j.out      # %j = job ID
#SBATCH --constraint="vram16"

# Run from repo root:
#   sbatch scripts/train_dpo.sh
#
# Override any default via --export=ALL,VAR=val, e.g.:
#   sbatch --export=ALL,CONFIG=configs/dpo_qwen05b_sgd.yaml scripts/train_dpo.sh
#   sbatch --export=ALL,CONFIG=configs/dpo_qwen15b.yaml,EPOCHS=5 scripts/train_dpo.sh

# ── Environment ───────────────────────────────────────────────────────────────
if [ ! -d "./scripts/jobs" ]; then
    mkdir -p ./scripts/jobs
fi

export HF_HOME="${HF_HOME:-$(pwd)/models}"
mkdir -p "${HF_HOME}"

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONUNBUFFERED=1

module load conda/latest
CONDA_ENV="${CONDA_ENV:-cs651}"
conda activate "${CONDA_ENV}"

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/dpo_qwen05b.yaml}"

# Optional per-key overrides (written into a temp config if set).
# Leave blank to use whatever is in the YAML file.
OPTIMIZER="${OPTIMIZER:-}"
LR="${LR:-}"
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
MODEL_NAME="${MODEL_NAME:-}"
SEED="${SEED:-}"

# Derive a run label for logging from the config filename.
CONFIG_STEM=$(basename "${CONFIG}" .yaml)

echo "[train_dpo] CONFIG=${CONFIG}"
echo "[train_dpo] CONFIG_STEM=${CONFIG_STEM}"
echo "[train_dpo] CONDA_ENV=${CONDA_ENV}"
echo "[train_dpo] HF_HOME=${HF_HOME}"
[ -n "${OPTIMIZER}"   ] && echo "[train_dpo] OVERRIDE optimizer=${OPTIMIZER}"
[ -n "${LR}"          ] && echo "[train_dpo] OVERRIDE lr=${LR}"
[ -n "${EPOCHS}"      ] && echo "[train_dpo] OVERRIDE epochs=${EPOCHS}"
[ -n "${BATCH_SIZE}"  ] && echo "[train_dpo] OVERRIDE batch_size=${BATCH_SIZE}"
[ -n "${MODEL_NAME}"  ] && echo "[train_dpo] OVERRIDE model_name=${MODEL_NAME}"
[ -n "${SEED}"        ] && echo "[train_dpo] OVERRIDE seed=${SEED}"

# ── Build effective config ────────────────────────────────────────────────────
# If any per-key overrides are set, patch them into a temp YAML so train_dpo.py
# still reads a single config file.
EFFECTIVE_CONFIG="${CONFIG}"
if [ -n "${OPTIMIZER}" ] || [ -n "${LR}" ] || [ -n "${EPOCHS}" ] || \
   [ -n "${BATCH_SIZE}" ] || [ -n "${MODEL_NAME}" ] || [ -n "${SEED}" ]; then
    EFFECTIVE_CONFIG="/tmp/dpo_config_${SLURM_JOB_ID:-$$}.yaml"
    cp "${CONFIG}" "${EFFECTIVE_CONFIG}"
    python - "${EFFECTIVE_CONFIG}" <<PYEOF
import sys, yaml
path = sys.argv[1]
with open(path) as f:
    cfg = yaml.safe_load(f)
import os
if os.environ.get("OPTIMIZER"):
    cfg.setdefault("optimizer", {})["name"] = os.environ["OPTIMIZER"]
if os.environ.get("LR"):
    cfg.setdefault("optimizer", {})["lr"] = float(os.environ["LR"])
if os.environ.get("EPOCHS"):
    cfg.setdefault("training", {})["epochs"] = int(os.environ["EPOCHS"])
if os.environ.get("BATCH_SIZE"):
    cfg.setdefault("data", {})["batch_size"] = int(os.environ["BATCH_SIZE"])
if os.environ.get("MODEL_NAME"):
    cfg.setdefault("model", {})["model_name"] = os.environ["MODEL_NAME"]
if os.environ.get("SEED"):
    cfg.setdefault("experiment", {})["seed"] = int(os.environ["SEED"])
with open(path, "w") as f:
    yaml.dump(cfg, f)
PYEOF
    echo "[train_dpo] Patched config written to ${EFFECTIVE_CONFIG}"
fi

# ── Run metadata ──────────────────────────────────────────────────────────────
START_EPOCH="$(date +%s)"
START_ISO="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[train_dpo] start_utc=${START_ISO}"
echo "[train_dpo] slurm_job_id=${SLURM_JOB_ID:-local}"
echo "[train_dpo] hostname=$(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
fi

# ── Train ─────────────────────────────────────────────────────────────────────
python -u scripts/train_dpo.py --config "${EFFECTIVE_CONFIG}"
EXIT_CODE=$?

# ── Timing ───────────────────────────────────────────────────────────────────
END_EPOCH="$(date +%s)"
DURATION_SEC="$((END_EPOCH - START_EPOCH))"
DURATION_HMS="$(printf "%02d:%02d:%02d" \
    "$((DURATION_SEC/3600))" "$(((DURATION_SEC%3600)/60))" "$((DURATION_SEC%60))")"
echo "[train_dpo] end_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[train_dpo] duration=${DURATION_HMS}  exit_code=${EXIT_CODE}"

# Clean up temp config if we created one.
[ "${EFFECTIVE_CONFIG}" != "${CONFIG}" ] && rm -f "${EFFECTIVE_CONFIG}"

exit "${EXIT_CODE}"
