#!/bin/bash
#  Extension C -- one simulation run over the evaluation frame.
#  JOBLIST=ft    -> jobs_simulate_ft.txt    (needs the LoRA adapters)
#  JOBLIST=other -> jobs_simulate_other.txt (ICL and the alpha = 0 reference)
#  Submitted by scripts/run_extension_c.sh.
 
#SBATCH --job-name=extc-sim
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/extc_sim_%A_%a.out
#SBATCH --error=logs/extc_sim_%A_%a.err
 
set -euo pipefail
mkdir -p logs
 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${SOEPSIM_ENV:-soepsim}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TOKENIZERS_PARALLELISM=false
 
FOCAL="${FOCAL:-q1}"
SEED="${SEED:-0}"
JOBLIST="${JOBLIST:?set JOBLIST=ft or JOBLIST=other}"
 
echo "Job ${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID} on $(hostname): ${JOBLIST} #${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
 
DS_ARG=()
[ -n "${DATASET_DIR:-}" ] && DS_ARG=(--dataset-dir "${DATASET_DIR}")
 
python experiments/extension_c_allocation.py simulate --focal "${FOCAL}" --seed "${SEED}" \
    --job-list "${JOBLIST}" --job-index "${SLURM_ARRAY_TASK_ID}" "${DS_ARG[@]}"