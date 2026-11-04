#!/bin/bash
#  Extension C -- one LoRA per (model, alpha) on the wave-2021 adaptation set A.
#  Submitted by scripts/run_extension_c.sh; array index = line in jobs_finetune.txt.
#
#      FOCAL=q1 SEED=0 sbatch --array=0-5 scripts/extension_c_finetune.slurm
 
#SBATCH --job-name=extc-ft
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=logs/extc_ft_%A_%a.out
#SBATCH --error=logs/extc_ft_%A_%a.err
 
set -euo pipefail
mkdir -p logs
 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${SOEPSIM_ENV:-soepsim}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TOKENIZERS_PARALLELISM=false
 
FOCAL="${FOCAL:-q1}"
SEED="${SEED:-0}"
DATASET_DIR="${DATASET_DIR:-datasets/soep_${FOCAL}_full}"
WORK="extc/seed${SEED}"
 
cd "${DATASET_DIR}"
read -r MODEL TAG < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${WORK}/jobs_finetune.txt")
CFG="${WORK}/configs/finetune_${MODEL}_${TAG}.yaml"
 
echo "Job ${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID} on $(hostname): model=${MODEL} ${TAG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
 
accelerate launch -m axolotl.cli.train "${CFG}" --debug=False --debug-text-only=False --debug-num-examples=0 --shard=False
 
echo "Done: ${WORK}/models/${MODEL}/${TAG}/lora"