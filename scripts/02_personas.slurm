#!/bin/bash
# Persona generation: one paragraph per respondent from the demographic
# profile and response history, reused by all three synthesis models.
#
#   sbatch scripts/02_personas.slurm

#SBATCH --job-name=soep-personas
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/personas_%j.out
#SBATCH --error=logs/personas_%j.err

set -euo pipefail
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${SOEPSIM_ENV:-soepsim}"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TOKENIZERS_PARALLELISM=false

FOCAL="${FOCAL:-q1}"
cd "datasets/soep_${FOCAL}"

echo "Job $SLURM_JOB_ID on $(hostname), focal=${FOCAL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python simulate.py --config configs/simulate_qwen_concept_guided.yaml

echo "Done. Personas cached at datasets/soep_${FOCAL}/outputs/personas/personas.csv"
