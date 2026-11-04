#!/bin/bash
# Response synthesis: 3 models x 4 methods for one focal question.
#
#   FOCAL=q1 sbatch scripts/03_simulate.slurm
#   FOCAL=q2 sbatch scripts/03_simulate.slurm
#
# Array indices (demo-only / persona-guided / Domain-FT / ICL):
#   0-3 llama, 4-7 ministral, 8-11 qwen
# Domain-FT requires a trained adapter (scripts/04_finetune.slurm).
# Subset: sbatch --array=0,4,8 scripts/03_simulate.slurm

#SBATCH --job-name=soep-sim
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-11
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err

set -euo pipefail
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${SOEPSIM_ENV:-soepsim}"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TOKENIZERS_PARALLELISM=false

MODELS=(llama ministral qwen)
METHODS=(silicon_sampling concept_guided ft icl)

MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID / 4))]}"
METHOD="${METHODS[$((SLURM_ARRAY_TASK_ID % 4))]}"
FOCAL="${FOCAL:-q1}"

echo "Job ${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "focal=${FOCAL}  model=${MODEL}  method=${METHOD}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "datasets/soep_${FOCAL}"

if [ "${METHOD}" = "ft" ] && [ ! -d "outputs/models/${MODEL}/lora" ]; then
    echo "ERROR: no LoRA adapter at outputs/models/${MODEL}/lora"
    echo "Run scripts/04_finetune.slurm for ${MODEL} first."
    exit 1
fi

if [ "${METHOD}" = "icl" ] && [ ! -f "data/persona_input.csv" ]; then
    echo "ERROR: data/persona_input.csv missing -- ICL retrieval needs it to build"
    echo "its example pool. Run scripts/01_build_dataset.sh first."
    exit 1
fi

python simulate.py --config "configs/simulate_${MODEL}_${METHOD}.yaml"

echo "Done: outputs/simulated_data/${MODEL}/${METHOD}.csv"