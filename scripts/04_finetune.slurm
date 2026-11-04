#!/bin/bash
#  Domain-FT (LoRA) 
#  Fine-tunes each base model on the history-wave corpus
#  (data/human_train.chat.jsonl). 
#
#      FOCAL=q1 sbatch scripts/04_finetune.slurm
#
#  Array indices: 0 llama, 1 ministral, 2 qwen
#
#  axolotl is not a project dependency because it pins its own torch build.
#  Install it in the same env once:   pip install axolotl==0.12.2

#SBATCH --job-name=soep-ft
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --array=0-2
#SBATCH --output=logs/ft_%A_%a.out
#SBATCH --error=logs/ft_%A_%a.err

set -euo pipefail
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${SOEPSIM_ENV:-soepsim}"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TOKENIZERS_PARALLELISM=false

MODELS=(llama ministral qwen)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
FOCAL="${FOCAL:-q1}"

echo "Job ${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "focal=${FOCAL}  model=${MODEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "datasets/soep_${FOCAL}"

if [ ! -f data/human_train.chat.jsonl ]; then
    echo "ERROR: data/human_train.chat.jsonl missing. Run scripts/01_build_dataset.sh first."
    exit 1
fi

accelerate launch -m axolotl.cli.train "configs/finetune_${MODEL}.yaml" --debug=False --debug-text-only=False --debug-num-examples=0 --shard=False

echo "Done: outputs/models/${MODEL}/lora"
