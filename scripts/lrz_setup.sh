#!/usr/bin/env bash
# One-time environment setup on the LRZ AI Systems.
#
#   ssh <username>@login.ai.lrz.de
#   tmux                       # keeps the session alive if the connection drops
#   bash scripts/lrz_setup.sh

set -euo pipefail

ENV_NAME="${1:-soepsim}"
PY_VERSION="3.11"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi
conda activate "${ENV_NAME}"

# CUDA build of PyTorch; adjust the index URL to the cluster's driver version.
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126

# The project itself (editable), including the ICL dependencies.
pip install -e ".[icl,dev]"

# Fine-tuning (Domain-FT); not a project dependency because it pins its own torch build.
pip install axolotl==0.12.2

echo ""
echo "Environment '${ENV_NAME}' ready."
echo ""
echo "Log in to Hugging Face (needed for the gated Llama 3.1 weights):"
echo "  huggingface-cli login"
echo ""
echo "Optionally set the model cache (default: \$HOME/hf_cache):"
echo "  export HF_HOME=/path/to/hf_cache"
echo ""
echo "Check available partitions with:  sinfo"
