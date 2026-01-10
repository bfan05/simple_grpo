#!/usr/bin/env bash
set -euo pipefail

echo "=============================="
echo " RL LOCAL SETUP (CONDA, MAC)"
echo "=============================="

ENV_NAME="${ENV_NAME:-rl-local}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Make conda available in non-interactive shells
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found in PATH. Install Miniforge/Anaconda first."
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"
python --version

pip install --upgrade pip setuptools wheel

# On Mac: install CPU/MPS torch via pip (no CUDA wheels here)
pip install --upgrade \
  torch \
  transformers datasets accelerate trl peft \
  sentencepiece \
  wandb evaluate numpy scipy packaging rich matplotlib

echo "------------------------------"
echo "Sanity checks:"
python - << 'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
EOF
echo "=============================="
echo " LOCAL SETUP COMPLETE"
echo "=============================="
