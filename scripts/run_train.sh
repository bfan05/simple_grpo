#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/local_debug.yaml}"

echo "Using config: $CONFIG"
python -m src.train_grpo --config "$CONFIG"
