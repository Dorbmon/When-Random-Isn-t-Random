#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f8_heads.py --model ${MODEL_DIR}/Qwen3-32B --model-name Qwen3-32B --layer-start 48 --layer-end 64 --head-batch 32 --out data/f8_Qwen3-32B.jsonl
