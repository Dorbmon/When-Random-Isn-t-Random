#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f8_heads.py --model ${MODEL_DIR}/Qwen3-8B --model-name Qwen3-8B --layer-start 18 --layer-end 36 --out data/f8_Qwen3-8B.jsonl
