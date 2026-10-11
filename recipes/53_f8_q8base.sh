#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f8_heads.py --raw --model ${MODEL_DIR}/Qwen3-8B-Base --model-name Qwen3-8B-Base --layer-start 18 --layer-end 36 --out data/f8raw_Qwen3-8B-Base.jsonl
