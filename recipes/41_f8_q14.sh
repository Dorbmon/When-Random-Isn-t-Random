#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f8_heads.py --model ${MODEL_DIR}/Qwen3-14B --model-name Qwen3-14B --layer-start 20 --layer-end 40 --out data/f8_Qwen3-14B.jsonl
