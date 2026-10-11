#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f10_heads.py --model ${MODEL_DIR}/Qwen3-8B --model-name Qwen3-8B --heads 32:9,29:4,29:16 --out data/f10b_Qwen3-8B.jsonl && ${PYTHON} f10_heads.py --raw --model ${MODEL_DIR}/Qwen3-8B-Base --model-name Qwen3-8B-Base --heads 32:9,29:4,29:16 --out data/f10braw_Qwen3-8B-Base.jsonl
