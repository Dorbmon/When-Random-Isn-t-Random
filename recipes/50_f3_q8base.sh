#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f3_knockout.py --raw --model ${MODEL_DIR}/Qwen3-8B-Base --model-name Qwen3-8B-Base --window 4 --out data/f3raw_Qwen3-8B-Base.jsonl
