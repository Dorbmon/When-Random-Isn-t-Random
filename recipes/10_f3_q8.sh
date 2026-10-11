#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f3_knockout.py --model ${MODEL_DIR}/Qwen3-8B --model-name Qwen3-8B --window 4 --out data/f3_Qwen3-8B.jsonl
