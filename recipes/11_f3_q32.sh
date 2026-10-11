#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f3_knockout.py --model ${MODEL_DIR}/Qwen3-32B --model-name Qwen3-32B --window 8 --out data/f3_Qwen3-32B.jsonl
