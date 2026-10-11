#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f7_thinking.py --model ${MODEL_DIR}/Qwen3-8B --model-name Qwen3-8B --batch 64 --out data/f7_Qwen3-8B.jsonl
