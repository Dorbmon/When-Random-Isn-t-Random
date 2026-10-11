#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f7_thinking.py --model ${MODEL_DIR}/Qwen3-32B --model-name Qwen3-32B --batch 16 --out data/f7_Qwen3-32B.jsonl
