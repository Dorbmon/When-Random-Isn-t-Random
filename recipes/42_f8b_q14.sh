#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f8b_head_detail.py --model ${MODEL_DIR}/Qwen3-14B --model-name Qwen3-14B --heads 33:26,32:12,30:11,37:29 --out data/f8b_Qwen3-14B.jsonl
