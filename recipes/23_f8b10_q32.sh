#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f8b_head_detail.py --model ${MODEL_DIR}/Qwen3-32B --model-name Qwen3-32B --heads 57:41,56:18,61:47,54:17,56:21,58:62,62:63,59:45,54:20,58:29 --n-random 4 --out data/f8b10_Qwen3-32B.jsonl
