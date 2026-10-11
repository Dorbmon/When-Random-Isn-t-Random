#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f8c_patch.py --model ${MODEL_DIR}/Qwen3-32B --model-name Qwen3-32B --heads 57:41,56:18,61:47 --out data/f8c_Qwen3-32B.jsonl
