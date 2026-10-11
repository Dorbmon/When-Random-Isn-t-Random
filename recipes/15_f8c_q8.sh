#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f8c_patch.py --model ${MODEL_DIR}/Qwen3-8B --model-name Qwen3-8B --heads 32:9,29:4,29:16 --out data/f8c_Qwen3-8B.jsonl
