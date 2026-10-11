#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f10_categorical.py --model ${MODEL_DIR}/Qwen3-14B --model-name Qwen3-14B --family qwen --out data/f10_Qwen3-14B.jsonl
