#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f14_multidigit.py --raw --model ${MODEL_DIR}/Qwen3-8B-Base --model-name Qwen3-8B-Base --family qwen --out data/f14raw_Qwen3-8B-Base.jsonl
