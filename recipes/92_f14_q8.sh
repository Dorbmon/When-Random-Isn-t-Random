#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f14_multidigit.py --model ${MODEL_DIR}/Qwen3-8B --model-name Qwen3-8B --family qwen --out data/f14_Qwen3-8B.jsonl
