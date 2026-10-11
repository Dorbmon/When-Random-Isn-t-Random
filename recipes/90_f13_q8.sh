#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} repulsion_f2.py --raw --model ${MODEL_DIR}/Qwen3-8B-Base --model-name Qwen3-8B-Base --family qwen --out data/f2raw_Qwen3-8B-Base.jsonl && ${PYTHON} repulsion_f2.py --raw --model ${MODEL_DIR}/Qwen3-8B --model-name Qwen3-8B --family qwen --out data/f2raw_Qwen3-8B.jsonl
