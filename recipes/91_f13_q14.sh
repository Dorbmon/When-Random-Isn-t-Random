#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} repulsion_f2.py --raw --model ${MODEL_DIR}/Qwen3-14B-Base --model-name Qwen3-14B-Base --family qwen --out data/f2raw_Qwen3-14B-Base.jsonl && ${PYTHON} repulsion_f2.py --raw --model ${MODEL_DIR}/Qwen3-14B --model-name Qwen3-14B --family qwen --out data/f2raw_Qwen3-14B.jsonl
