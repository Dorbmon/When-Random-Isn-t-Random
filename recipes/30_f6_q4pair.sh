#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR"
${PYTHON} score_grid.py --model ${MODEL_DIR}/Qwen3-4B-Base --model-name Qwen3-4B-Base --family qwen --raw --grid f45_grids:main_grid --out data/f6raw_Qwen3-4B-Base.jsonl || exit 1
${PYTHON} score_grid.py --model ${MODEL_DIR}/Qwen3-4B --model-name Qwen3-4B --family qwen --raw --grid f45_grids:main_grid --out data/f6raw_Qwen3-4B.jsonl
