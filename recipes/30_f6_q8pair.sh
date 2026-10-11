#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR"
${PYTHON} score_grid.py --model ${MODEL_DIR}/Qwen3-8B-Base --model-name Qwen3-8B-Base --family qwen --raw --grid f45_grids:main_grid --out data/f6raw_Qwen3-8B-Base.jsonl || exit 1
${PYTHON} score_grid.py --model ${MODEL_DIR}/Qwen3-8B --model-name Qwen3-8B --family qwen --raw --grid f45_grids:main_grid --out data/f6raw_Qwen3-8B.jsonl
