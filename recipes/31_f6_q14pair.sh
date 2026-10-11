#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR"
${PYTHON} score_grid.py --model ${MODEL_DIR}/Qwen3-14B-Base --model-name Qwen3-14B-Base --family qwen --raw --grid f45_grids:main_grid --out data/f6raw_Qwen3-14B-Base.jsonl || exit 1
${PYTHON} score_grid.py --model ${MODEL_DIR}/Qwen3-14B --model-name Qwen3-14B --family qwen --raw --grid f45_grids:main_grid --out data/f6raw_Qwen3-14B.jsonl || exit 1
${PYTHON} repulsion_score.py --model ${MODEL_DIR}/Qwen3-14B --model-name Qwen3-14B --family qwen --out data/Qwen3-14B.jsonl
