#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && for g in f4 f5; do ${PYTHON} score_grid.py --model ${MODEL_DIR}/Qwen3-8B --model-name Qwen3-8B --family qwen --x-separator '  \n' --grid f45_grids:$g --out data/${g}_Qwen3-8B.jsonl || exit 1; done
