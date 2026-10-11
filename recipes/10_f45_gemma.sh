#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && for g in f4 f5; do ${PYTHON} score_grid.py --model ${MODEL_DIR}/gemma-4-31B-it --tokenizer ${TOKENIZER_DIR}/gemma-4-31B-it --model-name gemma-4-31B-it --family gemma --grid f45_grids:$g --out data/${g}_gemma-4-31B-it.jsonl || exit 1; done
