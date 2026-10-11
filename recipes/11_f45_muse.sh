#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && for g in f4 f5; do ${PYTHON} score_grid.py --model ${MODEL_DIR}/Muse-Glimmer-30B --tokenizer ${TOKENIZER_DIR}/Muse-Glimmer-30B --model-name Muse-Glimmer-30B --family muse --grid f45_grids:$g --out data/${g}_Muse-Glimmer-30B.jsonl || exit 1; done
