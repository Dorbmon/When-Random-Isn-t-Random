#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR"
CUDA_VISIBLE_DEVICES=0 ${PYTHON} repulsion_score.py --model ${MODEL_DIR}/Muse-Glimmer-30B --tokenizer ${TOKENIZER_DIR}/Muse-Glimmer-30B --model-name Muse-Glimmer-30B --family muse --x-separator "\n" --out data/Muse-Glimmer-30B.jsonl > logs/muse.log 2>&1
echo done > logs/muse.done
