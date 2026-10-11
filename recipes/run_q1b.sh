#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR"
PY=${PYTHON}
T=${TOKENIZER_DIR}
CUDA_VISIBLE_DEVICES=1 $PY repulsion_score.py --model ${MODEL_DIR}/gemma-4-31B-it --tokenizer $T/gemma-4-31B-it --model-name gemma-4-31B-it --family gemma --x-separator "\n" --out data/gemma-4-31B-it.jsonl > logs/gemma.log 2>&1
CUDA_VISIBLE_DEVICES=1 $PY repulsion_score.py --model ${MODEL_DIR}/Muse-Glimmer-30B --tokenizer $T/Muse-Glimmer-30B --model-name Muse-Glimmer-30B --family muse --x-separator "\n" --out data/Muse-Glimmer-30B.jsonl > logs/muse.log 2>&1
echo Q1_DONE > logs/q1.done
