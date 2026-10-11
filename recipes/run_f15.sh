#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR"
A="--model ${MODEL_DIR}/gemma-4-31B-it --tokenizer ${TOKENIZER_DIR}/gemma-4-31B-it --model-name gemma-4-31B-it --outdir data/f15"
CUDA_VISIBLE_DEVICES=1 ${PYTHON} gemma_circuit.py $A --phases f3 > logs/f15_f3.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 ${PYTHON} gemma_circuit.py $A --phases f8 f8b f8c > logs/f15_f8.log 2>&1 &
wait
echo F15_DONE > logs/f15.done
