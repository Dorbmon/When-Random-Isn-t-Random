#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR"
PY=${PYTHON}; T=${TOKENIZER_DIR}
CUDA_VISIBLE_DEVICES=1 $PY repulsion_followup.py --model ${MODEL_DIR}/Qwen3-32B --model-name Qwen3-32B --family qwen --out data/f1_Qwen3-32B.jsonl > logs/f1_qwen32.log 2>&1
CUDA_VISIBLE_DEVICES=1 $PY repulsion_followup.py --model ${MODEL_DIR}/gemma-4-31B-it --tokenizer $T/gemma-4-31B-it --model-name gemma-4-31B-it --family gemma --out data/f1_gemma-4-31B-it.jsonl > logs/f1_gemma.log 2>&1
echo done > logs/f1.done
