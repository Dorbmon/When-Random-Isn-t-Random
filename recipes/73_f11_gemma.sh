#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f10_categorical.py --f11 --model ${MODEL_DIR}/gemma-4-31B-it --tokenizer ${TOKENIZER_DIR}/gemma-4-31B-it --model-name gemma-4-31B-it --family gemma --out data/f11_gemma-4-31B-it.jsonl
