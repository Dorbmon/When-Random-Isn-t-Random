#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f14_multidigit.py --model ${MODEL_DIR}/gemma-4-31B-it --tokenizer ${TOKENIZER_DIR}/gemma-4-31B-it --model-name gemma-4-31B-it --family gemma --out data/f14_gemma-4-31B-it.jsonl
