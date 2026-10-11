#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR" && ${PYTHON} f14_multidigit.py --model ${MODEL_DIR}/Muse-Glimmer-30B --tokenizer ${TOKENIZER_DIR}/Muse-Glimmer-30B --model-name Muse-Glimmer-30B --family muse --out data/f14_Muse-Glimmer-30B.jsonl
