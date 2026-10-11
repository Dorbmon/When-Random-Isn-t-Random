#!/usr/bin/env bash
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR
export PYTHON="${PYTHON:-python}"
export MODEL_DIR="${MODEL_DIR:-$REPO_DIR/models}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-$REPO_DIR/tokenizers}"
export PYTHONUTF8=1
cd "$REPO_DIR"
mkdir -p data api_data results logs data/f15
