#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR"
CUDA_VISIBLE_DEVICES=0 ${PYTHON} repulsion_freegen.py --model ${MODEL_DIR}/Qwen3-8B --model-name Qwen3-8B --out data/freegen_Qwen3-8B.jsonl > logs/freegen_8b.log 2>&1
echo done > logs/freegen.done
