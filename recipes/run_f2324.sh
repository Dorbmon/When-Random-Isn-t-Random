#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
# Usage: run_f2324.sh <items.jsonl> <outprefix>. Two queues, one per GPU.
cd "$REPO_DIR"
PY=${PYTHON}
IT=$1; P=$2
run() { CUDA_VISIBLE_DEVICES=$1 $PY f23_score.py --items $IT --model $2 ${4:+--tokenizer $4} --model-name $3 --family $5 --out data/${P}_$3.jsonl > logs/${P}_$3.log 2>&1; }
q0() {
  run 0 ${MODEL_DIR}/Qwen3-4B Qwen3-4B "" qwen
  run 0 ${MODEL_DIR}/Qwen3-8B Qwen3-8B "" qwen
  run 0 ${MODEL_DIR}/Qwen3-32B Qwen3-32B "" qwen
  echo done > logs/${P}_q0.done
}
q1() {
  run 1 ${MODEL_DIR}/Qwen3-14B Qwen3-14B "" qwen
  run 1 ${MODEL_DIR}/gemma-4-31B-it gemma-4-31B-it ${TOKENIZER_DIR}/gemma-4-31B-it gemma
  run 1 ${MODEL_DIR}/Muse-Glimmer-30B Muse-Glimmer-30B ${TOKENIZER_DIR}/Muse-Glimmer-30B muse
  echo done > logs/${P}_q1.done
}
q0 & q1 & wait
