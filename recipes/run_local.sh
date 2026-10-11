#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
# Two sequential queues, one per GPU.
cd "$REPO_DIR"
PY=${PYTHON}
q0() {
  CUDA_VISIBLE_DEVICES=0 $PY repulsion_score.py --model ${MODEL_DIR}/Qwen3-4B --model-name Qwen3-4B --family qwen --x-separator '\n' --out data/Qwen3-4B.jsonl > logs/Qwen3-4B.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY repulsion_score.py --model ${MODEL_DIR}/Qwen3-8B --model-name Qwen3-8B --family qwen --x-separator '  \n' --out data/Qwen3-8B.jsonl > logs/Qwen3-8B.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY repulsion_score.py --model ${MODEL_DIR}/Qwen3-32B --model-name Qwen3-32B --family qwen --x-separator '\n' --out data/Qwen3-32B.jsonl > logs/Qwen3-32B.log 2>&1
  echo Q0_DONE > logs/q0.done
}
q1() {
  CUDA_VISIBLE_DEVICES=1 $PY repulsion_score.py --model ${MODEL_DIR}/gemma-4-31B-it --model-name gemma-4-31B-it --family gemma --x-separator '\n' --out data/gemma-4-31B-it.jsonl > logs/gemma.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $PY repulsion_score.py --model ${MODEL_DIR}/Muse-Glimmer-30B --model-name Muse-Glimmer-30B --family muse --x-separator '\n' --out data/Muse-Glimmer-30B.jsonl > logs/muse.log 2>&1
  echo Q1_DONE > logs/q1.done
}
q0 & q1 & wait
