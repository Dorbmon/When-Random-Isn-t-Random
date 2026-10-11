#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR"
PY=${PYTHON}; T=${TOKENIZER_DIR}
g0() {
  CUDA_VISIBLE_DEVICES=0 $PY repulsion_f2.py --model ${MODEL_DIR}/Qwen3-4B --model-name Qwen3-4B --family qwen --out data/f2_Qwen3-4B.jsonl > logs/f2_q4.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY repulsion_f2.py --model ${MODEL_DIR}/Qwen3-8B --model-name Qwen3-8B --family qwen --x-separator '  \n' --out data/f2_Qwen3-8B.jsonl > logs/f2_q8.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY repulsion_f2.py --model ${MODEL_DIR}/Qwen3-32B --model-name Qwen3-32B --family qwen --out data/f2_Qwen3-32B.jsonl > logs/f2_q32.log 2>&1
  echo done > logs/f2_g0.done
}
g1() {
  CUDA_VISIBLE_DEVICES=1 $PY repulsion_f2.py --model ${MODEL_DIR}/gemma-4-31B-it --tokenizer $T/gemma-4-31B-it --model-name gemma-4-31B-it --family gemma --out data/f2_gemma-4-31B-it.jsonl > logs/f2_gemma.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $PY repulsion_f2.py --model ${MODEL_DIR}/Muse-Glimmer-30B --tokenizer $T/Muse-Glimmer-30B --model-name Muse-Glimmer-30B --family muse --out data/f2_Muse-Glimmer-30B.jsonl > logs/f2_muse.log 2>&1
  echo done > logs/f2_g1.done
}
g0 & g1 & wait
