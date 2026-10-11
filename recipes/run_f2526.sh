#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
# F25 (anchor in sampling) and F26 (OV layer rank). One queue per GPU.
cd "$REPO_DIR"
PY=${PYTHON}; T=${TOKENIZER_DIR}
M4=${MODEL_DIR}/Qwen3-4B; M8=${MODEL_DIR}/Qwen3-8B
M32=${MODEL_DIR}/Qwen3-32B; B=${MODEL_DIR}
G=${MODEL_DIR}/gemma-4-31B-it; MU=${MODEL_DIR}/Muse-Glimmer-30B
H8=32:9,29:4,29:16; H14=33:26,32:12,30:11,37:29; H32=57:41,56:18,61:47; HG=57:3,41:10,41:11
f25() { CUDA_VISIBLE_DEVICES=$1 $PY f25_anchor_sampling.py --model $2 --model-name $3 --family $4 --x-separator "$5" ${6:+--tokenizer $6} --out data/f25_$3.jsonl > logs/f25_$3.log 2>&1; }
f26() { CUDA_VISIBLE_DEVICES=$1 $PY f26_ov_layer_rank.py --model $2 --model-name $3 --family $4 --heads $5 ${6:+--tokenizer $6} --out data/f26_$3.json > logs/f26_$3.log 2>&1; }
g0() {
  f25 0 $M8 Qwen3-8B qwen '  \n'
  f25 0 $M4 Qwen3-4B qwen '\n'
  f25 0 $M32 Qwen3-32B qwen '\n'
  f26 0 $M8 Qwen3-8B qwen $H8
  f26 0 $B/Qwen3-8B-Base Qwen3-8B-Base qwen $H8
  f26 0 $M32 Qwen3-32B qwen $H32
  echo done > logs/f2526_g0.done
}
g1() {
  f25 1 $B/Qwen3-14B Qwen3-14B qwen '\n'
  f25 1 $G gemma-4-31B-it gemma '\n' $T/gemma-4-31B-it
  f25 1 $MU Muse-Glimmer-30B muse '\n' $T/Muse-Glimmer-30B
  f26 1 $B/Qwen3-14B Qwen3-14B qwen $H14
  f26 1 $G gemma-4-31B-it gemma $HG $T/gemma-4-31B-it
  echo done > logs/f2526_g1.done
}
g0 & g1 & wait
