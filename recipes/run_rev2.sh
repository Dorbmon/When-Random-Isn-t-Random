#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR"
PY=${PYTHON}; T=${TOKENIZER_DIR}
M8=${MODEL_DIR}/Qwen3-8B; M32=${MODEL_DIR}/Qwen3-32B; B=${MODEL_DIR}; G=${MODEL_DIR}/gemma-4-31B-it
H8=32:9,29:4,29:16; H14=33:26,32:12,30:11,37:29; H32=57:41,56:18,61:47; H32T=57:41,56:18,61:47,54:17,56:21,58:62,62:63,59:45,54:20,58:29
g0() {
  CUDA_VISIBLE_DEVICES=0 $PY f20_ov.py --model $M8 --model-name Qwen3-8B --family qwen --heads $H8 --out data/f20_Qwen3-8B.json > logs/f20_q8.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY f20_ov.py --model $B/Qwen3-8B-Base --model-name Qwen3-8B-Base --family qwen --heads $H8 --out data/f20_Qwen3-8B-Base.json > logs/f20_q8b.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY f20_ov.py --model $B/Qwen3-14B --model-name Qwen3-14B --family qwen --heads $H14 --out data/f20_Qwen3-14B.json > logs/f20_q14.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY f20_ov.py --model $M32 --model-name Qwen3-32B --family qwen --heads $H32 --out data/f20_Qwen3-32B.json > logs/f20_q32.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY f21_yrep.py --model $M8 --model-name Qwen3-8B --heads $H8 --x-separator '  \n' --out data/f21_Qwen3-8B.jsonl > logs/f21_q8.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY f21_yrep.py --model $B/Qwen3-14B --model-name Qwen3-14B --heads $H14 --out data/f21_Qwen3-14B.jsonl > logs/f21_q14.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY f21_yrep.py --model $M32 --model-name Qwen3-32B --heads $H32T --out data/f21_Qwen3-32B.jsonl > logs/f21_q32.log 2>&1
  echo done > logs/rev2_g0.done
}
g1() {
  CUDA_VISIBLE_DEVICES=1 $PY f20_ov.py --model $G --tokenizer $T/gemma-4-31B-it --model-name gemma-4-31B-it --family gemma --heads 57:3,41:10,41:11 --out data/f20_gemma-4-31B-it.json > logs/f20_gemma.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $PY f22_digitpos.py --model $M8 --model-name Qwen3-8B --heads $H8 --target 17 --out data/f22_Qwen3-8B.jsonl > logs/f22_q8.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $PY f22_digitpos.py --model $B/Qwen3-14B --model-name Qwen3-14B --heads $H14 --target 14 --out data/f22_Qwen3-14B.jsonl > logs/f22_q14.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $PY f22_digitpos.py --model $M32 --model-name Qwen3-32B --heads $H32T --target 13 --out data/f22_Qwen3-32B.jsonl > logs/f22_q32.log 2>&1
  echo done > logs/rev2_g1.done
}
g0 & g1 & wait
