#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../runtime.sh"
cd "$REPO_DIR"
PY=${PYTHON}; T=${TOKENIZER_DIR}
G=${MODEL_DIR}/gemma-4-31B-it; MU=${MODEL_DIR}/Muse-Glimmer-30B; B=${MODEL_DIR}
g0() {
  CUDA_VISIBLE_DEVICES=0 $PY gemma_circuit.py --model $G --tokenizer $T/gemma-4-31B-it --model-name gemma-4-31B-it --outdir data/f15 --phases f8b --f8b-tag _v2 > logs/f17_dla.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY repulsion_freegen.py --model $G --tokenizer $T/gemma-4-31B-it --family gemma --model-name gemma-4-31B-it --out data/freegen_gemma-4-31B-it.jsonl > logs/f16_gemma.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY f18_anchoring.py --model $G --tokenizer $T/gemma-4-31B-it --family gemma --model-name gemma-4-31B-it --out data/f18_gemma-4-31B-it.jsonl > logs/f18_gemma.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY f18_anchoring.py --model ${MODEL_DIR}/Qwen3-32B --family qwen --model-name Qwen3-32B --out data/f18_Qwen3-32B.jsonl > logs/f18_q32.log 2>&1
  echo done > logs/rev_g0.done
}
g1() {
  CUDA_VISIBLE_DEVICES=1 $PY repulsion_freegen.py --model $MU --tokenizer $T/Muse-Glimmer-30B --family muse --model-name Muse-Glimmer-30B --out data/freegen_Muse-Glimmer-30B.jsonl > logs/f16_muse.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $PY f18_anchoring.py --model $MU --tokenizer $T/Muse-Glimmer-30B --family muse --model-name Muse-Glimmer-30B --out data/f18_Muse-Glimmer-30B.jsonl > logs/f18_muse.log 2>&1
  for m in Qwen3-4B Qwen3-8B; do CUDA_VISIBLE_DEVICES=1 $PY f18_anchoring.py --model ${MODEL_DIR}/$m --family qwen --model-name $m --out data/f18_$m.jsonl > logs/f18_$m.log 2>&1; done
  CUDA_VISIBLE_DEVICES=1 $PY f18_anchoring.py --model $B/Qwen3-14B --family qwen --model-name Qwen3-14B --out data/f18_Qwen3-14B.jsonl > logs/f18_q14.log 2>&1
  echo done > logs/rev_g1.done
}
g0 & g1 & wait
