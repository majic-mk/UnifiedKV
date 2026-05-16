#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/.autodl/kv_cache_middleware
PY=/root/miniconda3/bin/python
OUT_DIR=benchmarks/results/paper/quality_v3
mkdir -p "$OUT_DIR"
TASKS=(passage_retrieval_en multifieldqa_en hotpotqa 2wikimqa musique)
METHODS=(off_compress_page16 p2_page16_offline)
for METHOD in "${METHODS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    echo "===== LongBench isolated method=${METHOD} task=${TASK} ====="
    $PY benchmarks/benchmark_quality_longbench_v3.py \
      --mode gate \
      --tasks "$TASK" \
      --methods "$METHOD" \
      --out "$OUT_DIR/longbench_gate_${METHOD}_${TASK}.json"
  done
done