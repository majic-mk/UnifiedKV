#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/.autodl/kv_cache_middleware
PY=/root/miniconda3/bin/python
OUT_DIR=benchmarks/results/paper/quality_v3
mkdir -p "$OUT_DIR"
for METHOD in hf_vanilla off_compress_page16 p2_page16_offline off_compress_page16_r015 p2_page16_offline_r015; do
  echo "===== passage official maxgen method=${METHOD} ====="
  $PY benchmarks/benchmark_quality_longbench_v3.py \
    --mode gate \
    --tasks passage_retrieval_en \
    --methods "$METHOD" \
    --use-official-max-gen \
    --gpu-mem-frac-map off_compress_page16:0.60,p2_page16_offline:0.60,off_compress_page16_r015:0.60,p2_page16_offline_r015:0.60 \
    --out "$OUT_DIR/longbench_gate_officialmax_${METHOD}_passage_retrieval_en.json"
done