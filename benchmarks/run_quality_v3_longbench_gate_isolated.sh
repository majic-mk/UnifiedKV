#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/.autodl/kv_cache_middleware
PY=/root/miniconda3/bin/python
OUT_DIR=benchmarks/results/paper/quality_v3
mkdir -p "$OUT_DIR"
for METHOD in hf_vanilla off_compress_page16 p2_page16_offline; do
  echo "===== LongBench isolated method=${METHOD} ====="
  $PY benchmarks/benchmark_quality_longbench_v3.py \
    --mode gate \
    --methods "$METHOD" \
    --out "$OUT_DIR/longbench_gate_${METHOD}.json"
done