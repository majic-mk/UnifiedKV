#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/.autodl/kv_cache_middleware
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export KV_MIDDLEWARE_BENCH_PROGRESS=1
export KV_MIDDLEWARE_SELECTED_WRITEBACK=1
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct
OUT_DIR=benchmarks/results/paper/quality_v5/longbench_fixed_budget_selected_writeback_smoke_llama
mkdir -p "$OUT_DIR"
TASKS=(narrativeqa gov_report qmsum)
METHODS=(off_compress_page16_b1024 off_compress_page16_b2048 off_compress_page16_b4096)
GPU_MAP="hf_vanilla:0.60,off_compress_page16_b1024:0.35,off_compress_page16_b2048:0.35,off_compress_page16_b4096:0.35"
for method in "${METHODS[@]}"; do
  for task in "${TASKS[@]}"; do
    out="$OUT_DIR/longbench_smoke_${method}_${task}.json"
    echo "[smoke] method=$method task=$task out=$out"
    "$PY" benchmarks/benchmark_quality_longbench_v3.py \
      --mode formal \
      --model-name "$MODEL" \
      --tasks "$task" \
      --methods "$method" \
      --samples-per-task 1 \
      --use-official-max-gen \
      --max-prompt-tokens 32768 \
      --concurrency 1 \
      --gpu-mem-frac-map "$GPU_MAP" \
      --out "$out"
  done
done
