#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
LOG=/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/run_main_auto_serial_20260327_serial_clean.log
run_one() {
  local frac="$1"
  local suffix="$2"
  echo "=== START $frac $suffix $(date '+%F %T') ===" | tee -a "$LOG"
  pkill -f "benchmark_exp1_capacity_table.py --worker" || true
  sleep 2
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader | tee -a "$LOG"
  $PY benchmark_exp1_capacity_table.py     --worker     --model-name "$MODEL"     --max-new-tokens 512     --repeats 1     --gpu-mem-frac-fallback-step 1.0     --gpu-mem-frac-min "$frac"     --worker-group main_auto_compress     --worker-input-len 32768     --worker-concurrency 48     --worker-gpu-mem-frac-initial "$frac"     --worker-out "/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/main_auto_compress_32k48_n512_f${suffix}_20260327_serial_clean.json"     --worker-step-jsonl "/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/main_auto_compress_32k48_n512_f${suffix}_20260327_serial_clean.steps.jsonl"     --worker-step-jsonl-every 1     > "/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/main_auto_compress_32k48_n512_f${suffix}_20260327_serial_clean.log" 2>&1 || true
  echo "=== END $frac $suffix $(date '+%F %T') ===" | tee -a "$LOG"
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader | tee -a "$LOG"
  sleep 5
}
run_one 0.22 022
run_one 0.20 020
run_one 0.18 018
