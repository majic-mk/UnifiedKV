#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
LOG=/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/run_fixedfrac_rechecks_20260327.log
run_one() {
  local group="$1"
  local frac="$2"
  local suffix="$3"
  echo "=== START ${group} ${frac} $(date '+%F %T') ===" | tee -a "$LOG"
  pkill -f "benchmark_exp1_capacity_table.py --worker" || true
  sleep 2
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader | tee -a "$LOG"
  $PY benchmark_exp1_capacity_table.py \
    --worker \
    --model-name "$MODEL" \
    --max-new-tokens 512 \
    --repeats 1 \
    --gpu-mem-frac-fallback-step 0 \
    --gpu-mem-frac-min "$frac" \
    --worker-group "$group" \
    --worker-input-len 8192 \
    --worker-concurrency 128 \
    --worker-gpu-mem-frac-initial "$frac" \
    --worker-out "/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/${group}_8k_c128_f${suffix}_fixed_20260327.json" \
    --worker-step-jsonl "/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/${group}_8k_c128_f${suffix}_fixed_20260327.steps.jsonl" \
    --worker-step-jsonl-every 1 \
    > "/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/${group}_8k_c128_f${suffix}_fixed_20260327.log" 2>&1 || true
  echo "=== END ${group} ${frac} $(date '+%F %T') ===" | tee -a "$LOG"
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader | tee -a "$LOG"
  sleep 5
}
run_one p2_only_compress 0.70 070
run_one p2_only_compress 0.60 060
run_one main_auto_compress 0.50 050

