#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks
TS=$(date +%Y%m%d_%H%M%S)
LOG=/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/p2_block_jsonl_${TS}.log
PY=/root/miniconda3/bin/python
run_one () {
  GROUP="$1"
  PREFIX="/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/${GROUP}_32k48_n512_f022_${TS}"
  echo "[$(date '+%F %T')] START ${GROUP}" | tee -a "$LOG"
  $PY benchmark_exp1_capacity_table.py     --worker     --model-name /root/autodl-tmp/models/Qwen2.5-7B-Instruct     --max-new-tokens 512     --repeats 1     --gpu-mem-frac-fallback-step 1.0     --gpu-mem-frac-min 0.22     --decode-micro-batch-size 0     --decode-active-cap-initial 0     --max-decode-active-cap 0     --worker-group "$GROUP"     --worker-input-len 32768     --worker-concurrency 48     --worker-gpu-mem-frac-initial 0.22     --worker-out "${PREFIX}.json"     --worker-step-jsonl "${PREFIX}.steps.jsonl"     --worker-step-jsonl-every 1 >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] DONE ${GROUP}" | tee -a "$LOG"
}
run_one p2_only_compress
run_one main_auto_compress
echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG"
