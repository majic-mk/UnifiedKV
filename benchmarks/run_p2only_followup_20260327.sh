#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks
LOG=/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/p2only_8k128_followup_20260327.log
PY=/root/miniconda3/bin/python
wait_for_main() {
  while pgrep -af "20260327_mainauto_8k_c128_f050_clean" >/dev/null; do
    echo "[$(date '+%F %T')] waiting main_auto 0.50" >> "$LOG"
    sleep 20
  done
}
run_case() {
  local frac="$1"
  local tag="20260327_p2only_8k_c128_f${frac/./}_clean"
  echo "[$(date '+%F %T')] START ${tag}" >> "$LOG"
  $PY benchmark_exp1_capacity_table.py \
    --worker \
    --worker-group p2_only_compress \
    --worker-input-len 8192 \
    --worker-concurrency 128 \
    --worker-gpu-mem-frac-initial "$frac" \
    --max-new-tokens 512 \
    --worker-out "/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/${tag}.json" \
    --worker-step-jsonl "/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/${tag}.steps.jsonl" \
    --worker-step-jsonl-every 1 \
    > "/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/${tag}.log" 2>&1 || true
  echo "[$(date '+%F %T')] END ${tag}" >> "$LOG"
}
: > "$LOG"
echo "[$(date '+%F %T')] followup watcher started" >> "$LOG"
wait_for_main
run_case 0.70
run_case 0.60
run_case 0.50
echo "[$(date '+%F %T')] all followup cases finished" >> "$LOG"
