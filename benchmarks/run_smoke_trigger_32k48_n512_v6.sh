#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks
rm -f smoke_trigger_main_auto_32k48_n512_v6.json smoke_trigger_p2_only_32k48_n512_v6.json
/root/miniconda3/bin/python benchmark_exp1_capacity_table.py --worker --model-name /root/autodl-tmp/models/Qwen2.5-7B-Instruct --max-new-tokens 512 --repeats 1 --gpu-mem-frac-fallback-step 0.02 --gpu-mem-frac-min 0.18 --worker-group main_auto_compress --worker-input-len 32768 --worker-concurrency 48 --worker-gpu-mem-frac-initial 0.22 --worker-out smoke_trigger_main_auto_32k48_n512_v6.json > /tmp/smoke_trigger_main_auto_32k48_n512_v6.log 2>&1
/root/miniconda3/bin/python benchmark_exp1_capacity_table.py --worker --model-name /root/autodl-tmp/models/Qwen2.5-7B-Instruct --max-new-tokens 512 --repeats 1 --gpu-mem-frac-fallback-step 0.02 --gpu-mem-frac-min 0.18 --worker-group p2_only_compress --worker-input-len 32768 --worker-concurrency 48 --worker-gpu-mem-frac-initial 0.22 --worker-out smoke_trigger_p2_only_32k48_n512_v6.json > /tmp/smoke_trigger_p2_only_32k48_n512_v6.log 2>&1
