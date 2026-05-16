#!/bin/bash
cd /root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Running P2 target=384 benchmark ==="
/root/miniconda3/bin/python benchmark_exp1_capacity_table.py \
  --worker \
  --model-name /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --max-new-tokens 1024 \
  --repeats 1 \
  --gpu-mem-frac-fallback-step 0.1 \
  --gpu-mem-frac-min 0.2 \
  --chunk-size 512 \
  --prefill-batch-size 4 \
  --decode-micro-batch-size 0 \
  --decode-active-cap-initial 0 \
  --max-decode-active-cap 0 \
  --worker-group p2_only_compress \
  --worker-input-len 32768 \
  --worker-concurrency 64 \
  --worker-gpu-mem-frac-initial 0.2 \
  --worker-fixed-gpu-mem-frac \
  --worker-out results/paper/p2_target_free_sweep_32k64_n1024/p2_target384_c64_n1024.json \
  --worker-step-jsonl results/paper/p2_target_free_sweep_32k64_n1024/p2_target384_c64_n1024.steps.jsonl \
  --worker-step-jsonl-every 100 \
  --engine-overrides-json '{"p2_target_free_blocks":384}'

echo "=== P2 target=384 done ==="

echo "=== Running P2 target=448 benchmark ==="
/root/miniconda3/bin/python benchmark_exp1_capacity_table.py \
  --worker \
  --model-name /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --max-new-tokens 1024 \
  --repeats 1 \
  --gpu-mem-frac-fallback-step 0.1 \
  --gpu-mem-frac-min 0.2 \
  --chunk-size 512 \
  --prefill-batch-size 4 \
  --decode-micro-batch-size 0 \
  --decode-active-cap-initial 0 \
  --max-decode-active-cap 0 \
  --worker-group p2_only_compress \
  --worker-input-len 32768 \
  --worker-concurrency 64 \
  --worker-gpu-mem-frac-initial 0.2 \
  --worker-fixed-gpu-mem-frac \
  --worker-out results/paper/p2_target_free_sweep_32k64_n1024/p2_target448_c64_n1024.json \
  --worker-step-jsonl results/paper/p2_target_free_sweep_32k64_n1024/p2_target448_c64_n1024.steps.jsonl \
  --worker-step-jsonl-every 100 \
  --engine-overrides-json '{"p2_target_free_blocks":448}'

echo "=== P2 target=448 done ==="
echo "=== All benchmarks complete ==="
