#!/usr/bin/env bash
set -uo pipefail

ROOT="/root/autodl-tmp/.autodl/kv_cache_middleware"
PY="/root/miniconda3/bin/python"
MODEL="/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
OUT_DIR="${ROOT}/benchmarks/results/paper/quality_v4/passkey_fixed_budget_llama"
mkdir -p "${OUT_DIR}"
cd "${ROOT}" || exit 1

METHODS="hf_vanilla,off_compress_page16_b1024,off_compress_page16_b2048,off_compress_page16_b4096"
GPU_MAP="hf_vanilla:0.60,off_compress_page16_b1024:0.60,off_compress_page16_b2048:0.60,off_compress_page16_b4096:0.60"

MODE="${1:-gate}"
OUT="${OUT_DIR}/passkey_fixed_budget_${MODE}.json"
"$PY" benchmarks/benchmark_quality_passkey_v3.py \
  --mode "${MODE}" \
  --model-name "${MODEL}" \
  --methods "${METHODS}" \
  --gpu-mem-frac-map "${GPU_MAP}" \
  --concurrency 1 \
  --max-new-tokens 32 \
  --out "${OUT}"