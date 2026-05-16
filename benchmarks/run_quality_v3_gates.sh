#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/.autodl/kv_cache_middleware
PY=/root/miniconda3/bin/python
OUT_DIR=benchmarks/results/paper/quality_v3
mkdir -p "$OUT_DIR"
$PY benchmarks/benchmark_quality_passkey_v3.py --mode raw_sanity --out "$OUT_DIR/passkey_raw_sanity.json" 2>&1 | tee "$OUT_DIR/passkey_raw_sanity.log"
$PY benchmarks/benchmark_quality_passkey_v3.py --mode gate --out "$OUT_DIR/passkey_gate.json" 2>&1 | tee "$OUT_DIR/passkey_gate.log"
$PY benchmarks/benchmark_quality_longbench_v3.py --mode gate --out "$OUT_DIR/longbench_gate.json" 2>&1 | tee "$OUT_DIR/longbench_gate.log"