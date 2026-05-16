#!/usr/bin/env bash
set -uo pipefail

ROOT="/root/autodl-tmp/.autodl/kv_cache_middleware"
PY="/root/miniconda3/bin/python"
MODEL="/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
SRC_HF_DIR="${ROOT}/benchmarks/results/paper/quality_v3/longbench_full_official_llama"
OUT_DIR="${ROOT}/benchmarks/results/paper/quality_v4/longbench_fixed_budget_llama"
mkdir -p "${OUT_DIR}"
cd "${ROOT}" || exit 1
export KV_MIDDLEWARE_BENCH_PROGRESS=1

TASKS=(
  narrativeqa qasper multifieldqa_en
  hotpotqa 2wikimqa musique
  gov_report qmsum multi_news
  trec triviaqa samsum
  passage_count passage_retrieval_en
  lcc repobench-p
)
METHODS=(hf_vanilla off_compress_page16_b1024 off_compress_page16_b2048 off_compress_page16_b4096)
GPU_MAP="hf_vanilla:0.60,off_compress_page16_b1024:0.35,off_compress_page16_b2048:0.35,off_compress_page16_b4096:0.35"

is_valid_json() {
  local file="$1"
  [ -s "$file" ] || return 1
  "$PY" - "$file" "$MODEL" <<'PY'
import json, sys
p, model = sys.argv[1], sys.argv[2]
try:
    d=json.load(open(p, encoding='utf-8'))
    if d.get('meta', {}).get('model_name') != model:
        raise SystemExit(1)
    rows=d.get('rows', [])
    if not rows:
        raise SystemExit(1)
    r=rows[0]
    if not r.get('task') or not r.get('method'):
        raise SystemExit(1)
except Exception:
    raise SystemExit(1)
PY
}

maybe_copy_hf() {
  local task="$1"
  local src="${SRC_HF_DIR}/longbench_full_official_hf_vanilla_${task}.json"
  local dst="${OUT_DIR}/longbench_full_official_hf_vanilla_${task}.json"
  if [ "${FORCE:-0}" != "1" ] && is_valid_json "$dst"; then
    return 0
  fi
  if is_valid_json "$src"; then
    cp "$src" "$dst"
    echo "[$(date '+%F %T')] COPIED HF baseline task=${task}"
    return 0
  fi
  return 1
}

run_one() {
  local method="$1"
  local task="$2"
  local out="${OUT_DIR}/longbench_full_official_${method}_${task}.json"
  if [ "$method" = "hf_vanilla" ] && maybe_copy_hf "$task"; then
    return 0
  fi
  if [ "${FORCE:-0}" != "1" ] && is_valid_json "$out"; then
    echo "[$(date '+%F %T')] SKIP existing method=${method} task=${task} out=${out}"
    return 0
  fi
  echo "[$(date '+%F %T')] START method=${method} task=${task} out=${out}"
  "$PY" benchmarks/benchmark_quality_longbench_v3.py \
    --mode formal \
    --model-name "${MODEL}" \
    --tasks "${task}" \
    --methods "${method}" \
    --samples-per-task 1000000 \
    --use-official-max-gen \
    --max-prompt-tokens 32768 \
    --concurrency 1 \
    --gpu-mem-frac-map "${GPU_MAP}" \
    --out "${out}"
  local rc=$?
  echo "[$(date '+%F %T')] DONE method=${method} task=${task} rc=${rc}"
  "$PY" - "$out" <<'PY' || true
import json, sys
p=sys.argv[1]
try:
    d=json.load(open(p, encoding='utf-8')); r=d['rows'][0]
    print('RESULT', r.get('method'), r.get('task'), 'score=', r.get('score'), 'completed=', r.get('completed_requests'), '/', r.get('requested_requests'), 'status=', r.get('status'), 'budget=', r.get('retain_budget_tokens'), 'mode=', r.get('compression_mode'), 'rebuild=', r.get('decode_rebuild_steps'), 'mat_bytes=', r.get('decode_materialize_kv_bytes'), 'miss=', r.get('resident_miss_steps'), 'err=', r.get('error_reason'))
except Exception as exc:
    print('RESULT_PARSE_FAILED', p, exc)
PY
}

for method in "${METHODS[@]}"; do
  for task in "${TASKS[@]}"; do
    run_one "$method" "$task"
  done
done

"$PY" benchmarks/summarize_longbench_full_official.py --input-dir "${OUT_DIR}" --out-dir "${OUT_DIR}" || true

echo "[$(date '+%F %T')] ALL DONE llama longbench fixed budget"