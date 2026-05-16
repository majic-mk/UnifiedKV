#!/usr/bin/env bash
set -uo pipefail

ROOT="/root/autodl-tmp/.autodl/kv_cache_middleware"
PY="/root/miniconda3/bin/python"
MODEL="/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
OUT_DIR="${ROOT}/benchmarks/results/paper/quality_v3/longbench_full_official_llama"
mkdir -p "${OUT_DIR}"
cd "${ROOT}" || exit 1

if [ ! -d "${MODEL}" ]; then
  echo "MODEL_NOT_FOUND: ${MODEL}" >&2
  exit 2
fi

TASKS=(
  narrativeqa qasper multifieldqa_en
  hotpotqa 2wikimqa musique
  gov_report qmsum multi_news
  trec triviaqa samsum
  passage_count passage_retrieval_en
  lcc repobench-p
)
METHODS=(hf_vanilla off_compress_page16 off_compress_page16_r015)
GPU_MAP="hf_vanilla:0.60,off_compress_page16:0.60,off_compress_page16_r015:0.60"

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

run_one() {
  local method="$1"
  local task="$2"
  local out="${OUT_DIR}/longbench_full_official_${method}_${task}.json"
  if [ "${FORCE:-0}" != "1" ] && is_valid_json "$out"; then
    echo "[$(date '+%F %T')] SKIP existing method=${method} task=${task} out=${out}"
    return 0
  fi
  echo "[$(date '+%F %T')] START method=${method} task=${task} model=${MODEL} out=${out}"
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
  if [ "$rc" -ne 0 ]; then
    echo "[$(date '+%F %T')] ERROR method=${method} task=${task}; continuing to next task" >&2
  fi
  "$PY" - "$out" <<'PY' || true
import json, sys
p=sys.argv[1]
try:
    d=json.load(open(p, encoding='utf-8')); r=d['rows'][0]
    print('RESULT', r.get('method'), r.get('task'), 'score=', r.get('score'), 'completed=', r.get('completed_requests'), '/', r.get('requested_requests'), 'status=', r.get('status'), 'p2=', r.get('p2_attempted_steps'), 'miss=', r.get('resident_miss_steps'), 'rebuild=', r.get('decode_rebuild_steps'), 'mat_bytes=', r.get('decode_materialize_kv_bytes'), 'err=', r.get('error_reason'))
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

echo "[$(date '+%F %T')] ALL DONE llama longbench official"