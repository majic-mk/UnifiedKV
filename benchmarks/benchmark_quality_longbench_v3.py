#!/usr/bin/env python3
import argparse, importlib.util, json, os, random, sys, time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "core", Path(__file__).resolve().parent / "configs", Path(__file__).resolve().parent):
    if str(p) not in sys.path: sys.path.insert(0, str(p))

from benchmark_internal_common import is_hf_style_method, parse_method_frac_map, parse_methods, run_internal_prompt_batches
from benchmark_vllm_common import load_tokenizer
from hf_style_common import run_hf_style_prompt_batches
from strategy_groups import LOCAL_MODEL_PATH

GATE_TASKS = ["passage_retrieval_en", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]
FORMAL_TASKS = ["narrativeqa","qasper","multifieldqa_en","hotpotqa","2wikimqa","musique","gov_report","qmsum","multi_news","trec","triviaqa","samsum","passage_count","passage_retrieval_en","lcc","repobench-p"]
TASK_CATEGORIES = {"narrativeqa":"Single-Doc QA","qasper":"Single-Doc QA","multifieldqa_en":"Single-Doc QA","hotpotqa":"Multi-Doc QA","2wikimqa":"Multi-Doc QA","musique":"Multi-Doc QA","gov_report":"Summarization","qmsum":"Summarization","multi_news":"Summarization","trec":"Few-shot","triviaqa":"Few-shot","samsum":"Few-shot","passage_count":"Synthetic","passage_retrieval_en":"Synthetic","lcc":"Code","repobench-p":"Code"}
DEFAULT_METHODS = "hf_vanilla,off_compress_page16,p2_page16_offline"
DEFAULT_GPU_MEM_FRAC_MAP = {"off_compress_page16":0.60,"p2_page16_offline":0.60,"legacy_off_raw_page16":0.60,"off_compress_page16_b1024":0.35,"off_compress_page16_b2048":0.35,"off_compress_page16_b4096":0.35}
DEFAULT_SEED = 20260415

def csv_ints(s: str) -> List[int]: return [int(x.strip()) for x in str(s).split(',') if x.strip()]
def csv_strs(s: str) -> List[str]: return [x.strip() for x in str(s).split(',') if x.strip()]

def resolve_dataset_root(root: str) -> Path:
    if str(root).strip():
        p = Path(root).expanduser();
        if p.exists(): return p.resolve()
        raise FileNotFoundError(root)
    for p in [Path('/root/autodl-tmp/datasets/LongBench/data/data'), Path('/root/autodl-tmp/datasets/LongBench')]:
        if p.exists(): return p.resolve()
    raise FileNotFoundError('LongBench dataset root not found')

def find_repo(dataset_root: Path, explicit: str) -> Path:
    if str(explicit).strip(): return Path(explicit).resolve()
    for p in [Path('/root/autodl-tmp/datasets/LongBench/repo/LongBench'), dataset_root/'repo'/'LongBench', dataset_root/'LongBench'/'repo'/'LongBench']:
        if (p/'eval.py').exists(): return p.resolve()
    for hit in dataset_root.rglob('eval.py'):
        if hit.parent.name == 'LongBench' and (hit.parent/'config'/'dataset2prompt.json').exists(): return hit.parent.resolve()
    raise FileNotFoundError('official LongBench repo not found')

def resolve_task_file(root: Path, task: str) -> Path:
    for p in [root/f'{task}.jsonl', root/'data'/f'{task}.jsonl', root/'data'/'data'/f'{task}.jsonl']:
        if p.exists(): return p.resolve()
    hits = list(root.rglob(f'{task}.jsonl'))
    if hits: return hits[0].resolve()
    raise FileNotFoundError(f'{task}.jsonl under {root}')

def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f: return [json.loads(line) for line in f if line.strip()]

def load_assets(repo: Path):
    sys.path.insert(0, str(repo))
    prompt_map = json.loads((repo/'config'/'dataset2prompt.json').read_text(encoding='utf-8'))
    maxgen_map = {str(k): int(v) for k,v in json.loads((repo/'config'/'dataset2maxlen.json').read_text(encoding='utf-8')).items()}
    spec = importlib.util.spec_from_file_location('longbench_official_eval', str(repo/'eval.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return prompt_map, maxgen_map, dict(mod.dataset2metric)

def middle_truncate(tokenizer, prompt: str, max_tokens: int) -> Tuple[str,int,int]:
    ids = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids; orig=len(ids)
    if orig <= int(max_tokens): return prompt, orig, 0
    half = int(max_tokens)//2; kept = ids[:half] + ids[-(int(max_tokens)-half):]
    return tokenizer.decode(kept, skip_special_tokens=True), len(kept), orig

def build_samples(tokenizer, task: str, records: Sequence[Dict[str,Any]], n: int, max_prompt_tokens: int, seed: int, prompt_map: Dict[str,str]) -> List[Dict[str,Any]]:
    candidates=[]
    for idx, rec in enumerate(records):
        try: raw = str(prompt_map[task]).format(**rec)
        except Exception: continue
        prompt, toks, trunc = middle_truncate(tokenizer, raw, max_prompt_tokens)
        answers = rec.get('answers', []); answers = answers if isinstance(answers, list) else [answers]
        candidates.append({"task":task,"category":TASK_CATEGORIES.get(task,"Other"),"sample_idx":idx,"prompt":prompt,"prompt_tokens":toks,"truncated_from_tokens":trunc,"answers":[str(x) for x in answers],"all_classes":list(rec.get('all_classes', []) or []),"length":int(rec.get('length', toks) or toks)})
    if not candidates: raise RuntimeError(f'no usable samples for task={task}')
    rng = random.Random(int(seed) ^ sum((i+1)*ord(c) for i,c in enumerate(task)))
    if len(candidates) <= int(n): rng.shuffle(candidates); return candidates
    return rng.sample(candidates, int(n))

def score_official(task: str, output: str, answers: Sequence[str], all_classes: Sequence[str], metric_map: Dict[str,Any]) -> Tuple[float,int]:
    if not answers: return 0.0, 0
    pred = str(output or '')
    if task in ["trec","triviaqa","samsum","lsht"]: pred = pred.lstrip('\n').split('\n')[0]
    metric = metric_map[task]; score = 0.0
    for ans in answers: score = max(score, float(metric(pred, str(ans), all_classes=list(all_classes))))
    return 100.0*score, 1

def make_eval(metric_map: Dict[str,Any]):
    def _eval(item: Dict[str,Any], output: str) -> Dict[str,Any]:
        score, scoreable = score_official(str(item['task']), output, list(item.get('answers', []) or []), list(item.get('all_classes', []) or []), metric_map)
        return {"valid": int(scoreable and bool(str(output or '').strip())), "task": str(item['task']), "category": str(item.get('category','Other')), "sample_idx": int(item.get('sample_idx',0)), "prompt_tokens": int(item.get('prompt_tokens',0)), "truncated_from_tokens": int(item.get('truncated_from_tokens',0)), "score": float(score if str(output or '').strip() else 0.0), "scoreable": int(scoreable), "official_metric": getattr(metric_map.get(str(item['task'])), '__name__', 'unknown')}
    return _eval

def telemetry(summary: Dict[str,Any]) -> Dict[str,Any]:
    return {
        "decode_backend": str(summary.get('decode_backend','')),
        "compression_mode": str(summary.get('compression_mode','')),
        "retain_budget_tokens": int(summary.get('retain_budget_tokens',0) or 0),
        "effective_retained_tokens": int(summary.get('effective_retained_tokens',0) or 0),
        "retained_block_count": int(summary.get('retained_block_count',0) or 0),
        "decode_page16_native_steps": int(summary.get('decode_page16_native_steps',0) or 0),
        "decode_rebuild_steps": int(summary.get('decode_rebuild_steps',0) or 0),
        "decode_materialize_kv_bytes": int(summary.get('decode_materialize_kv_bytes',0) or 0),
        "p2_attempted_steps": int(summary.get('p2_attempted_steps',0) or 0),
        "p2_success_steps": int(summary.get('p2_success_steps',0) or 0),
        "p2_offload_blocks": int(summary.get('p2_ready_offload_blocks_total', summary.get('p2_ready_offload_blocks_last',0)) or 0),
        "resident_miss_steps": max(int(summary.get('decode_page16_native_resident_miss_steps',0) or 0), int(summary.get('decode_paged_direct_resident_miss_steps',0) or 0)),
        "kv_logical_block_size": int(summary.get('kv_logical_block_size',0) or 0),
        "flash_attn_enabled": int(summary.get('flash_attn_enabled',0) or 0),
        "selected_writeback_enabled": int(summary.get('selected_writeback_enabled',0) or 0),
        "prefill_writeback_backend": str(summary.get('prefill_writeback_backend','')),
        "gpu_selected_writeback_steps": int(summary.get('gpu_selected_writeback_steps',0) or 0),
        "cpu_selected_compaction_steps": int(summary.get('cpu_selected_compaction_steps',0) or 0),
        "gpu_writeback_oom_fallbacks": int(summary.get('gpu_writeback_oom_fallbacks',0) or 0),
        "writeback_transaction_rollbacks": int(summary.get('writeback_transaction_rollbacks',0) or 0),
        "raw_kv_cpu_stash_bytes": int(summary.get('raw_kv_cpu_stash_bytes',0) or 0),
        "selected_global_block_count": int(summary.get('selected_global_block_count',0) or 0),
        "writeback_est_required_gb": float(summary.get('writeback_est_required_gb',0.0) or 0.0),
        "writeback_free_gb": float(summary.get('writeback_free_gb',0.0) or 0.0),
        "writeback_block_selection_shared_layers": int(summary.get('writeback_block_selection_shared_layers',0) or 0),
        "score_full_attention_materialized": int(summary.get('score_full_attention_materialized',0) or 0),
        "min_free_blocks": int(summary.get('min_free_blocks',-1) or -1),
    }

def run_task_method(model: str, method: str, samples: Sequence[Dict[str,Any]], tokenizer, concurrency: int, max_new: int, frac_map: Dict[str,float], eval_fn) -> Dict[str,Any]:
    t0 = time.perf_counter()
    if is_hf_style_method(method):
        summary = run_hf_style_prompt_batches(model, method, list(samples), concurrency, max_new, 1, eval_fn, tokenizer=tokenizer)
        tel = {"decode_backend":"hf_vanilla","decode_page16_native_steps":0,"decode_rebuild_steps":0,"decode_materialize_kv_bytes":0,"p2_attempted_steps":0,"p2_success_steps":0,"p2_offload_blocks":0,"resident_miss_steps":0,"kv_logical_block_size":0,"min_free_blocks":-1}
    else:
        if method not in frac_map: raise ValueError(f'missing gpu_mem_frac for {method}')
        summary = run_internal_prompt_batches(model, method, float(frac_map[method]), [str(x['prompt']) for x in samples], list(samples), concurrency, max_new, tokenizer, 1, eval_fn)
        tel = telemetry(summary)
        selected_writeback_active = int(tel.get('selected_writeback_enabled', 0) or 0) == 1
        selected_writeback_forced = str(os.environ.get('KV_MIDDLEWARE_SELECTED_WRITEBACK', '')).strip().lower() in {'1','true','yes','on'}
        if (selected_writeback_active or selected_writeback_forced) and int(tel.get('flash_attn_enabled', 0)) != 1:
            summary['error_reason'] = 'flash_attn_fallback_invalid'
            summary['status'] = 'Failed/Invalid'
    rows = list(summary.get('rows', [])); req=max(1, len(samples)); completed=sum(int(r.get('completed',0)) for r in rows); valid=sum(int(r.get('valid',0)) for r in rows); score=sum(float(r.get('score',0.0)) for r in rows)/req
    return {"task": str(samples[0].get('task','')) if samples else '', "category": str(samples[0].get('category','Other')) if samples else 'Other', "method": method, "concurrency": concurrency, "max_new_tokens": max_new, "gpu_mem_frac": float(frac_map.get(method,0.0)), "requested_requests": len(samples), "completed_requests": completed, "completion_rate": completed/req, "valid_completion_rate": valid/req, "score": score, "status": str(summary.get('status','')), "tokens_per_sec": float(summary.get('tokens_per_sec',0.0) or 0.0), "wall_clock_total_runtime_ms": float(summary.get('wall_clock_total_runtime_ms',0.0) or 0.0), "runner_wall_clock_s": round(time.perf_counter()-t0,3), "error_reason": str(summary.get('error_reason','')), **tel, "raw_summary": {k:v for k,v in summary.items() if k != 'rows'}, "rows": rows}

def completed(row: Dict[str,Any]) -> bool:
    return float(row.get('completion_rate',0.0)) >= 1.0 and float(row.get('valid_completion_rate',0.0)) >= 1.0 and not str(row.get('error_reason','')).strip()

def macro(rows: Sequence[Dict[str,Any]], tasks: Sequence[str]) -> Dict[str,Any]:
    cats=["Single-Doc QA","Multi-Doc QA","Summarization","Few-shot","Synthetic","Code"]; table=[]
    for m in sorted({r['method'] for r in rows}):
        mr=[r for r in rows if r['method']==m]; item={"method":m}; scores=[]; comp=0; fail=0
        for cat in cats:
            cr=[r for r in mr if r.get('category')==cat]; ok=[r for r in cr if completed(r)]
            item[cat] = round(mean(float(r.get('score',0.0)) for r in ok),2) if ok else ("Failed/OOM" if cr else "NA")
            scores.extend(float(r.get('score',0.0)) for r in ok); comp += len(ok); fail += max(0, len(cr)-len(ok))
        item['Avg'] = round(mean(scores),2) if scores else "NA"; item['Completed / Total']=f"{comp}/{len(tasks)}"; item['Failed']=fail; table.append(item)
    per_task=[{"method":r['method'],"task":r['task'],"category":r.get('category','Other'),"score":round(float(r.get('score',0.0)),2),"completed":int(completed(r)),"status":r.get('status',''),"error_reason":r.get('error_reason',''),"p2_attempted_steps":int(r.get('p2_attempted_steps',0) or 0),"p2_success_steps":int(r.get('p2_success_steps',0) or 0),"resident_miss_steps":int(r.get('resident_miss_steps',0) or 0)} for r in rows]
    return {"main_table": table, "per_task": per_task}

def main():
    ap=argparse.ArgumentParser(description='LongBench quality v3 runner with official prompts and metrics.')
    ap.add_argument('--mode', choices=['gate','formal'], default='gate'); ap.add_argument('--model-name', default=LOCAL_MODEL_PATH); ap.add_argument('--dataset-root', default=''); ap.add_argument('--longbench-repo', default='')
    ap.add_argument('--tasks', default=''); ap.add_argument('--methods', default=DEFAULT_METHODS); ap.add_argument('--samples-per-task', type=int, default=0); ap.add_argument('--max-prompt-tokens', type=int, default=32768); ap.add_argument('--max-new-tokens', type=int, default=64); ap.add_argument('--use-official-max-gen', action='store_true'); ap.add_argument('--concurrency', type=int, default=1); ap.add_argument('--seed', type=int, default=DEFAULT_SEED); ap.add_argument('--gpu-mem-frac-map', default=''); ap.add_argument('--out', required=True)
    args=ap.parse_args(); root=resolve_dataset_root(args.dataset_root); repo=find_repo(root, args.longbench_repo); prompt_map,maxgen_map,metric_map=load_assets(repo)
    tasks = csv_strs(args.tasks) if args.tasks.strip() else (GATE_TASKS if args.mode=='gate' else FORMAL_TASKS); methods=parse_methods(args.methods, allow_vllm=False); n=int(args.samples_per_task or (20 if args.mode=='gate' else 100)); frac_map=parse_method_frac_map(args.gpu_mem_frac_map, defaults=DEFAULT_GPU_MEM_FRAC_MAP); tok=load_tokenizer(args.model_name); eval_fn=make_eval(metric_map)
    rows=[]; manifest={}
    for task in tasks:
        path=resolve_task_file(root, task); records=load_records(path); samples=build_samples(tok, task, records, n, args.max_prompt_tokens, args.seed, prompt_map); max_gen=int(maxgen_map.get(task,args.max_new_tokens)) if args.use_official_max_gen else int(args.max_new_tokens)
        manifest[task]={"task_file":str(path),"available_records":len(records),"selected_samples":len(samples),"max_new_tokens_effective":max_gen,"prompt_tokens_min":min(int(x['prompt_tokens']) for x in samples),"prompt_tokens_max":max(int(x['prompt_tokens']) for x in samples),"prompt_tokens_mean":round(mean(int(x['prompt_tokens']) for x in samples),2)}
        print(f"[longbench] task={task} samples={len(samples)} max_gen={max_gen}", flush=True)
        for m in methods:
            print(f"[longbench] running method={m} task={task}", flush=True); r=run_task_method(str(args.model_name), m, samples, tok, args.concurrency, max_gen, frac_map, eval_fn); rows.append(r)
            print(json.dumps({k:r.get(k) for k in ['method','task','score','completion_rate','valid_completion_rate','p2_attempted_steps','resident_miss_steps','error_reason']}, ensure_ascii=False), flush=True)
    payload={"meta":{"task":"longbench_quality_v3","mode":args.mode,"model_name":args.model_name,"dataset_root":str(root),"longbench_repo":str(repo),"tasks":tasks,"methods":methods,"samples_per_task":n,"max_prompt_tokens":args.max_prompt_tokens,"max_new_tokens_arg":args.max_new_tokens,"use_official_max_gen":bool(args.use_official_max_gen),"concurrency":args.concurrency,"seed":args.seed,"metric_note":"Official LongBench metric functions from repo/LongBench/eval.py; scores are 0-100.","gate_note":"Gate mode detects severe quality regressions only; not a paper main-table result.","gpu_mem_frac_map":{str(k):float(v) for k,v in frac_map.items()}},"sample_manifest":manifest,"rows":rows,"tables":macro(rows,tasks)}
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'); print(f"Saved: {out}")
if __name__ == '__main__': main()
