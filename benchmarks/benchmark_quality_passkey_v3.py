#!/usr/bin/env python3
import argparse, gc, json, os, random, re, sys, time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "core", Path(__file__).resolve().parent / "configs", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from transformers import AutoTokenizer
from benchmark_internal_common import is_hf_style_method, parse_method_frac_map, parse_methods, run_internal_prompt_batches
from hf_style_common import run_hf_style_prompt_batches
from strategy_groups import LOCAL_MODEL_PATH
from benchmark_passkey_under_load import build_cases

QUALITY_METHODS = "hf_vanilla,off_compress_page16_b1024,off_compress_page16_b2048,off_compress_page16_b4096"
RAW_METHODS = "hf_vanilla,legacy_off_raw_page16"
DEFAULT_GPU_MEM_FRAC_MAP = {"legacy_off_raw_page16": 0.60, "off_compress_page16": 0.60, "off_compress_page16_b1024": 0.60, "off_compress_page16_b2048": 0.60, "off_compress_page16_b4096": 0.60, "snapkv_dense_b2048": 0.08, "p2_page16_offline": 0.60}
DEFAULT_SEED = 20260415
MODES = {
    "raw_sanity": ("8192,32768", "0.0,0.5,1.0", 5, RAW_METHODS),
    "gate": ("8192,32768", "0.0,0.5,1.0", 5, QUALITY_METHODS),
    "formal": ("1024,2048,4096,8192,16384,32768,65536,131072", "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0", 20, QUALITY_METHODS),
    "offline_sanity": ("32768", "0.5", 5, "off_compress_page16,p2_page16_offline"),
}

def csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(',') if x.strip()]

def csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(',') if x.strip()]

def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try: torch.cuda.ipc_collect()
        except Exception: pass

def normalize_passkey(text: str) -> str:
    return str(text or "").strip()

def first_6digit_compat(text: str) -> str:
    m = re.search(r"\b(\d{6})\b", str(text or ""))
    return m.group(1) if m else ""

def eval_passkey(item: Dict[str, Any], output: str) -> Dict[str, Any]:
    pred = normalize_passkey(output)
    ans = str(item.get("answer", ""))
    compat = first_6digit_compat(output)
    depth = float(item.get("depth", 0.0))
    case_id = str(item.get("case_id") or f"L{int(item.get('input_len', 0))}_D{int(round(depth*100)):03d}_K{int(item.get('sample_idx', 0)):03d}")
    return {
        "valid": int(bool(pred)), "case_id": case_id,
        "input_len": int(item.get("input_len", 0)), "depth": depth,
        "depth_percent": int(round(depth * 100)), "sample_idx": int(item.get("sample_idx", 0)),
        "answer": ans, "prediction_stripped": pred,
        "passkey_em": int(pred == ans),
        "passkey_compat_prediction": compat, "passkey_compat_em": int(compat == ans and bool(compat)),
        "normalize_rule": "strip_whitespace_only_case_sensitive_punctuation_sensitive",
    }

def retain_from_profile(profile: str) -> Optional[float]:
    m = re.search(r"retain=([0-9.]+)", str(profile))
    return float(m.group(1)) if m else None

def internal_telemetry(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "retain_ratio": retain_from_profile(summary.get("compression_profile", "")),
        "compression_mode": str(summary.get("compression_mode", "")),
        "retain_budget_tokens": int(summary.get("retain_budget_tokens", 0) or 0),
        "effective_retained_tokens": int(summary.get("effective_retained_tokens", 0) or 0),
        "retained_block_count": int(summary.get("retained_block_count", 0) or 0),
        "decode_backend": str(summary.get("decode_backend", "")),
        "decode_page16_native_steps": int(summary.get("decode_page16_native_steps", 0) or 0),
        "decode_rebuild_steps": int(summary.get("decode_rebuild_steps", 0) or 0),
        "decode_materialize_kv_bytes": int(summary.get("decode_materialize_kv_bytes", 0) or 0),
        "p2_attempted_steps": int(summary.get("p2_attempted_steps", 0) or 0),
        "p2_success_steps": int(summary.get("p2_success_steps", 0) or 0),
        "p2_offload_blocks": int(summary.get("p2_ready_offload_blocks_total", summary.get("p2_ready_offload_blocks_last", 0)) or 0),
        "resident_miss_steps": max(int(summary.get("decode_page16_native_resident_miss_steps", 0) or 0), int(summary.get("decode_paged_direct_resident_miss_steps", 0) or 0)),
        "kv_logical_block_size": int(summary.get("kv_logical_block_size", 0) or 0),
        "flash_attn_enabled": int(summary.get("flash_attn_enabled", 0) or 0),
        "selected_writeback_enabled": int(summary.get("selected_writeback_enabled", 0) or 0),
        "prefill_writeback_backend": str(summary.get("prefill_writeback_backend", "")),
        "gpu_selected_writeback_steps": int(summary.get("gpu_selected_writeback_steps", 0) or 0),
        "cpu_selected_compaction_steps": int(summary.get("cpu_selected_compaction_steps", 0) or 0),
        "gpu_writeback_oom_fallbacks": int(summary.get("gpu_writeback_oom_fallbacks", 0) or 0),
        "writeback_transaction_rollbacks": int(summary.get("writeback_transaction_rollbacks", 0) or 0),
        "raw_kv_cpu_stash_bytes": int(summary.get("raw_kv_cpu_stash_bytes", 0) or 0),
        "selected_global_block_count": int(summary.get("selected_global_block_count", 0) or 0),
        "writeback_est_required_gb": float(summary.get("writeback_est_required_gb", 0.0) or 0.0),
        "writeback_free_gb": float(summary.get("writeback_free_gb", 0.0) or 0.0),
        "writeback_block_selection_shared_layers": int(summary.get("writeback_block_selection_shared_layers", 0) or 0),
        "score_full_attention_materialized": int(summary.get("score_full_attention_materialized", 0) or 0),
        "min_free_blocks": int(summary.get("min_free_blocks", -1) or -1),
    }

def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = max(1, len(rows)); completed = sum(int(r.get("completed", 0)) for r in rows); valid = sum(int(r.get("valid", 0)) for r in rows)
    em = sum(int(r.get("passkey_em", 0)) for r in rows); compat = sum(int(r.get("passkey_compat_em", 0)) for r in rows)
    by_depth: Dict[str, Dict[str, int]] = {}
    for r in rows:
        k = f"{int(r.get('depth_percent', 0)):03d}"; b = by_depth.setdefault(k, {"n":0,"em":0,"compat":0,"completed":0,"valid":0})
        b["n"] += 1; b["em"] += int(r.get("passkey_em", 0)); b["compat"] += int(r.get("passkey_compat_em", 0)); b["completed"] += int(r.get("completed", 0)); b["valid"] += int(r.get("valid", 0))
    return {
        "n_cases": len(rows), "completed": completed, "completion_rate": completed/n, "valid_rate": valid/n,
        "passkey_em": em/n, "passkey_compat_em": compat/n,
        "by_depth": {k: {"n": v["n"], "em": v["em"]/max(1,v["n"]), "compat_em": v["compat"]/max(1,v["n"]), "completion_rate": v["completed"]/max(1,v["n"])} for k,v in sorted(by_depth.items())}
    }

def run_one(model: str, method: str, cases: Sequence[Dict[str, Any]], tokenizer, concurrency: int, max_new: int, frac_map: Dict[str, float]) -> Dict[str, Any]:
    if is_hf_style_method(method):
        summary = run_hf_style_prompt_batches(model, method, list(cases), int(concurrency), int(max_new), 1, eval_passkey, tokenizer=tokenizer)
        telemetry = {"retain_ratio": None, "decode_backend": "hf_vanilla", "decode_page16_native_steps": 0, "decode_rebuild_steps": 0, "decode_materialize_kv_bytes": 0, "p2_attempted_steps": 0, "p2_success_steps": 0, "p2_offload_blocks": 0, "resident_miss_steps": 0, "kv_logical_block_size": 0, "flash_attn_enabled": 0, "selected_writeback_enabled": 0, "prefill_writeback_backend": "", "gpu_selected_writeback_steps": 0, "cpu_selected_compaction_steps": 0, "gpu_writeback_oom_fallbacks": 0, "writeback_transaction_rollbacks": 0, "raw_kv_cpu_stash_bytes": 0, "selected_global_block_count": 0, "writeback_est_required_gb": 0.0, "writeback_free_gb": 0.0, "writeback_block_selection_shared_layers": 0, "score_full_attention_materialized": 0, "min_free_blocks": -1}
    else:
        if method not in frac_map: raise ValueError(f"missing gpu_mem_frac for {method}")
        summary = run_internal_prompt_batches(model, method, float(frac_map[method]), [str(x["prompt"]) for x in cases], list(cases), int(concurrency), int(max_new), tokenizer, 1, eval_passkey)
        telemetry = internal_telemetry(summary)
        selected_writeback_active = int(telemetry.get("selected_writeback_enabled", 0) or 0) == 1
        selected_writeback_forced = str(os.environ.get("KV_MIDDLEWARE_SELECTED_WRITEBACK", "")).strip().lower() in {"1", "true", "yes", "on"}
        if (selected_writeback_active or selected_writeback_forced) and int(telemetry.get("flash_attn_enabled", 0) or 0) != 1:
            summary["status"] = "Failed/Invalid"
            summary["error_reason"] = "flash_attn_fallback_invalid"
    return {**summarize_rows(summary.get("rows", [])), **telemetry, "status": str(summary.get("status", "")), "error_reason": str(summary.get("error_reason", "")), "raw_summary": {k:v for k,v in summary.items() if k != "rows"}, "rows": list(summary.get("rows", []))}

def first_div(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if int(x) != int(y): return i
    return None if len(a) == len(b) else min(len(a), len(b))

def raw_comparisons(tokenizer, method_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_case: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for mr in method_rows:
        for rr in mr.get("rows", []) or []:
            cid = str(rr.get("case_id", ""));
            if cid: by_case.setdefault(cid, {})[str(mr.get("method", ""))] = dict(rr)
    out = []
    for cid, mm in sorted(by_case.items()):
        hf = mm.get("hf_vanilla")
        if not hf: continue
        hf_text = str(hf.get("output_text", "")); hf_ids = tokenizer(hf_text, add_special_tokens=False).input_ids
        for method, rr in sorted(mm.items()):
            if method == "hf_vanilla": continue
            text = str(rr.get("output_text", "")); ids = tokenizer(text, add_special_tokens=False).input_ids; div = first_div(hf_ids, ids)
            out.append({"case_id": cid, "method": method, "text_exact_match": int(hf_text == text), "retokenized_token_ids_exact_match": int(div is None), "divergence_step": div, "hf_prediction": str(hf.get("prediction_stripped", "")), "method_prediction": str(rr.get("prediction_stripped", "")), "note": "divergence_step uses re-tokenized generated text; first-logit top1 needs a lower-level debug runner."})
    return out

def table(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    methods = sorted({r["method"] for r in rows}); lens = sorted({int(r["input_len"]) for r in rows}); out=[]
    for m in methods:
        item={"method":m}; vals=[]; cov=0
        for L in lens:
            r = next((x for x in rows if x["method"] == m and int(x["input_len"]) == L), None)
            if not r: item[str(L)]="Missing"; continue
            if int(r.get("completed",0)) == int(r.get("n_cases",-1)) and not str(r.get("error_reason","")).strip():
                cov += 1; vals.append(float(r.get("passkey_em",0.0))); item[str(L)] = round(100*float(r.get("passkey_em",0.0)),2)
            else: item[str(L)]="Failed/OOM"
        item["Avg_common_lengths"] = round(100*mean(vals),2) if vals else "NA"; item["Coverage"] = f"{cov}/{len(lens)}"; out.append(item)
    return out

def main():
    ap = argparse.ArgumentParser(description="Passkey quality v3 runner.")
    ap.add_argument("--mode", choices=sorted(MODES), default="gate"); ap.add_argument("--model-name", default=LOCAL_MODEL_PATH)
    ap.add_argument("--methods", default=""); ap.add_argument("--input-lens", default=""); ap.add_argument("--depths", default=""); ap.add_argument("--keys-per-depth", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1); ap.add_argument("--max-new-tokens", type=int, default=32); ap.add_argument("--gpu-mem-frac-map", default=""); ap.add_argument("--seed", type=int, default=DEFAULT_SEED); ap.add_argument("--out", required=True)
    args = ap.parse_args(); d_lens, d_depths, d_keys, d_methods = MODES[args.mode]
    lens = csv_ints(args.input_lens or d_lens); depths = csv_floats(args.depths or d_depths); keys = int(args.keys_per_depth or d_keys); methods = parse_methods(args.methods or d_methods, allow_vllm=False); frac_map = parse_method_frac_map(args.gpu_mem_frac_map, defaults=DEFAULT_GPU_MEM_FRAC_MAP)
    tok = AutoTokenizer.from_pretrained(str(args.model_name), trust_remote_code=True)
    rows=[]
    for L in lens:
        cases = build_cases(tok, L, depths, keys, args.seed)
        for c in cases: c["input_len"] = int(L); c["case_id"] = f"L{L}_D{int(round(float(c['depth'])*100)):03d}_K{int(c['sample_idx']):03d}"
        print(f"[passkey] input_len={L} cases={len(cases)} methods={methods}", flush=True)
        for m in methods:
            t0=time.perf_counter(); print(f"[passkey] running method={m} input_len={L}", flush=True)
            r = run_one(str(args.model_name), m, cases, tok, args.concurrency, args.max_new_tokens, frac_map); r.update({"method":m,"input_len":L,"concurrency":args.concurrency,"max_new_tokens":args.max_new_tokens,"gpu_mem_frac":float(frac_map.get(m,0.0)),"wall_clock_s":round(time.perf_counter()-t0,3),"actual_prompt_tokens_mean":mean(int(x.get("actual_prompt_tokens",0)) for x in cases)})
            rows.append(r); print(json.dumps({k:r.get(k) for k in ["method","input_len","status","completed","n_cases","passkey_em","p2_attempted_steps","p2_success_steps","resident_miss_steps","decode_page16_native_steps","decode_rebuild_steps","error_reason"]}, ensure_ascii=False), flush=True); cleanup_cuda()
    payload={"meta":{"task":"passkey_quality_v3","mode":args.mode,"model_name":args.model_name,"methods":methods,"input_lens":lens,"depths":depths,"keys_per_depth":keys,"concurrency":args.concurrency,"max_new_tokens":args.max_new_tokens,"seed":args.seed,"normalize_rule":"strip_whitespace_only_case_sensitive_punctuation_sensitive","gpu_mem_frac_map":{str(k):float(v) for k,v in frac_map.items()}},"rows":rows,"table":table(rows),"raw_sanity_comparisons":raw_comparisons(tok, rows) if "hf_vanilla" in methods else []}
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); print(f"Saved: {out}")
if __name__ == "__main__": main()
