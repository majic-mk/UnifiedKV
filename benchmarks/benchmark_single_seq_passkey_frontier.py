import gc
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from engine import ManagedInferenceEngine

import benchmarks.benchmark_niah_single_limit as niah
import benchmarks.benchmark_exp1_capacity_table as exp1

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TS = time.strftime("%Y%m%d_%H%M%S")
RESULT_JSON = RESULTS_DIR / f"single_seq_passkey_frontier_qwen_{TS}.json"
PROGRESS_JSONL = RESULTS_DIR / f"single_seq_passkey_frontier_qwen_{TS}.progress.jsonl"

MODEL_NAME = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
TRACKS = [
    ("off", "raw", "off_raw"),
    ("off", "compress", "off_compress"),
    ("p2_only", "compress", "p2_only_mainline"),
]
COARSE_MEM_FRACS = [0.92, 0.84, 0.76, 0.68, 0.60, 0.52, 0.44]
SMOKE_LEN = 16000
SMOKE_DEPTH = 0.50
DEPTHS = [0.10, 0.50, 0.90]
MAX_NEW_TOKENS = 32
MIN_TOKENS = 2000
MAX_TOKENS = 256000
PRECISION = 1000
ANCHOR1 = 128000
ANCHOR2 = 256000
SEED = 20260406
CASE_TIMEOUT_S = 300


def round2(x: float) -> float:
    return round(float(x) + 1e-9, 2)


def emit_progress(obj: Dict) -> None:
    with PROGRESS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def cleanup_engine(engine) -> None:
    if engine is None:
        return
    try:
        if hasattr(engine, "has_pending_requests") and engine.has_pending_requests():
            req_map = dict(getattr(engine, "_requests", {}) or {})
            for _, req in req_map.items():
                try:
                    engine._mark_request_failed(req, RuntimeError("single_seq_cleanup"))
                except Exception:
                    pass
    except Exception:
        pass
    try:
        if hasattr(engine, "_reset_online_runtime"):
            engine._reset_online_runtime(clear_request_counter=False)
    except Exception:
        pass
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def patch_niah_config() -> None:
    common = dict(exp1.COMMON_BASE)
    common["model_name"] = MODEL_NAME
    common["max_new_tokens"] = MAX_NEW_TOKENS
    common["prefill_batch_size"] = 1
    common["decode_micro_batch_size"] = 0
    niah.LOCAL_MODEL_PATH = MODEL_NAME
    niah.COMMON_BASE = common
    niah.TARGET_EM_THRESHOLD = 1.0
    niah.TARGET_VALID_THRESHOLD = 1.0
    niah.GPU_MEM_FRAC_FALLBACK_TRIES = 1
    niah.GPU_MEM_FRAC_FALLBACK_STEP = 0.0
    niah.MODE_OFF_ARGS = {"p2_enabled": False}
    p2_args = dict(exp1.P2_ONLY_COMPRESS_ARGS)
    p2_args.pop("retain_ratio", None)
    niah.MODE_P2_ONLY_ARGS = p2_args
    niah.PREFILL_RAW_ARGS = {"retain_ratio": 1.0}
    niah.PREFILL_COMPRESS_ARGS = {"retain_ratio": float(exp1.OFF_COMPRESS_ARGS["retain_ratio"])}


patch_niah_config()


def run_single_case(engine: ManagedInferenceEngine, mode: str, prefill_track: str, target_len: int, depth: float, seed: int, stage: str, case_timeout_s: Optional[int] = None) -> Dict:
    import signal as _signal

    rng = niah.random.Random((seed * 1000003) ^ (target_len * 1009) ^ int(depth * 1000))
    answer = f"{rng.randint(100000, 999999)}"
    prompt, actual_len = niah._make_prompt_for_target_tokens(engine.tokenizer, target_len, depth, answer)
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "prefill_track": prefill_track,
        "stage": stage,
        "target_len": int(target_len),
        "actual_prompt_tokens": int(actual_len),
        "depth": float(depth),
        "success": 0,
        "passkey_first_em": 0.0,
        "valid_rate": 0.0,
        "em_given_valid": 0.0,
        "tokens_s": 0.0,
        "p95_ms": 0.0,
        "decode_min_n_free": 0,
        "decode_path_selected": "",
        "decode_path_fallback_count": 0,
        "decode_path_fallback_reason_topk": {},
        "p2_attempted_steps": 0,
        "p2_success_steps": 0,
        "p2_ready_candidate_steps": 0,
        "p2_decode_candidate_steps": 0,
        "ensure_fail": 0,
        "prefetch_fail": 0,
        "error": "",
        "wall_ms": 0.0,
    }
    t0 = time.perf_counter()

    def _cleanup_pending(reason: str) -> None:
        try:
            if hasattr(niah, "_cleanup_engine_pending"):
                niah._cleanup_engine_pending(engine, reason)
                return
        except Exception:
            pass
        try:
            if hasattr(engine, "has_pending_requests") and engine.has_pending_requests():
                req_map = dict(getattr(engine, "_requests", {}) or {})
                for _, req in req_map.items():
                    try:
                        engine._mark_request_failed(req, RuntimeError(reason or "single_case_exception"))
                    except Exception:
                        pass
        except Exception:
            pass

    try:
        def _do_generate():
            return engine.generate([prompt], return_metrics=True, return_details=True)

        timeout_s = int(case_timeout_s or 0)
        if timeout_s > 0 and hasattr(_signal, "SIGALRM"):
            old_handler = _signal.getsignal(_signal.SIGALRM)

            def _timeout_handler(signum, frame):
                raise TimeoutError(f"single_case_timeout>{timeout_s}s")

            _signal.signal(_signal.SIGALRM, _timeout_handler)
            _signal.alarm(timeout_s)
            try:
                outputs, metrics, _ = _do_generate()
            finally:
                _signal.alarm(0)
                _signal.signal(_signal.SIGALRM, old_handler)
        else:
            outputs, metrics, _ = _do_generate()

        wall_ms = (time.perf_counter() - t0) * 1000.0
        output_text = outputs[0] if outputs and isinstance(outputs[0], str) else ""
        quality = niah._eval_passkey_single(output_text, answer)
        delta = dict(metrics.get("offloader_delta", {}) or {})
        gen_ok = bool(str(output_text).strip())
        timeout_like = timeout_s > 0 and wall_ms >= (timeout_s * 1000.0 - 50.0)
        err_msg = ""
        if not gen_ok:
            err_msg = "empty_output"
            if timeout_like:
                err_msg = f"single_case_timeout>{timeout_s}s_or_stalled"
            _cleanup_pending(err_msg)
        rec.update({
            "success": 1 if gen_ok else 0,
            "passkey_first_em": float(quality["passkey_first_em"]),
            "valid_rate": float(quality["valid_rate"]),
            "em_given_valid": float(quality["em_given_valid"]),
            "tokens_s": float(metrics.get("tokens_per_sec", 0.0)),
            "p95_ms": float(metrics.get("decode_step_p95_ms", 0.0)),
            "decode_min_n_free": int(metrics.get("decode_min_n_free", 0)),
            "decode_path_selected": str(metrics.get("decode_path_selected", "")),
            "decode_path_fallback_count": int(metrics.get("decode_path_fallback_count", 0)),
            "decode_path_fallback_reason_topk": dict(metrics.get("decode_path_fallback_reason_topk", {}) or {}),
            "p2_attempted_steps": int(metrics.get("p2_attempted_steps", 0)),
            "p2_success_steps": int(metrics.get("p2_success_steps", 0)),
            "p2_ready_candidate_steps": int(metrics.get("p2_ready_candidate_steps", 0)),
            "p2_decode_candidate_steps": int(metrics.get("p2_decode_candidate_steps", 0)),
            "ensure_fail": int(delta.get("ensure_fail", 0)),
            "prefetch_fail": int(delta.get("prefetch_fail", 0)),
            "error": err_msg,
            "wall_ms": float(round(wall_ms, 3)),
        })
    except Exception as exc:
        rec["error"] = str(exc)
        rec["wall_ms"] = float(round((time.perf_counter() - t0) * 1000.0, 3))
        _cleanup_pending(rec["error"] or "single_case_exception")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rec


niah._run_single_case = run_single_case


def smoke_result(mode: str, prefill_track: str, label: str, mem_frac: float) -> Dict:
    engine = None
    try:
        engine = niah._build_engine(mem_frac, mode, prefill_track)
        rec = run_single_case(engine, mode, prefill_track, SMOKE_LEN, SMOKE_DEPTH, SEED, "memfrac_smoke")
        rec["track_label"] = label
        rec["candidate_mem_frac"] = float(mem_frac)
        emit_progress({"kind": "smoke", **rec})
        return rec
    finally:
        cleanup_engine(engine)


def summarize_track_result(label: str, candidate_mem_frac: float, track_res: Dict) -> Dict:
    lq = int(track_res.get("l_quality_max", 0))
    ls = int(track_res.get("l_survival_max", 0))
    recs = [r for r in track_res.get("records", []) if int(r.get("target_len", -1)) == lq and str(r.get("stage", "")).startswith("quality")]
    if not recs:
        return {
            "track_label": label,
            "candidate_mem_frac": float(candidate_mem_frac),
            "l_survival_max": ls,
            "l_quality_max": lq,
            "tokens_s_at_l_quality": 0.0,
            "p95_ms_at_l_quality": 0.0,
            "ensure_fail": 0,
            "prefetch_fail": 0,
            "p2_attempted_steps": 0,
            "p2_success_steps": 0,
            "p2_ready_candidate_steps": 0,
            "p2_decode_candidate_steps": 0,
            "decode_path_selected": "",
            "decode_path_fallback_count": 0,
            "decode_path_fallback_reason_topk": {},
            "track_result": track_res,
        }
    path_counts = {}
    fallback_reasons = {}
    for r in recs:
        p = str(r.get("decode_path_selected", "") or "")
        if p:
            path_counts[p] = path_counts.get(p, 0) + 1
        for k, v in dict(r.get("decode_path_fallback_reason_topk", {}) or {}).items():
            fallback_reasons[str(k)] = fallback_reasons.get(str(k), 0) + int(v)
    decode_path_selected = ""
    if path_counts:
        decode_path_selected = sorted(path_counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[0][0]
    topk = {k: int(v) for k, v in sorted(fallback_reasons.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:3]}
    return {
        "track_label": label,
        "candidate_mem_frac": float(candidate_mem_frac),
        "l_survival_max": ls,
        "l_quality_max": lq,
        "tokens_s_at_l_quality": float(mean(float(r.get("tokens_s", 0.0)) for r in recs)),
        "p95_ms_at_l_quality": float(mean(float(r.get("p95_ms", 0.0)) for r in recs)),
        "ensure_fail": int(sum(int(r.get("ensure_fail", 0)) for r in recs)),
        "prefetch_fail": int(sum(int(r.get("prefetch_fail", 0)) for r in recs)),
        "p2_attempted_steps": int(sum(int(r.get("p2_attempted_steps", 0)) for r in recs)),
        "p2_success_steps": int(sum(int(r.get("p2_success_steps", 0)) for r in recs)),
        "p2_ready_candidate_steps": int(sum(int(r.get("p2_ready_candidate_steps", 0)) for r in recs)),
        "p2_decode_candidate_steps": int(sum(int(r.get("p2_decode_candidate_steps", 0)) for r in recs)),
        "decode_path_selected": decode_path_selected,
        "decode_path_fallback_count": int(sum(int(r.get("decode_path_fallback_count", 0)) for r in recs)),
        "decode_path_fallback_reason_topk": topk,
        "track_result": track_res,
    }


def ranking_key(summary: Dict) -> Tuple[int, int, float, float]:
    return (
        int(summary.get("l_quality_max", 0)),
        int(summary.get("l_survival_max", 0)),
        float(summary.get("tokens_s_at_l_quality", 0.0)),
        -float(summary.get("p95_ms_at_l_quality", 0.0)),
    )


def run_track(mode: str, prefill_track: str, label: str) -> Dict:
    smoke_records = []
    tested = {}
    for frac in COARSE_MEM_FRACS:
        rec = smoke_result(mode, prefill_track, label, frac)
        smoke_records.append(rec)
        tested[round2(frac)] = rec

    success_recs = [r for r in smoke_records if int(r.get("success", 0)) == 1]
    if not success_recs:
        return {
            "track_label": label,
            "status": "no_smoke_success",
            "coarse_smoke": smoke_records,
            "seed_candidates": [],
            "local_smoke": [],
            "shortlist": [],
            "selected": None,
            "candidates": [],
        }

    def smoke_perf_key(rec: Dict) -> Tuple[float, float, float]:
        return (
            float(rec.get("tokens_s", 0.0)),
            -float(rec.get("p95_ms", 0.0)),
            float(rec.get("candidate_mem_frac", 0.0)),
        )

    highest_success = sorted(success_recs, key=lambda r: float(r.get("candidate_mem_frac", 0.0)), reverse=True)[0]
    best_smoke = sorted(success_recs, key=smoke_perf_key, reverse=True)[0]

    seed_candidates = []
    for rec in [highest_success, best_smoke]:
        cand = round2(rec["candidate_mem_frac"])
        if cand not in seed_candidates:
            seed_candidates.append(cand)

    local_candidates = []
    for seed in seed_candidates:
        offsets = [0.04, 0.0, -0.04]
        if seed == round2(highest_success["candidate_mem_frac"]):
            offsets.append(-0.08)
        for off in offsets:
            cand = round2(seed + off)
            if cand < 0.10 or cand > 0.92:
                continue
            if cand not in local_candidates:
                local_candidates.append(cand)

    local_smoke = []
    for cand in local_candidates:
        if cand in tested:
            local_smoke.append(tested[cand])
            continue
        rec = smoke_result(mode, prefill_track, label, cand)
        tested[cand] = rec
        local_smoke.append(rec)

    tested_success = [rec for _, rec in sorted(tested.items()) if int(rec.get("success", 0)) == 1]
    shortlist = []

    def _add_shortlist(cand: float) -> None:
        cand = round2(cand)
        if cand in tested and int(tested[cand].get("success", 0)) == 1 and cand not in shortlist:
            shortlist.append(cand)

    for rec in sorted(tested_success, key=smoke_perf_key, reverse=True):
        _add_shortlist(rec["candidate_mem_frac"])
        if len(shortlist) >= 2:
            break
    _add_shortlist(highest_success["candidate_mem_frac"])
    if len(shortlist) < 3:
        for rec in sorted(tested_success, key=smoke_perf_key, reverse=True):
            _add_shortlist(rec["candidate_mem_frac"])
            if len(shortlist) >= 3:
                break
    if len(shortlist) < 3:
        for rec in sorted(tested_success, key=lambda r: float(r.get("candidate_mem_frac", 0.0)), reverse=True):
            _add_shortlist(rec["candidate_mem_frac"])
            if len(shortlist) >= 3:
                break
    shortlist = shortlist[:3]

    candidate_summaries = []
    for cand in shortlist:
        def _add(rec):
            emit_progress({"kind": "frontier_case", "track_label": label, "candidate_mem_frac": cand, **rec})

        track_res = niah._run_track(
            mode,
            prefill_track,
            SEED,
            float(cand),
            [SMOKE_LEN],
            ANCHOR1,
            ANCHOR2,
            MIN_TOKENS,
            MAX_TOKENS,
            PRECISION,
            _add,
            CASE_TIMEOUT_S,
            0.0,
        )
        summary = summarize_track_result(label, cand, track_res)
        candidate_summaries.append(summary)
        emit_progress({
            "kind": "candidate_done",
            "track_label": label,
            "candidate_mem_frac": cand,
            "l_survival_max": summary["l_survival_max"],
            "l_quality_max": summary["l_quality_max"],
            "tokens_s_at_l_quality": summary["tokens_s_at_l_quality"],
            "p95_ms_at_l_quality": summary["p95_ms_at_l_quality"],
        })

    selected = None
    if candidate_summaries:
        selected = sorted(candidate_summaries, key=ranking_key, reverse=True)[0]
    return {
        "track_label": label,
        "status": "ok",
        "coarse_smoke": smoke_records,
        "seed_candidates": seed_candidates,
        "local_smoke": local_smoke,
        "shortlist": shortlist,
        "selected": selected,
        "candidates": candidate_summaries,
    }



def main() -> None:
    payload = {
        "meta": {
            "timestamp": TS,
            "model_name": MODEL_NAME,
            "task": "single_sequence_longest_passkey_frontier",
            "selection_policy": [
                "maximize L_quality_max",
                "then maximize L_survival_max",
                "then maximize tokens_s_at_L_quality",
                "then minimize p95_ms_at_L_quality",
            ],
            "strict_gate": {
                "passkey_first_em": 1.0,
                "valid_rate": 1.0,
                "depths": DEPTHS,
            },
            "coarse_mem_fracs": COARSE_MEM_FRACS,
            "smoke_length": SMOKE_LEN,
            "anchors": [ANCHOR1, ANCHOR2],
            "max_new_tokens": MAX_NEW_TOKENS,
            "precision_tokens": PRECISION,
            "tracks": [t[2] for t in TRACKS],
        },
        "tracks": [],
    }
    RESULT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode, prefill_track, label in TRACKS:
        print(f"===== TRACK {label} START =====", flush=True)
        res = run_track(mode, prefill_track, label)
        payload["tracks"].append(res)
        RESULT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        sel = res.get("selected")
        emit_progress({
            "kind": "track_done",
            "track_label": label,
            "selected_gpu_mem_frac": None if not sel else sel.get("candidate_mem_frac"),
            "l_survival_max": None if not sel else sel.get("l_survival_max"),
            "l_quality_max": None if not sel else sel.get("l_quality_max"),
        })
        print(f"===== TRACK {label} DONE =====", flush=True)
    print("RESULT_JSON=" + str(RESULT_JSON), flush=True)
    print("PROGRESS_JSONL=" + str(PROGRESS_JSONL), flush=True)


if __name__ == "__main__":
    main()
