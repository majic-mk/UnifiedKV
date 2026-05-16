import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from benchmark_internal_common import (
    is_hf_style_method,
    parse_method_frac_map,
    parse_methods,
    run_internal_prompt_batches,
)
from hf_style_common import run_hf_style_prompt_batches
from benchmark_vllm_common import (
    DEFAULT_REQUEST_TIMEOUT_S,
    DEFAULT_SERVER_HOST,
    VLLM_SERVED_MODEL_NAME,
    aggregate_streaming_metrics,
    load_tokenizer,
    run_prompt_batch,
    start_vllm_server,
    stop_vllm_server,
)


SEARCH_ROOTS = [
    Path.cwd(),
    Path.cwd().parent,
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent.parent,
    Path("/root/autodl-tmp"),
    Path("/root/autodl-tmp/datasets"),
    Path("/root/autodl-tmp/data"),
]


def parse_csv_ints(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty integer list")
    return vals


def resolve_dataset_path(dataset: str) -> Path:
    candidate = Path(str(dataset))
    if candidate.exists():
        return candidate.resolve()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        direct = root / str(dataset)
        if direct.exists():
            return direct.resolve()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        hits = list(root.rglob(str(dataset)))
        if hits:
            return hits[0].resolve()
    raise FileNotFoundError(f"dataset not found: {dataset}")


def load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [dict(x) for x in payload]
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return [dict(x) for x in payload["data"]]
        if isinstance(payload.get("records"), list):
            return [dict(x) for x in payload["records"]]
    raise ValueError(f"unsupported dataset layout: {path}")


def normalize_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = None
    if isinstance(record.get("messages"), list):
        raw = record.get("messages")
    elif isinstance(record.get("conversations"), list):
        raw = record.get("conversations")
    else:
        raw = []
    out: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", item.get("from", ""))).strip().lower()
        content = item.get("content", item.get("value", ""))
        text = str(content or "").strip()
        if not text:
            continue
        if role in {"human", "user"}:
            mapped = "user"
        elif role in {"gpt", "assistant", "bot"}:
            mapped = "assistant"
        elif role == "system":
            mapped = "system"
        else:
            continue
        out.append({"role": mapped, "content": text})
    return out


def render_chat_prompt(tokenizer, messages: Sequence[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            pass
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\n\nassistant:"


def count_tokens(tokenizer, text: str) -> int:
    return int(len(tokenizer(str(text), add_special_tokens=False).input_ids))



def summarize_int_distribution(values: Sequence[int]) -> Dict[str, Any]:
    vals = sorted(int(x) for x in values)
    if not vals:
        return {
            "count": 0,
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "max": 0,
            "mean": 0.0,
        }

    def percentile(q: float) -> int:
        if len(vals) == 1:
            return int(vals[0])
        idx = int(round((len(vals) - 1) * float(q)))
        idx = max(0, min(len(vals) - 1, idx))
        return int(vals[idx])

    return {
        "count": int(len(vals)),
        "min": int(vals[0]),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": int(vals[-1]),
        "mean": float(sum(vals) / max(1, len(vals))),
    }


def scan_sharegpt_candidates(
    tokenizer,
    records: Sequence[Dict[str, Any]],
    prompt_len_min: int,
    prompt_len_max: int,
    target_len_min: int,
    target_len_max: int,
    progress_prefix: str = "",
    progress_every: int = 5000,
) -> List[Dict[str, Any]]:
    prompt_char_min = max(1, int(prompt_len_min) * 2)
    prompt_char_max = max(int(prompt_len_max), int(prompt_len_max) * 6)
    target_char_min = max(32, int(target_len_min) // 2)
    target_char_max = max(int(target_len_max), int(target_len_max) * 8)
    candidates: List[Dict[str, Any]] = []
    for rec_idx, record in enumerate(records):
        messages = normalize_messages(record)
        if not messages:
            continue
        for turn_idx, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            prompt_messages = messages[:turn_idx]
            if not prompt_messages:
                continue
            answer_text = str(msg.get("content", ""))
            answer_chars = len(answer_text)
            if answer_chars < target_char_min or answer_chars > target_char_max:
                continue
            prompt_chars = sum(len(str(m.get("content", ""))) for m in prompt_messages)
            if prompt_chars < prompt_char_min or prompt_chars > prompt_char_max:
                continue
            prompt = render_chat_prompt(tokenizer, prompt_messages)
            prompt_tokens = count_tokens(tokenizer, prompt)
            target_tokens = count_tokens(tokenizer, answer_text)
            if prompt_tokens < int(prompt_len_min) or prompt_tokens > int(prompt_len_max):
                continue
            if target_tokens < int(target_len_min) or target_tokens > int(target_len_max):
                continue
            candidates.append(
                {
                    "record_idx": int(rec_idx),
                    "turn_idx": int(turn_idx),
                    "prompt": prompt,
                    "prompt_tokens": int(prompt_tokens),
                    "target_tokens": int(target_tokens),
                }
            )
        scanned = rec_idx + 1
        if progress_prefix and progress_every and scanned % int(progress_every) == 0:
            print(
                f"{progress_prefix}: scan_progress scanned_records={scanned} candidates={len(candidates)}",
                flush=True,
            )
    return candidates


def select_sharegpt_samples(
    candidates: Sequence[Dict[str, Any]],
    sample_count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    pool = [dict(x) for x in candidates]
    rng = random.Random(int(seed))
    if len(pool) <= int(sample_count):
        rng.shuffle(pool)
        return pool
    return [dict(x) for x in rng.sample(pool, int(sample_count))]


def build_sharegpt_samples(
    tokenizer,
    records: Sequence[Dict[str, Any]],
    sample_count: int,
    prompt_len_min: int,
    prompt_len_max: int,
    target_len_min: int,
    target_len_max: int,
    seed: int,
) -> List[Dict[str, Any]]:
    candidates = scan_sharegpt_candidates(
        tokenizer=tokenizer,
        records=records,
        prompt_len_min=int(prompt_len_min),
        prompt_len_max=int(prompt_len_max),
        target_len_min=int(target_len_min),
        target_len_max=int(target_len_max),
    )
    if not candidates:
        raise RuntimeError("no ShareGPT samples matched the requested prompt/target length filters")
    return select_sharegpt_samples(candidates, sample_count=int(sample_count), seed=int(seed))


def valid_sharegpt_output(text: str) -> bool:
    return bool(str(text or "").strip())


def eval_sharegpt_output(item: Dict[str, Any], output_text: str) -> Dict[str, Any]:
    return {
        "valid": int(valid_sharegpt_output(output_text)),
        "record_idx": int(item["record_idx"]),
        "turn_idx": int(item["turn_idx"]),
        "prompt_tokens": int(item["prompt_tokens"]),
        "target_tokens": int(item["target_tokens"]),
    }


def iter_batches(items: Sequence[Dict[str, Any]], concurrency: int) -> Iterable[Sequence[Dict[str, Any]]]:
    size = max(1, int(concurrency))
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def run_vllm_method(
    model_name: str,
    samples: Sequence[Dict[str, Any]],
    concurrency: int,
    max_new_tokens: int,
    repeats: int,
    gpu_memory_utilization: float,
    request_timeout_s: float,
    server_log: str,
    base_url: str,
) -> Dict[str, Any]:
    tokenizer = load_tokenizer(model_name)
    endpoint = str(base_url).strip()
    server_proc = None
    launched_server = False
    if not endpoint:
        max_model_len = max(int(x["prompt_tokens"]) for x in samples) + int(max_new_tokens) + 512
        server_proc, port = start_vllm_server(
            model_name=model_name,
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=max_model_len,
            host=DEFAULT_SERVER_HOST,
            log_path=Path(server_log) if str(server_log).strip() else None,
        )
        endpoint = f"http://{DEFAULT_SERVER_HOST}:{int(port)}"
        launched_server = True

    request_rows: List[Dict[str, Any]] = []
    batch_wall_ms_list: List[float] = []
    try:
        for repeat_idx in range(int(repeats)):
            for batch in iter_batches(samples, int(concurrency)):
                prompts = [str(item["prompt"]) for item in batch]
                t0 = time.perf_counter()
                outputs = run_prompt_batch(
                    base_url=endpoint,
                    prompts=prompts,
                    max_new_tokens=int(max_new_tokens),
                    tokenizer=tokenizer,
                    request_timeout_s=float(request_timeout_s),
                )
                batch_wall_ms_list.append((time.perf_counter() - t0) * 1000.0)
                for item, rec in zip(batch, outputs):
                    request_rows.append(
                        {
                            **rec,
                            "repeat_idx": int(repeat_idx),
                            "record_idx": int(item["record_idx"]),
                            "turn_idx": int(item["turn_idx"]),
                            "prompt_tokens": int(item["prompt_tokens"]),
                            "target_tokens": int(item["target_tokens"]),
                        }
                    )
    finally:
        if launched_server:
            stop_vllm_server(server_proc)

    agg = aggregate_streaming_metrics(
        request_rows=request_rows,
        requested_repeats=int(repeats),
        valid_fn=valid_sharegpt_output,
        batch_wall_ms_list=batch_wall_ms_list,
    )
    return {
        "method": "vllm",
        "served_model_name": VLLM_SERVED_MODEL_NAME,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "completion_rate": agg["completion_rate"],
        "valid_completion_rate": agg["valid_completion_rate"],
        "oom_failure_count": agg["oom_failure_count"],
        "status": agg["status"],
        "frontier_reason": agg["frontier_reason"],
        "tokens_per_sec": agg["tokens_per_sec"],
        "ttft_p95_ms": agg["ttft_p95_ms"],
        "ttft_p99_ms": agg["ttft_p99_ms"],
        "itl_p95_ms": agg["itl_p95_ms"],
        "itl_p99_ms": agg["itl_p99_ms"],
        "min_free_blocks": -1,
        "min_free_block_ratio": -1.0,
        "wall_clock_total_runtime_ms": agg["wall_clock_total_runtime_ms"],
        "error_reason": agg["error_reason"],
        "actual_prompt_tokens_mean": float(sum(int(x["prompt_tokens"]) for x in samples) / max(1, len(samples))),
        "rows": request_rows,
        "meta_note": "vLLM server path does not expose KV block pool metrics in this runner, so min_free_blocks and min_free_block_ratio are set to -1.",
    }


def run_internal_method(
    model_name: str,
    method: str,
    samples: Sequence[Dict[str, Any]],
    concurrency: int,
    max_new_tokens: int,
    repeats: int,
    gpu_mem_frac: float,
):
    tokenizer = load_tokenizer(model_name)
    summary = run_internal_prompt_batches(
        model_name=model_name,
        method=method,
        gpu_mem_frac=float(gpu_mem_frac),
        prompts=[str(item["prompt"]) for item in samples],
        items=list(samples),
        concurrency=int(concurrency),
        max_new_tokens=int(max_new_tokens),
        tokenizer=tokenizer,
        repeats=int(repeats),
        evaluate_output_fn=eval_sharegpt_output,
    )
    row = {
        "method": str(method),
        "gpu_mem_frac": float(gpu_mem_frac),
        "completion_rate": summary["completion_rate"],
        "valid_completion_rate": summary["valid_completion_rate"],
        "oom_failure_count": summary["oom_failure_count"],
        "status": summary["status"],
        "frontier_reason": summary["frontier_reason"],
        "tokens_per_sec": summary["tokens_per_sec"],
        "ttft_p95_ms": summary["ttft_p95_ms"],
        "ttft_p99_ms": summary["ttft_p99_ms"],
        "itl_p95_ms": summary["itl_p95_ms"],
        "itl_p99_ms": summary["itl_p99_ms"],
        "min_free_blocks": summary["min_free_blocks"],
        "min_free_block_ratio": summary["min_free_block_ratio"],
        "wall_clock_total_runtime_ms": summary["wall_clock_total_runtime_ms"],
        "error_reason": summary["error_reason"],
        "actual_prompt_tokens_mean": float(sum(int(x["prompt_tokens"]) for x in samples) / max(1, len(samples))),
        "compression_profile": summary["compression_profile"],
        "rows": summary["rows"],
    }
    for key, value in summary.items():
        if key not in row and key != "rows":
            row[key] = value
    return row


def run_hf_style_method(
    model_name: str,
    method: str,
    samples: Sequence[Dict[str, Any]],
    concurrency: int,
    max_new_tokens: int,
    repeats: int,
):
    tokenizer = load_tokenizer(model_name)
    summary = run_hf_style_prompt_batches(
        model_name=model_name,
        method=method,
        items=list(samples),
        concurrency=int(concurrency),
        max_new_tokens=int(max_new_tokens),
        repeats=int(repeats),
        evaluate_output_fn=eval_sharegpt_output,
        tokenizer=tokenizer,
    )
    row = {
        "method": str(method),
        "completion_rate": summary["completion_rate"],
        "valid_completion_rate": summary["valid_completion_rate"],
        "oom_failure_count": summary["oom_failure_count"],
        "status": summary["status"],
        "frontier_reason": summary["frontier_reason"],
        "tokens_per_sec": summary["tokens_per_sec"],
        "ttft_p95_ms": summary["ttft_p95_ms"],
        "ttft_p99_ms": summary["ttft_p99_ms"],
        "itl_p95_ms": summary["itl_p95_ms"],
        "itl_p99_ms": summary["itl_p99_ms"],
        "min_free_blocks": summary["min_free_blocks"],
        "min_free_block_ratio": summary["min_free_block_ratio"],
        "wall_clock_total_runtime_ms": summary["wall_clock_total_runtime_ms"],
        "error_reason": summary["error_reason"],
        "actual_prompt_tokens_mean": float(sum(int(x["prompt_tokens"]) for x in samples) / max(1, len(samples))),
        "compression_profile": summary["compression_profile"],
        "stack_type": summary["stack_type"],
        "kv_backend": summary["kv_backend"],
        "fallback_policy": summary["fallback_policy"],
        "rows": summary["rows"],
    }
    for key, value in summary.items():
        if key not in row and key != "rows":
            row[key] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="ShareGPT serving benchmark for vLLM and internal methods.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--methods", type=str, default="vllm,off_compress,p2_only_compress")
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--concurrency-list", type=str, default="16,32,64")
    parser.add_argument("--prompt-len-min", type=int, default=4096)
    parser.add_argument("--prompt-len-max", type=int, default=32768)
    parser.add_argument("--target-len-min", type=int, default=64)
    parser.add_argument("--target-len-max", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--gpu-mem-frac-map", type=str, default="")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--server-log", type=str, default="")
    parser.add_argument("--samples-json", type=str, default="")
    parser.add_argument("--out", type=str, default="benchmark_sharegpt_serving.json")
    args = parser.parse_args()

    methods = parse_methods(args.methods, allow_vllm=True)
    concurrencies = parse_csv_ints(args.concurrency_list)
    tokenizer = load_tokenizer(args.model_name)
    dataset_path = resolve_dataset_path(args.dataset)
    sample_source = "scan"
    if str(args.samples_json).strip():
        sample_path = Path(str(args.samples_json))
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            samples = [dict(x) for x in payload.get("samples", [])]
        else:
            samples = [dict(x) for x in payload]
        if not samples:
            raise RuntimeError(f"empty samples_json: {sample_path}")
        sample_source = str(sample_path)
    else:
        records = load_records(dataset_path)
        samples = build_sharegpt_samples(
            tokenizer=tokenizer,
            records=records,
            sample_count=int(args.sample_count),
            prompt_len_min=int(args.prompt_len_min),
            prompt_len_max=int(args.prompt_len_max),
            target_len_min=int(args.target_len_min),
            target_len_max=int(args.target_len_max),
            seed=int(args.seed),
        )
    frac_map = parse_method_frac_map(args.gpu_mem_frac_map)

    rows: List[Dict[str, Any]] = []
    for method in methods:
        for concurrency in concurrencies:
            if method == "vllm":
                row = run_vllm_method(
                    model_name=args.model_name,
                    samples=samples,
                    concurrency=int(concurrency),
                    max_new_tokens=int(args.max_new_tokens),
                    repeats=int(args.repeats),
                    gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
                    request_timeout_s=float(args.request_timeout_s),
                    server_log=str(args.server_log),
                    base_url=str(args.base_url),
                )
            elif is_hf_style_method(method):
                row = run_hf_style_method(
                    model_name=args.model_name,
                    method=method,
                    samples=samples,
                    concurrency=int(concurrency),
                    max_new_tokens=int(args.max_new_tokens),
                    repeats=int(args.repeats),
                )
            else:
                if method not in frac_map:
                    raise ValueError(f"missing gpu mem frac for method '{method}'")
                row = run_internal_method(
                    model_name=args.model_name,
                    method=method,
                    samples=samples,
                    concurrency=int(concurrency),
                    max_new_tokens=int(args.max_new_tokens),
                    repeats=int(args.repeats),
                    gpu_mem_frac=float(frac_map[method]),
                )
            row["concurrency"] = int(concurrency)
            rows.append(row)
            print(
                json.dumps(
                    {
                        "method": row["method"],
                        "concurrency": row["concurrency"],
                        "status": row["status"],
                        "completion_rate": row["completion_rate"],
                        "valid_completion_rate": row["valid_completion_rate"],
                        "oom_failure_count": row["oom_failure_count"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    payload = {
        "meta": {
            "task": "sharegpt_serving",
            "model_name": str(args.model_name),
            "dataset": str(dataset_path),
            "methods": list(methods),
            "sample_count": int(len(samples)),
            "requested_sample_count": int(args.sample_count),
            "concurrency_list": [int(x) for x in concurrencies],
            "prompt_len_range": [int(args.prompt_len_min), int(args.prompt_len_max)],
            "target_len_range": [int(args.target_len_min), int(args.target_len_max)],
            "max_new_tokens": int(args.max_new_tokens),
            "repeats": int(args.repeats),
            "seed": int(args.seed),
            "sample_source": sample_source,
            "valid_definition": "request completed normally and output format is legal",
        },
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
