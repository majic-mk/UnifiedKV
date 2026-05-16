import argparse
import json
import random
import re
import string
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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

TASK_METRIC = {
    "passage_retrieval_en": "exact_match",
    "multifieldqa_en": "token_f1",
    "hotpotqa": "token_f1",
    "2wikimqa": "token_f1",
    "musique": "token_f1",
}


def parse_csv_ints(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty integer list")
    return vals


def parse_csv_strings(text: str) -> List[str]:
    vals = [str(x).strip() for x in str(text).split(",") if str(x).strip()]
    if not vals:
        raise ValueError("empty string list")
    return vals


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


def resolve_dataset_root(root: str) -> Path:
    if str(root or "").strip():
        candidate = Path(str(root))
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"LongBench dataset root not found: {root}")
    for probe in SEARCH_ROOTS:
        if not probe.exists():
            continue
        for name in ("LongBench", "longbench", "LongBench-data"):
            candidate = probe / name
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError("LongBench dataset root not found; pass --dataset-root explicitly")


def resolve_task_file(dataset_root: Path, task: str) -> Path:
    candidates = [
        dataset_root / f"{task}.jsonl",
        dataset_root / f"{task}.json",
        dataset_root / task / "test.jsonl",
        dataset_root / task / "test.json",
        dataset_root / task / "validation.jsonl",
        dataset_root / task / "validation.json",
        dataset_root / task / f"{task}.jsonl",
        dataset_root / task / f"{task}.json",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    hits = list(dataset_root.rglob(f"{task}.jsonl")) + list(dataset_root.rglob(f"{task}.json"))
    if hits:
        return hits[0].resolve()
    raise FileNotFoundError(f"task file not found for {task} under {dataset_root}")


def render_user_prompt(tokenizer, text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": str(text)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            pass
    return f"user: {text}\n\nassistant:"


def count_tokens(tokenizer, text: str) -> int:
    return int(len(tokenizer(str(text), add_special_tokens=False).input_ids))


def build_prompt_text(record: Dict[str, Any]) -> str:
    if str(record.get("prompt", "")).strip():
        return str(record["prompt"])
    parts: List[str] = []
    field_labels = [
        ("instruction", "Instruction"),
        ("context", "Context"),
        ("passage", "Passage"),
        ("article", "Article"),
        ("document", "Document"),
        ("documents", "Documents"),
        ("input", "Input"),
        ("query", "Query"),
        ("question", "Question"),
    ]
    for field, label in field_labels:
        value = record.get(field)
        if value is None:
            continue
        text = ""
        if isinstance(value, list):
            text = "\n".join(str(x) for x in value if str(x).strip())
        else:
            text = str(value).strip()
        if text:
            parts.append(f"{label}:\n{text}")
    if not parts:
        parts.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return "\n\n".join(parts)


def extract_answers(record: Dict[str, Any]) -> List[str]:
    for key in ("answers", "answer", "label", "ground_truth"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            out = [str(x).strip() for x in value if str(x).strip()]
        else:
            out = [str(value).strip()] if str(value).strip() else []
        if out:
            return out
    return []


def normalize_text(text: str) -> str:
    s = str(text or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common: Dict[str, int] = {}
    for tok in gold_tokens:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in pred_tokens:
        count = common.get(tok, 0)
        if count > 0:
            overlap += 1
            common[tok] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(gold_tokens))
    return float(2 * precision * recall / max(1e-8, precision + recall))


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def score_output(task: str, output_text: str, answers: Sequence[str]) -> Tuple[float, int]:
    if not str(output_text or "").strip() or not answers:
        return 0.0, 0
    metric = TASK_METRIC.get(str(task), "token_f1")
    if metric == "exact_match":
        score = max(exact_match(output_text, ans) for ans in answers)
    else:
        score = max(token_f1(output_text, ans) for ans in answers)
    return float(score), 1


def build_task_samples(
    tokenizer,
    task: str,
    records: Sequence[Dict[str, Any]],
    samples_per_task: int,
    max_prompt_tokens: int,
    seed: int,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        answers = extract_answers(record)
        prompt_text = build_prompt_text(record)
        prompt = render_user_prompt(tokenizer, prompt_text)
        prompt_tokens = count_tokens(tokenizer, prompt)
        if prompt_tokens > int(max_prompt_tokens):
            continue
        candidates.append(
            {
                "task": str(task),
                "sample_idx": int(idx),
                "prompt": prompt,
                "prompt_tokens": int(prompt_tokens),
                "answers": list(answers),
                "score_metric": TASK_METRIC.get(str(task), "token_f1"),
            }
        )
    if not candidates:
        raise RuntimeError(f"no usable samples for task {task}")
    task_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(task)))
    rng = random.Random(int(seed) ^ int(task_seed))
    if len(candidates) <= int(samples_per_task):
        rng.shuffle(candidates)
        return candidates
    return rng.sample(candidates, int(samples_per_task))


def eval_longbench_output(item: Dict[str, Any], output_text: str) -> Dict[str, Any]:
    answers = list(item.get("answers", []) or [])
    score, scoreable = score_output(str(item["task"]), output_text, answers)
    return {
        "valid": int(scoreable),
        "task": str(item["task"]),
        "sample_idx": int(item["sample_idx"]),
        "prompt_tokens": int(item["prompt_tokens"]),
        "score": float(score),
        "scoreable": int(scoreable),
        "score_metric": str(item.get("score_metric", "token_f1")),
    }


def iter_batches(items: Sequence[Dict[str, Any]], concurrency: int) -> Iterable[Sequence[Dict[str, Any]]]:
    size = max(1, int(concurrency))
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def run_vllm_task(
    endpoint: str,
    tokenizer,
    task: str,
    samples: Sequence[Dict[str, Any]],
    concurrency: int,
    max_new_tokens: int,
    request_timeout_s: float,
) -> Dict[str, Any]:
    request_rows: List[Dict[str, Any]] = []
    batch_wall_ms_list: List[float] = []
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
            extra = eval_longbench_output(item, str(rec.get("output_text", "")))
            request_rows.append({**rec, **extra})
    agg = aggregate_streaming_metrics(
        request_rows=request_rows,
        requested_repeats=1,
        valid_fn=lambda text: bool(str(text or "").strip()),
        batch_wall_ms_list=batch_wall_ms_list,
    )
    requested = len(samples)
    total_score = sum(float(row.get("score", 0.0)) for row in request_rows)
    valid_count = sum(int(row.get("scoreable", 0)) for row in request_rows)
    return {
        "task": str(task),
        "method": "vllm",
        "concurrency": int(concurrency),
        "score": float(total_score / max(1, requested)),
        "score_metric": str(samples[0].get("score_metric", "token_f1")) if samples else "token_f1",
        "completion_rate": agg["completion_rate"],
        "valid_completion_rate": float(valid_count / max(1, requested)),
        "oom_failure_count": agg["oom_failure_count"],
        "status": agg["status"],
        "tokens_per_sec": agg["tokens_per_sec"],
        "ttft_p95_ms": agg["ttft_p95_ms"],
        "ttft_p99_ms": agg["ttft_p99_ms"],
        "itl_p95_ms": agg["itl_p95_ms"],
        "itl_p99_ms": agg["itl_p99_ms"],
        "min_free_blocks": -1,
        "min_free_block_ratio": -1.0,
        "wall_clock_total_runtime_ms": agg["wall_clock_total_runtime_ms"],
        "error_reason": agg["error_reason"],
        "rows": request_rows,
    }


def run_internal_task(
    model_name: str,
    method: str,
    gpu_mem_frac: float,
    tokenizer,
    task: str,
    samples: Sequence[Dict[str, Any]],
    concurrency: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    summary = run_internal_prompt_batches(
        model_name=model_name,
        method=method,
        gpu_mem_frac=float(gpu_mem_frac),
        prompts=[str(item["prompt"]) for item in samples],
        items=list(samples),
        concurrency=int(concurrency),
        max_new_tokens=int(max_new_tokens),
        tokenizer=tokenizer,
        repeats=1,
        evaluate_output_fn=eval_longbench_output,
    )
    rows = list(summary["rows"])
    requested = max(1, len(samples))
    total_score = sum(float(row.get("score", 0.0)) for row in rows)
    valid_count = sum(int(row.get("scoreable", 0)) for row in rows)
    return {
        "task": str(task),
        "method": str(method),
        "concurrency": int(concurrency),
        "gpu_mem_frac": float(gpu_mem_frac),
        "score": float(total_score / requested),
        "score_metric": str(samples[0].get("score_metric", "token_f1")) if samples else "token_f1",
        "completion_rate": summary["completion_rate"],
        "valid_completion_rate": float(valid_count / requested),
        "oom_failure_count": summary["oom_failure_count"],
        "status": summary["status"],
        "tokens_per_sec": summary["tokens_per_sec"],
        "ttft_p95_ms": summary["ttft_p95_ms"],
        "ttft_p99_ms": summary["ttft_p99_ms"],
        "itl_p95_ms": summary["itl_p95_ms"],
        "itl_p99_ms": summary["itl_p99_ms"],
        "min_free_blocks": summary["min_free_blocks"],
        "min_free_block_ratio": summary["min_free_block_ratio"],
        "wall_clock_total_runtime_ms": summary["wall_clock_total_runtime_ms"],
        "error_reason": summary["error_reason"],
        "compression_profile": summary["compression_profile"],
        "rows": rows,
    }


def run_hf_style_task(
    model_name: str,
    method: str,
    tokenizer,
    task: str,
    samples: Sequence[Dict[str, Any]],
    concurrency: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    summary = run_hf_style_prompt_batches(
        model_name=model_name,
        method=method,
        items=list(samples),
        concurrency=int(concurrency),
        max_new_tokens=int(max_new_tokens),
        repeats=1,
        evaluate_output_fn=eval_longbench_output,
        tokenizer=tokenizer,
    )
    rows = list(summary["rows"])
    requested = max(1, len(samples))
    total_score = sum(float(row.get("score", 0.0)) for row in rows)
    valid_count = sum(int(row.get("scoreable", 0)) for row in rows)
    return {
        "task": str(task),
        "method": str(method),
        "concurrency": int(concurrency),
        "score": float(total_score / requested),
        "score_metric": str(samples[0].get("score_metric", "token_f1")) if samples else "token_f1",
        "completion_rate": summary["completion_rate"],
        "valid_completion_rate": float(valid_count / requested),
        "oom_failure_count": summary["oom_failure_count"],
        "status": summary["status"],
        "tokens_per_sec": summary["tokens_per_sec"],
        "ttft_p95_ms": summary["ttft_p95_ms"],
        "ttft_p99_ms": summary["ttft_p99_ms"],
        "itl_p95_ms": summary["itl_p95_ms"],
        "itl_p99_ms": summary["itl_p99_ms"],
        "min_free_blocks": summary["min_free_blocks"],
        "min_free_block_ratio": summary["min_free_block_ratio"],
        "wall_clock_total_runtime_ms": summary["wall_clock_total_runtime_ms"],
        "error_reason": summary["error_reason"],
        "stack_type": summary["stack_type"],
        "kv_backend": summary["kv_backend"],
        "fallback_policy": summary["fallback_policy"],
        "compression_profile": summary["compression_profile"],
        "rows": rows,
    }


def build_macro_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["method"]), int(row["concurrency"]))
        grouped.setdefault(key, []).append(dict(row))
    out: List[Dict[str, Any]] = []
    for (method, concurrency), items in sorted(grouped.items()):
        out.append(
            {
                "method": method,
                "concurrency": int(concurrency),
                "macro_score": float(sum(float(x["score"]) for x in items) / max(1, len(items))),
                "completion_rate": float(sum(float(x["completion_rate"]) for x in items) / max(1, len(items))),
                "valid_completion_rate": float(sum(float(x["valid_completion_rate"]) for x in items) / max(1, len(items))),
            }
        )
    base_scores = {(row["method"], row["task"]): float(row["score"]) for row in rows if int(row["concurrency"]) == 1}
    for row in rows:
        key = (row["method"], row["task"])
        base = base_scores.get(key)
        if base is None:
            row["score_delta_from_c1"] = 0.0
        else:
            row["score_delta_from_c1"] = float(float(row["score"]) - float(base))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="LongBench concurrency runner for vLLM and internal methods.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--methods", type=str, default="vllm,off_compress,p2_only_compress")
    parser.add_argument("--samples-per-task", type=int, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--concurrency-list", type=str, default="1,16")
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--gpu-mem-frac-map", type=str, default="")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--server-log", type=str, default="")
    parser.add_argument("--out", type=str, default="benchmark_longbench_concurrency.json")
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.dataset_root)
    methods = parse_methods(args.methods, allow_vllm=True)
    tasks = parse_csv_strings(args.tasks)
    concurrencies = parse_csv_ints(args.concurrency_list)
    tokenizer = load_tokenizer(args.model_name)
    frac_map = parse_method_frac_map(args.gpu_mem_frac_map)

    task_samples: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        task_path = resolve_task_file(dataset_root, task)
        records = load_records(task_path)
        task_samples[task] = build_task_samples(
            tokenizer=tokenizer,
            task=task,
            records=records,
            samples_per_task=int(args.samples_per_task),
            max_prompt_tokens=int(args.max_prompt_tokens),
            seed=int(args.seed),
        )

    rows: List[Dict[str, Any]] = []
    for method in methods:
        endpoint = str(args.base_url).strip()
        server_proc = None
        launched_server = False
        if method == "vllm" and not endpoint:
            max_prompt_len = max(int(sample["prompt_tokens"]) for samples in task_samples.values() for sample in samples)
            max_model_len = max_prompt_len + int(args.max_new_tokens) + 512
            server_proc, port = start_vllm_server(
                model_name=args.model_name,
                gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
                max_model_len=max_model_len,
                host=DEFAULT_SERVER_HOST,
                log_path=Path(args.server_log) if str(args.server_log).strip() else None,
            )
            endpoint = f"http://{DEFAULT_SERVER_HOST}:{int(port)}"
            launched_server = True
        try:
            for task in tasks:
                samples = task_samples[task]
                for concurrency in concurrencies:
                    if method == "vllm":
                        row = run_vllm_task(
                            endpoint=endpoint,
                            tokenizer=tokenizer,
                            task=task,
                            samples=samples,
                            concurrency=int(concurrency),
                            max_new_tokens=int(args.max_new_tokens),
                            request_timeout_s=float(args.request_timeout_s),
                        )
                    else:
                        if is_hf_style_method(method):
                            row = run_hf_style_task(
                                model_name=args.model_name,
                                method=method,
                                tokenizer=tokenizer,
                                task=task,
                                samples=samples,
                                concurrency=int(concurrency),
                                max_new_tokens=int(args.max_new_tokens),
                            )
                        else:
                            if method not in frac_map:
                                raise ValueError(f"missing gpu mem frac for method '{method}'")
                            row = run_internal_task(
                                model_name=args.model_name,
                                method=method,
                                gpu_mem_frac=float(frac_map[method]),
                                tokenizer=tokenizer,
                                task=task,
                                samples=samples,
                                concurrency=int(concurrency),
                                max_new_tokens=int(args.max_new_tokens),
                            )
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "method": row["method"],
                                "task": row["task"],
                                "concurrency": row["concurrency"],
                                "score": row["score"],
                                "completion_rate": row["completion_rate"],
                                "valid_completion_rate": row["valid_completion_rate"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        finally:
            if launched_server:
                stop_vllm_server(server_proc)

    macro_rows = build_macro_rows(rows)
    payload = {
        "meta": {
            "task": "longbench_concurrency",
            "model_name": str(args.model_name),
            "dataset_root": str(dataset_root),
            "tasks": list(tasks),
            "methods": list(methods),
            "samples_per_task": int(args.samples_per_task),
            "max_prompt_tokens": int(args.max_prompt_tokens),
            "max_new_tokens": int(args.max_new_tokens),
            "concurrency_list": [int(x) for x in concurrencies],
            "seed": int(args.seed),
            "valid_definition": "request completed normally and the task scoring script can score the sample",
            "score_note": "Selected LongBench tasks are scored with in-runner normalized exact-match or token-F1 proxies.",
        },
        "rows": rows,
        "macro_rows": macro_rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

