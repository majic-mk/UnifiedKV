#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import time
import traceback
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
sys.path.insert(0, str(BENCHMARKS / "configs"))

from benchmark_internal_common import build_engine, cleanup_engine  # noqa: E402
from benchmark_vllm_common import load_tokenizer  # noqa: E402
from hf_style_common import (  # noqa: E402
    _batch_eos_token_ids,
    _cleanup_cuda,
    _encode_manifest_batch,
    _ensure_padding_token,
    _load_model,
    _model_input_device,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--subset", choices=["gate", "full"], default="gate")
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--gpu-mem-frac", type=float, default=0.35)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--longbench-repo",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/LongBench/repo/LongBench"),
    )
    return parser.parse_args()


def hash_token_ids(token_ids):
    payload = b"".join(struct.pack("<I", int(token_id)) for token_id in token_ids)
    return hashlib.sha256(payload).hexdigest()


def load_metric_map(repo):
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("longbench_eval", repo / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.dataset2metric)


def score(metric_map, row, prediction):
    candidate = str(prediction or "")
    if row["task"] in {"trec", "triviaqa", "samsum", "lsht"}:
        candidate = candidate.lstrip("\n").split("\n")[0]
    value = 0.0
    for answer in row["answers"]:
        value = max(
            value,
            float(
                metric_map[row["task"]](
                    candidate,
                    str(answer),
                    all_classes=list(row.get("all_classes") or []),
                )
            ),
        )
    return 100.0 * value


def load_rows(manifest_dir, task, subset, limit):
    rows = [
        json.loads(line)
        for line in (manifest_dir / f"{task}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if subset == "gate":
        rows = [row for row in rows if row.get("gate_selected")]
    if limit > 0:
        rows = rows[:limit]
    for row in rows:
        actual = hash_token_ids(row["input_ids"])
        if actual != row["input_sha256"]:
            raise RuntimeError(
                f"manifest hash mismatch: {task}/{row['row_index']}"
            )
    return rows


def append_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def completed_keys(path):
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((row["task"], int(row["row_index"])))
    return keys


def run_hf(args, tasks, metric_map, tokenizer, done):
    model = _load_model(args.model, tokenizer)
    device = _model_input_device(model)
    try:
        for task in tasks:
            for row in load_rows(
                args.manifest_dir,
                task,
                args.subset,
                args.limit_per_task,
            ):
                key = (task, int(row["row_index"]))
                if key in done:
                    continue
                started = time.perf_counter()
                status = "success"
                error = ""
                prediction = ""
                generated_tokens = 0
                try:
                    encoded = _encode_manifest_batch(
                        [row],
                        int(tokenizer.pad_token_id),
                        device,
                    )
                    width = int(encoded["input_ids"].shape[1])
                    eos_ids = _batch_eos_token_ids([row])
                    with torch.inference_mode():
                        output = model.generate(
                            **encoded,
                            max_new_tokens=int(row["max_new_tokens"]),
                            min_new_tokens=1,
                            num_beams=1,
                            do_sample=False,
                            use_cache=True,
                            pad_token_id=int(tokenizer.pad_token_id),
                            eos_token_id=eos_ids,
                        )
                    generated = output[0, width:]
                    generated_tokens = int(generated.numel())
                    prediction = tokenizer.decode(
                        generated,
                        skip_special_tokens=True,
                    )
                    del encoded, output, generated
                except Exception as exc:
                    status = "failure"
                    error = (
                        f"{type(exc).__name__}: {exc}\n"
                        f"{traceback.format_exc()}"
                    )
                    _cleanup_cuda()
                result = {
                    **{key: row[key] for key in [
                        "task",
                        "row_index",
                        "input_tokens",
                        "input_sha256",
                        "max_new_tokens",
                        "eos_token_ids",
                        "prompt_mode",
                    ]},
                    "method": "hf_vanilla",
                    "status": status,
                    "prediction": prediction,
                    "score": score(metric_map, row, prediction)
                    if status == "success"
                    else 0.0,
                    "generated_tokens": generated_tokens,
                    "wall_time_s": time.perf_counter() - started,
                    "error": error,
                }
                append_result(args.output, result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        del model
        _cleanup_cuda()


def run_internal(args, tasks, metric_map, tokenizer, done):
    max_new = max(
        int(row["max_new_tokens"])
        for task in tasks
        for row in load_rows(
            args.manifest_dir,
            task,
            args.subset,
            max(1, args.limit_per_task) if args.limit_per_task else 1,
        )
    )
    engine = build_engine(
        args.model,
        args.method,
        float(args.gpu_mem_frac),
        max_new,
    )
    try:
        for task in tasks:
            for row in load_rows(
                args.manifest_dir,
                task,
                args.subset,
                args.limit_per_task,
            ):
                key = (task, int(row["row_index"]))
                if key in done:
                    continue
                engine.max_new_tokens = int(row["max_new_tokens"])
                started = time.perf_counter()
                status = "success"
                error = ""
                prediction = ""
                generated_tokens = 0
                metrics = {}
                try:
                    outputs, metrics, details = engine.generate(
                        prompts=[""],
                        prompt_token_ids=[row["input_ids"]],
                        eos_token_ids=[row["eos_token_ids"]],
                        return_metrics=True,
                        return_details=True,
                    )
                    prediction = str(outputs[0])
                    generated_tokens = len(details["token_ids"][0])
                    failed = list(metrics.get("failed_request_errors", []))
                    if failed and failed[0]:
                        raise RuntimeError(str(failed[0]))
                except Exception as exc:
                    status = "failure"
                    error = f"{type(exc).__name__}: {exc}"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                result = {
                    **{key: row[key] for key in [
                        "task",
                        "row_index",
                        "input_tokens",
                        "input_sha256",
                        "max_new_tokens",
                        "eos_token_ids",
                        "prompt_mode",
                    ]},
                    "method": args.method,
                    "status": status,
                    "prediction": prediction,
                    "score": score(metric_map, row, prediction)
                    if status == "success"
                    else 0.0,
                    "generated_tokens": generated_tokens,
                    "wall_time_s": time.perf_counter() - started,
                    "error": error,
                    "decode_backend": metrics.get("decode_backend", ""),
                    "decode_page16_native_steps": metrics.get(
                        "decode_page16_native_steps",
                        0,
                    ),
                    "decode_rebuild_steps": metrics.get(
                        "decode_rebuild_steps",
                        0,
                    ),
                    "peak_cuda_mem_gb": metrics.get("peak_cuda_mem_gb", 0.0),
                }
                append_result(args.output, result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        cleanup_engine(engine)


def main():
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    metric_map = load_metric_map(args.longbench_repo)
    tokenizer = _ensure_padding_token(load_tokenizer(args.model))
    done = completed_keys(args.output)
    if args.method == "hf_vanilla":
        run_hf(args, tasks, metric_map, tokenizer, done)
    else:
        run_internal(args, tasks, metric_map, tokenizer, done)


if __name__ == "__main__":
    main()
