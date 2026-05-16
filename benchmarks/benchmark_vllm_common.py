import json
import os
import random
import re
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests
from transformers import AutoConfig, AutoTokenizer


VLLM_SERVED_MODEL_NAME = "benchmark-model"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_READY_TIMEOUT_S = 240
DEFAULT_REQUEST_TIMEOUT_S = 900
DEFAULT_HEALTH_POLL_S = 1.0

OOM_KEYWORDS = (
    "out of memory",
    "cuda out of memory",
    "alloc_failed",
    "allocation failed",
    "no free memory",
)

MAX_LEN_SENTINEL = 1_000_000
CHAT_COMPLETION_SAFETY_TOKENS = 64


def percentile(vals: Sequence[float], q: float) -> float:
    seq = sorted(float(v) for v in vals if v is not None)
    if not seq:
        return 0.0
    if len(seq) == 1:
        return float(seq[0])
    pos = max(0.0, min(1.0, float(q))) * (len(seq) - 1)
    lo = int(pos)
    hi = min(len(seq) - 1, lo + 1)
    frac = pos - lo
    return float(seq[lo] * (1.0 - frac) + seq[hi] * frac)


def completion_rate_from_counts(completed: int, requested: int) -> float:
    return float(completed / max(1, requested))


def classify_status_from_rates(
    completion_rate: float,
    valid_completion_rate: float,
    oom_failure_count: int,
    error_reason: str = "",
) -> str:
    text = str(error_reason or "").strip()
    if valid_completion_rate >= 0.999 and oom_failure_count == 0 and not text:
        return "Success"
    if completion_rate > 0.0 and oom_failure_count == 0 and not text:
        return "Degraded"
    return "Failed/OOM"


def classify_frontier_reason(status: str, error_reason: str, completion_rate: float, valid_completion_rate: float) -> str:
    text = str(error_reason or "").lower()
    if status == "Success":
        return "stable"
    if any(word in text for word in OOM_KEYWORDS):
        return "oom"
    if "timeout" in text:
        return "timeout"
    if completion_rate > 0.0 and valid_completion_rate < 0.999:
        return "degraded_valid"
    return "runtime_failure"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((DEFAULT_SERVER_HOST, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


@lru_cache(maxsize=32)
def infer_model_max_len(model_name: str) -> Optional[int]:
    model_path = Path(str(model_name))
    cfg: Dict[str, Any] = {}
    try:
        if model_path.exists():
            config_path = model_path / "config.json"
            if config_path.exists():
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if not cfg:
            cfg = dict(AutoConfig.from_pretrained(str(model_name), trust_remote_code=True).to_dict())
    except Exception:
        return None
    candidates: List[int] = []
    for key in ("max_model_len", "model_max_length", "max_position_embeddings", "seq_length", "n_positions"):
        value = cfg.get(key)
        if isinstance(value, (int, float)):
            ivalue = int(value)
            if ivalue > 0 and ivalue < MAX_LEN_SENTINEL:
                candidates.append(ivalue)
    if not candidates:
        return None
    return int(min(candidates))


def resolve_vllm_max_model_len(model_name: str, requested_max_model_len: int) -> int:
    requested = max(1, int(requested_max_model_len))
    inferred = infer_model_max_len(str(model_name))
    if inferred is None:
        return requested
    return int(min(requested, inferred))


def wait_for_server_ready(host: str, port: int, timeout_s: float = DEFAULT_SERVER_READY_TIMEOUT_S) -> None:
    deadline = time.time() + float(timeout_s)
    url_candidates = [
        f"http://{host}:{port}/health",
        f"http://{host}:{port}/v1/models",
    ]
    last_error = ""
    while time.time() < deadline:
        for url in url_candidates:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code < 500:
                    return
            except Exception as exc:
                last_error = str(exc)
        time.sleep(DEFAULT_HEALTH_POLL_S)
    raise RuntimeError(f"vLLM server failed to become ready within {timeout_s}s: {last_error}")


def read_log_tail(log_path: Optional[Path], max_lines: int = 80) -> str:
    if log_path is None:
        return ""
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-max(1, int(max_lines)):])


def build_vllm_offload_extra_args(
    swap_space: Optional[float] = None,
    cpu_offload_gb: Optional[float] = None,
) -> List[str]:
    args: List[str] = []
    if swap_space is not None:
        args.extend(["--swap-space", str(float(swap_space))])
    if cpu_offload_gb is not None:
        args.extend(["--cpu-offload-gb", str(float(cpu_offload_gb))])
    return args


def start_vllm_server(
    model_name: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    host: str = DEFAULT_SERVER_HOST,
    port: Optional[int] = None,
    log_path: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> Tuple[subprocess.Popen, int]:
    port = int(port or find_free_port())
    effective_max_model_len = resolve_vllm_max_model_len(model_name, int(max_model_len))
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_name),
        "--served-model-name",
        VLLM_SERVED_MODEL_NAME,
        "--host",
        str(host),
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(float(gpu_memory_utilization)),
        "--max-model-len",
        str(int(effective_max_model_len)),
        "--trust-remote-code",
        "--disable-log-requests",
    ]
    if extra_args:
        cmd.extend([str(x) for x in extra_args])
    log_fh = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh or subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        cwd=str(Path(model_name).resolve().parent if Path(model_name).exists() else Path.cwd()),
    )
    try:
        wait_for_server_ready(host, port)
    except Exception as exc:
        log_tail = read_log_tail(Path(log_path) if log_path is not None else None)
        stop_vllm_server(proc)
        detail = str(exc)
        if log_tail:
            detail = f"{detail}\n--- vLLM log tail ---\n{log_tail}"
        raise RuntimeError(detail)
    return proc, port


def stop_vllm_server(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    except Exception:
        pass


def build_chat_payload(prompt: str, max_new_tokens: int, ignore_eos: bool = False) -> Dict[str, Any]:
    payload = {
        "model": VLLM_SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": str(prompt)}],
        "max_tokens": int(max_new_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
    }
    if bool(ignore_eos):
        # vLLM OpenAI-compatible API accepts vLLM sampling params as extra JSON fields.
        payload["ignore_eos"] = True
    return payload


def render_user_chat_prompt(tokenizer, user_text: str) -> str:
    prompt = str(user_text)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": str(user_text)}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = str(user_text)
    return str(prompt)


def count_prompt_tokens(tokenizer, prompt_text: str) -> int:
    return int(len(tokenizer(str(prompt_text), add_special_tokens=False).input_ids))


def max_prompt_tokens_for_request(max_model_len: int, max_new_tokens: int, safety_tokens: int = CHAT_COMPLETION_SAFETY_TOKENS) -> int:
    return max(1, int(max_model_len) - int(max_new_tokens) - int(safety_tokens))


def fit_user_text_to_max_prompt_tokens(tokenizer, user_text: str, max_prompt_tokens: int, max_iters: int = 24) -> Tuple[str, int]:
    text = str(user_text)
    prompt = render_user_chat_prompt(tokenizer, text)
    actual = count_prompt_tokens(tokenizer, prompt)
    limit = max(1, int(max_prompt_tokens))
    for _ in range(max(1, int(max_iters))):
        if actual <= limit:
            break
        overflow = max(1, actual - limit)
        trim_chars = max(64, min(len(text) // 6 if len(text) > 0 else 64, overflow * 6))
        if trim_chars >= len(text):
            text = text[: max(1, len(text) // 2)]
        else:
            text = text[:-trim_chars]
        prompt = render_user_chat_prompt(tokenizer, text)
        actual = count_prompt_tokens(tokenizer, prompt)
    return str(prompt), int(actual)


def stream_chat_completion(
    base_url: str,
    prompt: str,
    max_new_tokens: int,
    tokenizer,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ignore_eos: bool = False,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    started_at = time.perf_counter()
    first_token_at: Optional[float] = None
    piece_times: List[float] = []
    pieces: List[str] = []
    finish_reason = ""
    with requests.post(
        url,
        json=build_chat_payload(prompt, max_new_tokens, ignore_eos=ignore_eos),
        stream=True,
        timeout=(10, float(request_timeout_s)),
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = str(raw_line).strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            choice = dict((event.get("choices") or [{}])[0])
            delta = dict(choice.get("delta") or {})
            text = str(delta.get("content") or "")
            now = time.perf_counter()
            if text:
                pieces.append(text)
                piece_times.append(now)
                if first_token_at is None:
                    first_token_at = now
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice.get("finish_reason") or "")
    ended_at = time.perf_counter()
    output_text = "".join(pieces)
    completion_tokens = 0
    if str(output_text).strip():
        completion_tokens = len(tokenizer(str(output_text), add_special_tokens=False).input_ids)
    ttft_ms = 0.0
    if first_token_at is not None:
        ttft_ms = (first_token_at - started_at) * 1000.0
    avg_itl_ms = 0.0
    if len(piece_times) >= 2:
        deltas_ms = [(cur - prev) * 1000.0 for prev, cur in zip(piece_times, piece_times[1:])]
        avg_itl_ms = float(mean(deltas_ms)) if deltas_ms else 0.0
    elif first_token_at is not None and completion_tokens > 1:
        avg_itl_ms = ((ended_at - first_token_at) * 1000.0) / max(1, completion_tokens - 1)
    return {
        "output_text": output_text,
        "completion_tokens": int(completion_tokens),
        "ttft_ms": float(round(ttft_ms, 3)),
        "avg_itl_ms": float(round(avg_itl_ms, 3)),
        "finish_reason": finish_reason,
        "completed": int(bool(str(output_text).strip()) or finish_reason in {"stop", "length"}),
        "wall_ms": float(round((ended_at - started_at) * 1000.0, 3)),
    }


def run_prompt_batch(
    base_url: str,
    prompts: Sequence[str],
    max_new_tokens: int,
    tokenizer,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ignore_eos: bool = False,
) -> List[Dict[str, Any]]:
    items = list(prompts)
    results: List[Optional[Dict[str, Any]]] = [None] * len(items)

    def _worker(idx_prompt: Tuple[int, str]) -> Tuple[int, Dict[str, Any]]:
        idx, prompt = idx_prompt
        try:
            rec = stream_chat_completion(base_url, prompt, max_new_tokens, tokenizer, request_timeout_s=request_timeout_s, ignore_eos=ignore_eos)
            rec["error_reason"] = ""
            return idx, rec
        except Exception as exc:
            return idx, {
                "output_text": "",
                "completion_tokens": 0,
                "ttft_ms": 0.0,
                "avg_itl_ms": 0.0,
                "finish_reason": "",
                "completed": 0,
                "wall_ms": 0.0,
                "error_reason": str(exc),
            }

    with ThreadPoolExecutor(max_workers=max(1, len(items))) as pool:
        futures = [pool.submit(_worker, item) for item in enumerate(items)]
        for fut in as_completed(futures):
            idx, rec = fut.result()
            results[idx] = rec
    return [dict(x or {}) for x in results]


def align_prompt_tolerance(target_tokens: int) -> int:
    return max(1, min(int(round(target_tokens * 0.02)), 256))


def build_synthetic_prompts_for_cell(
    tokenizer,
    target_tokens: int,
    concurrency: int,
    max_prompt_tokens: Optional[int] = None,
) -> Tuple[List[str], List[int]]:
    prompts: List[str] = []
    actuals: List[int] = []
    tol = align_prompt_tolerance(target_tokens)
    base_user = (
        "You are a systems assistant. Read the long context and continue with a coherent, technical explanation "
        "about KV-cache management, prefill/decode scheduling, batching, and memory pressure handling. "
        "Do not answer with bullets only; write normal explanatory prose."
    )
    filler = (
        " Context fragment discusses cache locality, request interleaving, scheduler fairness, pinned-memory copies, "
        "windowed retention, and latency-throughput tradeoffs in long-context serving."
    )
    for seq_id in range(int(concurrency)):
        user_text = f"[seq={seq_id}] {base_user}"
        prompt = user_text
        actual = 0
        for _ in range(12):
            prompt = render_user_chat_prompt(tokenizer, user_text)
            actual = count_prompt_tokens(tokenizer, prompt)
            diff = int(target_tokens) - int(actual)
            if abs(diff) <= tol:
                break
            if diff > 0:
                user_text += filler * max(1, diff // 24)
            else:
                trim = max(64, min(len(user_text) // 8, abs(diff) * 4))
                user_text = user_text[:-trim] if trim < len(user_text) else user_text
        if max_prompt_tokens is not None and actual > int(max_prompt_tokens):
            prompt, actual = fit_user_text_to_max_prompt_tokens(tokenizer, user_text, int(max_prompt_tokens))
        prompts.append(prompt)
        actuals.append(int(actual))
    return prompts, actuals


def first_nonempty_line(text: str) -> str:
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def first_6digit_compat(text: str) -> str:
    m = re.search(r"\b(\d{6})\b", str(text))
    return m.group(1) if m else ""


def first_6digit_strict(text: str) -> str:
    first = first_nonempty_line(text).strip("`").strip()
    m = re.fullmatch(r"(?:answer\s*[:?]?\s*)?(\d{6})", first or "", flags=re.IGNORECASE)
    return m.group(1) if m else ""


def build_passkey_prompt(
    tokenizer,
    target_tokens: int,
    depth: float,
    answer: str,
    seq_label: str,
    max_prompt_tokens: Optional[int] = None,
) -> Tuple[str, int]:
    intro = (
        f"[{seq_label}] You are doing a long-context retrieval test.\n"
        "Find PASSKEY_RECORD in the context and return ONLY the 6-digit number.\n"
    )
    needle = f"\nPASSKEY_RECORD: passkey::{answer}::end\n"
    question = (
        "\nQuestion: What is the passkey in PASSKEY_RECORD?\n"
        "Instruction: Output exactly one line with only the 6-digit number.\n"
    )
    filler = "Irrelevant context about serving systems, KV blocks, scheduling, batching, and memory pressure.\n"

    intro_ids = tokenizer(intro, add_special_tokens=False).input_ids
    needle_ids = tokenizer(needle, add_special_tokens=False).input_ids
    question_ids = tokenizer(question, add_special_tokens=False).input_ids
    filler_ids = tokenizer(filler, add_special_tokens=False).input_ids or [32]

    fixed = len(intro_ids) + len(needle_ids) + len(question_ids)
    remaining = max(0, int(target_tokens) - fixed)
    left = int(round(remaining * float(depth)))
    right = max(0, remaining - left)

    def _fill_to(n: int) -> List[int]:
        reps = (n + len(filler_ids) - 1) // len(filler_ids)
        return (filler_ids * reps)[:n]

    current_target_tokens = int(target_tokens)
    for _ in range(16):
        remaining = max(0, int(current_target_tokens) - fixed)
        left = int(round(remaining * float(depth)))
        right = max(0, remaining - left)
        ids = intro_ids + _fill_to(left) + needle_ids + _fill_to(right) + question_ids
        prompt_user = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        prompt = render_user_chat_prompt(tokenizer, prompt_user)
        actual = count_prompt_tokens(tokenizer, prompt)
        if max_prompt_tokens is None or actual <= int(max_prompt_tokens):
            return prompt, int(actual)
        current_target_tokens = max(fixed, int(current_target_tokens) - max(32, actual - int(max_prompt_tokens)))
    return prompt, int(actual)


def build_passkey_cases(
    tokenizer,
    input_len: int,
    depths: Sequence[float],
    keys_per_depth: int,
    seed: int,
    max_prompt_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed) ^ (int(input_len) * 1009) ^ (len(depths) * 313))
    cases: List[Dict[str, Any]] = []
    for depth in depths:
        for sample_idx in range(int(keys_per_depth)):
            answer = f"{rng.randint(100000, 999999)}"
            seq_label = f"d{float(depth):.2f}_i{sample_idx}"
            prompt, actual_tokens = build_passkey_prompt(
                tokenizer,
                int(input_len),
                float(depth),
                answer,
                seq_label,
                max_prompt_tokens=max_prompt_tokens,
            )
            cases.append(
                {
                    "depth": float(depth),
                    "sample_idx": int(sample_idx),
                    "answer": str(answer),
                    "prompt": prompt,
                    "actual_prompt_tokens": int(actual_tokens),
                    "seq_label": str(seq_label),
                }
            )
    rng.shuffle(cases)
    return cases


def aggregate_streaming_metrics(
    request_rows: Sequence[Dict[str, Any]],
    requested_repeats: int,
    valid_fn: Callable[[str], bool],
    batch_wall_ms_list: Optional[Sequence[float]] = None,
    count_oom_from_errors: bool = True,
) -> Dict[str, Any]:
    requested = len(request_rows)
    completed = sum(int(row.get("completed", 0)) for row in request_rows)
    valid = sum(1 for row in request_rows if int(row.get("completed", 0)) == 1 and valid_fn(str(row.get("output_text", ""))))
    total_tokens = sum(int(row.get("completion_tokens", 0)) for row in request_rows)
    if batch_wall_ms_list:
        total_wall_ms = float(sum(float(x) for x in batch_wall_ms_list))
    else:
        total_wall_ms = float(sum(float(row.get("wall_ms", 0.0)) for row in request_rows))
    ttfts = [float(row.get("ttft_ms", 0.0)) for row in request_rows if float(row.get("ttft_ms", 0.0)) > 0]
    avg_itls = [float(row.get("avg_itl_ms", 0.0)) for row in request_rows if float(row.get("avg_itl_ms", 0.0)) > 0]
    errors = [str(row.get("error_reason", "")).strip() for row in request_rows if str(row.get("error_reason", "")).strip()]
    oom_count = 0
    if count_oom_from_errors:
        oom_count = sum(1 for err in errors if any(word in err.lower() for word in OOM_KEYWORDS) or "timeout" in err.lower())
    completion_rate = completion_rate_from_counts(completed, requested)
    valid_completion_rate = float(valid / max(1, requested))
    status = classify_status_from_rates(completion_rate, valid_completion_rate, oom_count, "; ".join(errors[:3]))
    return {
        "requested_repeats": int(requested_repeats),
        "completed_repeats": int(completed),
        "completion_rate": float(completion_rate),
        "valid_completion_rate": float(valid_completion_rate),
        "oom_failure_count": int(oom_count),
        "status": status,
        "frontier_reason": classify_frontier_reason(status, "; ".join(errors[:3]), completion_rate, valid_completion_rate),
        "tokens_per_sec": float((1000.0 * total_tokens / total_wall_ms) if total_wall_ms > 0 else 0.0),
        "ttft_p95_ms": float(round(percentile(ttfts, 0.95), 3)),
        "ttft_p99_ms": float(round(percentile(ttfts, 0.99), 3)),
        "itl_p95_ms": float(round(percentile(avg_itls, 0.95), 3)),
        "itl_p99_ms": float(round(percentile(avg_itls, 0.99), 3)),
        "wall_clock_total_runtime_ms": float(round(total_wall_ms, 3)),
        "min_free_blocks": -1,
        "min_free_block_ratio": -1.0,
        "error_reason": "; ".join(errors[:3]),
    }


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(str(model_name), trust_remote_code=True)

