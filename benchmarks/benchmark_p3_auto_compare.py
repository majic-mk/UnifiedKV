# LEGACY_BENCHMARK: retained for historical reference; use benchmark_p3_tiered_table.py and benchmark_p3_quality_passkey.py as paper-facing entry points.
import gc
import json
import time
from typing import Dict, List

import torch

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / 'core'
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from engine import ManagedInferenceEngine


LOCAL_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
CONCURRENCY = 8
PROMPT_REPEAT = 120
MAX_NEW_TOKENS = 256

COMMON_ENGINE_ARGS = {
    "model_name": LOCAL_MODEL_PATH,
    "gpu_mem_frac": 0.05,
    "cpu_mem_gb": 32.0,
    "chunk_size": 1024,
    "max_new_tokens": MAX_NEW_TOKENS,
    "prefill_batch_size": 4,
    "decode_micro_batch_size": 0,
    "decode_window_sink_len": 64,
    "decode_window_recent_len": 512,
}

P3_AUTO_TUNED_ARGS = {
    "decode_window_enabled": False,
    "decode_window_auto_on_pressure": True,
    "decode_window_check_interval": 16,
    "decode_window_emergency_check_interval": 4,
    "decode_window_min_trigger_tokens": 256,
    "decode_window_min_drop_tokens": 256,
    "decode_window_cooldown_steps": 32,
    "decode_window_pressure_steps": 3,
    "decode_window_recover_steps": 8,
    "decode_window_pressure_margin_blocks": 32,
    "decode_window_emergency_margin_blocks": 16,
}

P3_OFF_ARGS = {
    "decode_window_enabled": False,
    "decode_window_auto_on_pressure": False,
}


def build_prompts(n: int, repeat: int) -> List[str]:
    prompts = []
    base = (
        "This is a long-context stress prompt for concurrent decode benchmarking. "
        "Keep generating coherent technical explanation about cache management, "
        "prefill/decode behavior, and memory-pressure fallback strategy. "
    )
    for i in range(n):
        prompts.append(f"[seq={i}] {(base * repeat)} Continue with detailed and non-repetitive content.")
    return prompts


def run_case(case_name: str, extra_args: Dict) -> Dict:
    print(f"\n===== CASE: {case_name} =====")
    args = dict(COMMON_ENGINE_ARGS)
    args.update(extra_args)

    engine = ManagedInferenceEngine(**args)
    engine.tokenizer.eos_token = None
    prompts = build_prompts(CONCURRENCY, PROMPT_REPEAT)

    t0 = time.perf_counter()
    outputs, metrics = engine.generate(prompts, return_metrics=True)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    metrics["case"] = case_name
    metrics["wall_ms"] = round(wall_ms, 3)
    metrics["sample_output"] = outputs[0][:160]
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def pct(base: float, now: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    return (now - base) / base * 100.0


def print_summary(off: Dict, on: Dict):
    print("\n===== SUMMARY (AUTO P3 vs P3 OFF) =====")
    rows = [
        ("tokens_per_sec", "higher"),
        ("decode_step_p95_ms", "lower"),
        ("wall_ms", "lower"),
        ("peak_cuda_mem_gb", "lower"),
        ("decode_min_n_free", "higher"),
        ("decode_window_auto_active_steps", "higher"),
        ("decode_window_prunes", "higher"),
        ("decode_window_tokens_dropped", "higher"),
    ]
    for key, _ in rows:
        a = float(off.get(key, 0.0))
        b = float(on.get(key, 0.0))
        print(f"{key}: off={a:.4f} auto={b:.4f} delta={pct(a, b):+.2f}%")

    status = on.get("decode_window_status", {})
    print(
        "auto_status: "
        f"activations={int(status.get('auto_activations', 0))}, "
        f"auto_active={int(status.get('auto_active', 0))}, "
        f"pressure_streak={int(status.get('pressure_streak', 0))}, "
        f"recover_streak={int(status.get('recover_streak', 0))}, "
        f"check_interval_current={int(status.get('check_interval_current', 0))}"
    )


def main():
    off = run_case("p3_off", P3_OFF_ARGS)
    auto = run_case("p3_auto_tuned", P3_AUTO_TUNED_ARGS)
    print_summary(off, auto)


if __name__ == "__main__":
    main()
