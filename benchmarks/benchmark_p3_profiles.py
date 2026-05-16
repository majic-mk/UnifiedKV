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
PROMPT_REPEAT = 80
MAX_NEW_TOKENS = 192

COMMON_ENGINE_ARGS = {
    "model_name": LOCAL_MODEL_PATH,
    "gpu_mem_frac": 0.05,
    "cpu_mem_gb": 32.0,
    "chunk_size": 1024,
    "max_new_tokens": MAX_NEW_TOKENS,
    "prefill_batch_size": 4,
    "decode_micro_batch_size": 0,
    "decode_window_sink_len": 64,
}

PROFILES = [
    {
        "name": "quality_first",
        "decode_window_recent_len": 256,
        "decode_window_check_interval": 8,
        "decode_window_min_trigger_tokens": 128,
        "decode_window_min_drop_tokens": 64,
    },
    {
        "name": "performance_first",
        "decode_window_recent_len": 64,
        "decode_window_check_interval": 4,
        "decode_window_min_trigger_tokens": 32,
        "decode_window_min_drop_tokens": 32,
    },
]


def build_prompts(n: int, repeat: int) -> List[str]:
    prompts = []
    for i in range(n):
        text = (
            "这是一段高并发长文本压力测试语料，用于比较P3窗口化开启和关闭时的吞吐、延迟和稳定性。"
            * repeat
        )
        prompts.append(f"[seq={i}] {text}\\n请继续生成与主题相关的技术说明。")
    return prompts


def run_case(profile: Dict, p3_enabled: bool) -> Dict:
    case_name = f"{profile['name']}_{'p3_on' if p3_enabled else 'p3_off'}"
    print(f"\\n===== CASE: {case_name} =====")

    args = dict(COMMON_ENGINE_ARGS)
    args.update(
        decode_window_enabled=p3_enabled,
        decode_window_auto_on_pressure=False,
        decode_window_recent_len=profile["decode_window_recent_len"],
        decode_window_check_interval=profile["decode_window_check_interval"],
        decode_window_min_trigger_tokens=profile["decode_window_min_trigger_tokens"],
        decode_window_min_drop_tokens=profile["decode_window_min_drop_tokens"],
    )

    engine = ManagedInferenceEngine(**args)
    # Make decode length stable for throughput comparison.
    engine.tokenizer.eos_token = None
    prompts = build_prompts(CONCURRENCY, PROMPT_REPEAT)

    t0 = time.perf_counter()
    outputs, metrics = engine.generate(prompts, return_metrics=True)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    metrics["profile"] = profile["name"]
    metrics["case"] = case_name
    metrics["p3_enabled"] = int(p3_enabled)
    metrics["wall_ms"] = round(wall_ms, 3)
    metrics["sample_output"] = outputs[0][:120]
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


def print_profile_summary(off: Dict, on: Dict):
    print(f"\\n----- PROFILE: {off['profile']} -----")
    print(
        f"tokens_per_sec: off={off['tokens_per_sec']:.4f} on={on['tokens_per_sec']:.4f} "
        f"delta={pct(off['tokens_per_sec'], on['tokens_per_sec']):+.2f}%"
    )
    print(
        f"decode_step_p95_ms: off={off['decode_step_p95_ms']:.4f} on={on['decode_step_p95_ms']:.4f} "
        f"delta={pct(off['decode_step_p95_ms'], on['decode_step_p95_ms']):+.2f}%"
    )
    print(
        f"wall_ms: off={off['wall_ms']:.4f} on={on['wall_ms']:.4f} "
        f"delta={pct(off['wall_ms'], on['wall_ms']):+.2f}%"
    )
    print(
        f"decode_window_prunes: off={int(off['decode_window_prunes'])} "
        f"on={int(on['decode_window_prunes'])}"
    )
    print(
        f"decode_window_tokens_dropped: off={int(off['decode_window_tokens_dropped'])} "
        f"on={int(on['decode_window_tokens_dropped'])}"
    )


def print_table(rows: List[Dict]):
    print("\\n===== FINAL TABLE =====")
    print(
        "profile,off_tps,on_tps,delta_tps_pct,"
        "off_p95_ms,on_p95_ms,delta_p95_pct,off_wall_ms,on_wall_ms,delta_wall_pct,on_prunes,on_drop_tokens"
    )
    for row in rows:
        print(
            f"{row['profile']},"
            f"{row['off_tps']:.4f},{row['on_tps']:.4f},{row['delta_tps_pct']:+.2f},"
            f"{row['off_p95_ms']:.4f},{row['on_p95_ms']:.4f},{row['delta_p95_pct']:+.2f},"
            f"{row['off_wall_ms']:.4f},{row['on_wall_ms']:.4f},{row['delta_wall_pct']:+.2f},"
            f"{row['on_prunes']},{row['on_drop_tokens']}"
        )


def main():
    table_rows = []
    for profile in PROFILES:
        off = run_case(profile, p3_enabled=False)
        on = run_case(profile, p3_enabled=True)
        print_profile_summary(off, on)
        table_rows.append(
            {
                "profile": profile["name"],
                "off_tps": float(off["tokens_per_sec"]),
                "on_tps": float(on["tokens_per_sec"]),
                "delta_tps_pct": pct(float(off["tokens_per_sec"]), float(on["tokens_per_sec"])),
                "off_p95_ms": float(off["decode_step_p95_ms"]),
                "on_p95_ms": float(on["decode_step_p95_ms"]),
                "delta_p95_pct": pct(float(off["decode_step_p95_ms"]), float(on["decode_step_p95_ms"])),
                "off_wall_ms": float(off["wall_ms"]),
                "on_wall_ms": float(on["wall_ms"]),
                "delta_wall_pct": pct(float(off["wall_ms"]), float(on["wall_ms"])),
                "on_prunes": int(on["decode_window_prunes"]),
                "on_drop_tokens": int(on["decode_window_tokens_dropped"]),
            }
        )

    print_table(table_rows)


if __name__ == "__main__":
    main()
