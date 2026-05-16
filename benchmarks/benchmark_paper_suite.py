import argparse
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List


BENCH_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BENCH_DIR / "configs"
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

from paper_plan import (  # noqa: E402
    APPENDIX_TABLES,
    CALIBRATION_32K,
    CONCURRENCY_FRONTIER_32K,
    DRY_RUN_POINTS,
    FROZEN_SINGLE_SEQ_ROWS,
    LONGBENCH,
    METRIC_DEFINITIONS,
    P2_MAINLINE,
    PASSKEY_UNDER_LOAD_32K,
    PHASE_ORDER,
    QWEN_MODEL_PATH,
    SHAREGPT_SERVING,
    STATUS_DEFINITIONS,
    SYNTHETIC_FIXED_POINTS,
    SYNTHETIC_MAIN_TABLE_FIELDS,
    VALID_DEFINITIONS,
)


@dataclass
class CommandSpec:
    title: str
    argv: List[str]
    ready: bool = True
    notes: List[str] = field(default_factory=list)

    def shell(self) -> str:
        return shlex.join(self.argv)


@dataclass
class PhaseSpec:
    name: str
    title: str
    summary: str
    commands_fn: Callable[[str], List[CommandSpec]]
    outputs: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _results_dir(name: str) -> Path:
    return BENCH_DIR / "results" / "paper" / name


def _py(python_bin: str, script: str, *args: str) -> List[str]:
    return [python_bin, str(BENCH_DIR / script), *[str(x) for x in args]]


def _fmt_table(rows: List[Dict[str, object]]) -> List[str]:
    lines = [
        "| method | best `gpu_mem_frac` | `L_quality_max` | `tokens/s @ L_quality_max` | `p95 @ L_quality_max` |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['method']}` | `{row['best_gpu_mem_frac']}` | `{row['L_quality_max']}` | "
            f"`{row['tokens_s_at_L_quality_max']}` | `{row['p95_ms_at_L_quality_max']}` |"
        )
    return lines


def _group_frac_placeholder(input_len: int) -> str:
    return ",".join(
        [
            f"off_compress@{int(input_len)}:<off_compress_frac>",
            f"p2_only_compress@{int(input_len)}:<p2_frac>",
        ]
    )


def _single_seq_frozen_commands(_: str) -> List[CommandSpec]:
    return []


def _dry_run_commands(python_bin: str) -> List[CommandSpec]:
    commands: List[CommandSpec] = []
    out_dir = _results_dir("dry_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    for point in DRY_RUN_POINTS:
        point_name = f"{int(point['input_len'])}_c{int(point['concurrency'])}_n{int(point['max_new_tokens'])}"
        commands.append(
            CommandSpec(
                title=f"Dry run internal methods @ {point_name}",
                argv=_py(
                    python_bin,
                    "benchmark_exp1_capacity_table.py",
                    "--mode",
                    "table",
                    "--model-name",
                    QWEN_MODEL_PATH,
                    "--groups",
                    "off_compress,p2_only_compress",
                    "--input-lengths",
                    str(point["input_len"]),
                    "--concurrency-list",
                    str(point["concurrency"]),
                    "--max-new-tokens",
                    str(point["max_new_tokens"]),
                    "--gpu-mem-frac-map",
                    _group_frac_placeholder(int(point["input_len"])),
                    "--repeats",
                    "1",
                    "--out-prefix",
                    str(out_dir / f"internal_{point_name}"),
                ),
                ready=False,
                notes=[
                    "Fill <off_compress_frac> and <p2_frac> with conservative starting values before running.",
                    "Dry run must confirm Status, completion, valid, OOM, min-free, and wall-clock fields are populated.",
                ],
            )
        )
        commands.append(
            CommandSpec(
                title=f"Dry run vLLM @ {point_name}",
                argv=[
                    "python",
                    str(BENCH_DIR / "benchmark_vllm_dry_run.py"),
                    "--model-name",
                    QWEN_MODEL_PATH,
                    "--input-len",
                    str(point["input_len"]),
                    "--concurrency",
                    str(point["concurrency"]),
                    "--max-new-tokens",
                    str(point["max_new_tokens"]),
                    "--gpu-memory-utilization",
                    "0.85",
                    "--out",
                    str(out_dir / f"vllm_{point_name}.json"),
                ],
                notes=["Adjust gpu-memory-utilization if you want a different pre-calibration warm start."],
            )
        )
    return commands


def _calibration_commands(python_bin: str) -> List[CommandSpec]:
    out_dir = _results_dir("calibration32k_qwen")
    out_dir.mkdir(parents=True, exist_ok=True)
    commands: List[CommandSpec] = []
    for method in ("off_compress", "p2_only_compress"):
        label = CALIBRATION_32K["method_labels"][method]
        for frac in CALIBRATION_32K["gpu_mem_frac_grid"]:
            commands.append(
                CommandSpec(
                    title=f"Calibration rough sweep / {label} / frac={frac:.2f}",
                    argv=_py(
                        python_bin,
                        "benchmark_exp1_capacity_table.py",
                        "--mode",
                        "table",
                        "--model-name",
                        QWEN_MODEL_PATH,
                        "--groups",
                        method,
                        "--input-lengths",
                        str(CALIBRATION_32K["input_len"]),
                        "--concurrency-list",
                        str(CALIBRATION_32K["concurrency"]),
                        "--max-new-tokens",
                        str(CALIBRATION_32K["max_new_tokens"]),
                        "--gpu-mem-frac-map",
                        f"{method}@{int(CALIBRATION_32K['input_len'])}:{float(frac):.2f}",
                        "--repeats",
                        str(CALIBRATION_32K["rough_repeats"]),
                        "--worker-fixed-gpu-mem-frac",
                        "--out-prefix",
                        str(out_dir / f"{method}_frac_{float(frac):.2f}"),
                    ),
                    notes=[
                        "Selection rule: OOM/failure count = 0 first, then valid completion, completion, tokens/s, and TTFT p99.",
                        "TTFT and ITL still need dedicated instrumentation; use this command now for status, completion, and memory calibration.",
                    ],
                )
            )
        commands.append(
            CommandSpec(
                title=f"Calibration top-2 confirm / {label}",
                argv=_py(
                    python_bin,
                    "benchmark_exp1_capacity_table.py",
                    "--mode",
                    "table",
                    "--model-name",
                    QWEN_MODEL_PATH,
                    "--groups",
                    method,
                    "--input-lengths",
                    str(CALIBRATION_32K["input_len"]),
                    "--concurrency-list",
                    str(CALIBRATION_32K["concurrency"]),
                    "--max-new-tokens",
                    str(CALIBRATION_32K["max_new_tokens"]),
                    "--gpu-mem-frac-map",
                    f"{method}@{int(CALIBRATION_32K['input_len'])}:<top2_candidate_frac>",
                    "--repeats",
                    str(CALIBRATION_32K["rough_repeats"] + CALIBRATION_32K["confirm_extra_repeats"]),
                    "--worker-fixed-gpu-mem-frac",
                    "--out-prefix",
                    str(out_dir / f"{method}_top2_confirm"),
                ),
                ready=False,
                notes=["Replace <top2_candidate_frac> with each of the two best rough-sweep candidates and run once per candidate."],
            )
        )
    for util in CALIBRATION_32K["vllm_gpu_memory_utilization_grid"]:
        commands.append(
            CommandSpec(
                title=f"Calibration rough sweep / vLLM / gpu_memory_utilization={util:.2f}",
                argv=[
                    "python",
                    str(BENCH_DIR / "benchmark_vllm_calibration.py"),
                    "--model-name",
                    QWEN_MODEL_PATH,
                    "--input-len",
                    str(CALIBRATION_32K["input_len"]),
                    "--concurrency",
                    str(CALIBRATION_32K["concurrency"]),
                    "--max-new-tokens",
                    str(CALIBRATION_32K["max_new_tokens"]),
                    "--gpu-memory-utilization",
                    f"{float(util):.2f}",
                    "--repeats",
                    str(CALIBRATION_32K["rough_repeats"]),
                    "--out",
                    str(out_dir / f"vllm_util_{float(util):.2f}.json"),
                ],
                notes=["This now uses the OpenAI-compatible vLLM server path and records TTFT/ITL from streamed responses."],
            )
        )
    return commands


def _synthetic_commands(python_bin: str) -> List[CommandSpec]:
    out_dir = _results_dir("synthetic_qwen")
    out_dir.mkdir(parents=True, exist_ok=True)
    commands: List[CommandSpec] = []
    for point in SYNTHETIC_FIXED_POINTS:
        commands.append(
            CommandSpec(
                title=f"Synthetic fixed-point internal methods @ {point['name']}",
                argv=_py(
                    python_bin,
                    "benchmark_exp1_capacity_table.py",
                    "--mode",
                    "table",
                    "--model-name",
                    QWEN_MODEL_PATH,
                    "--groups",
                    "off_compress,p2_only_compress",
                    "--input-lengths",
                    str(point["input_len"]),
                    "--concurrency-list",
                    str(point["concurrency"]),
                    "--max-new-tokens",
                    str(point["max_new_tokens"]),
                    "--gpu-mem-frac-map",
                    _group_frac_placeholder(int(point["input_len"])),
                    "--repeats",
                    "3",
                    "--worker-fixed-gpu-mem-frac",
                    "--out-prefix",
                    str(out_dir / point["name"]),
                ),
                ready=False,
                notes=["Fill calibrated mem-fracs before running this point."],
            )
        )
        commands.append(
            CommandSpec(
                title=f"Synthetic fixed-point vLLM @ {point['name']}",
                argv=[
                    "python",
                    str(BENCH_DIR / "benchmark_vllm_fixedpoint.py"),
                    "--model-name",
                    QWEN_MODEL_PATH,
                    "--input-len",
                    str(point["input_len"]),
                    "--concurrency",
                    str(point["concurrency"]),
                    "--max-new-tokens",
                    str(point["max_new_tokens"]),
                    "--gpu-memory-utilization",
                    "<vllm_util>",
                    "--repeats",
                    "3",
                    "--out",
                    str(out_dir / f"vllm_{point['name']}.json"),
                ],
                ready=False,
                notes=["Fill <vllm_util> with the calibration winner before running."],
            )
        )
    return commands


def _frontier_commands(python_bin: str) -> List[CommandSpec]:
    out_dir = _results_dir("frontier32k_qwen")
    out_dir.mkdir(parents=True, exist_ok=True)
    concurrency_csv = ",".join(str(x) for x in CONCURRENCY_FRONTIER_32K["concurrency_list"])
    return [
        CommandSpec(
            title="32K concurrency frontier / internal methods",
            argv=_py(
                python_bin,
                "benchmark_exp1_capacity_table.py",
                "--mode",
                "table",
                "--model-name",
                QWEN_MODEL_PATH,
                "--groups",
                "off_compress,p2_only_compress",
                "--input-lengths",
                str(CONCURRENCY_FRONTIER_32K["input_len"]),
                "--concurrency-list",
                concurrency_csv,
                "--max-new-tokens",
                str(CONCURRENCY_FRONTIER_32K["max_new_tokens"]),
                "--gpu-mem-frac-map",
                _group_frac_placeholder(int(CONCURRENCY_FRONTIER_32K["input_len"])),
                "--repeats",
                str(CONCURRENCY_FRONTIER_32K["repeats"]),
                "--worker-fixed-gpu-mem-frac",
                "--out-prefix",
                str(out_dir / "internal"),
            ),
            ready=False,
            notes=["Fill calibrated 32K mem-fracs. Use status, valid_completion_rate, and frontier_reason to determine the stable frontier."],
        ),
        CommandSpec(
            title="32K concurrency frontier / vLLM",
            argv=[
                "python",
                str(BENCH_DIR / "benchmark_vllm_frontier.py"),
                "--model-name",
                QWEN_MODEL_PATH,
                "--input-len",
                str(CONCURRENCY_FRONTIER_32K["input_len"]),
                "--concurrency-list",
                concurrency_csv,
                "--max-new-tokens",
                str(CONCURRENCY_FRONTIER_32K["max_new_tokens"]),
                "--gpu-memory-utilization",
                "<vllm_util>",
                "--repeats",
                str(CONCURRENCY_FRONTIER_32K["repeats"]),
                "--out",
                str(out_dir / "vllm_frontier.json"),
            ],
            ready=False,
            notes=["Fill <vllm_util> with the calibration winner before running."],
        ),
    ]


def _passkey_commands(_: str) -> List[CommandSpec]:
    out_dir = _results_dir("passkey32k_load_qwen")
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        CommandSpec(
            title="32K passkey under load / internal methods",
            argv=[
                "python",
                str(BENCH_DIR / "benchmark_passkey_under_load.py"),
                "--model-name",
                QWEN_MODEL_PATH,
                "--input-len",
                str(PASSKEY_UNDER_LOAD_32K["input_len"]),
                "--max-new-tokens",
                str(PASSKEY_UNDER_LOAD_32K["max_new_tokens"]),
                "--depths",
                ",".join(str(x) for x in PASSKEY_UNDER_LOAD_32K["depths"]),
                "--keys-per-depth",
                str(PASSKEY_UNDER_LOAD_32K["keys_per_depth"]),
                "--concurrency-list",
                ",".join(str(x) for x in PASSKEY_UNDER_LOAD_32K["concurrency_list"]),
                "--methods",
                "off_compress,p2_only_compress",
                "--gpu-mem-frac-map",
                "off_compress:<off_compress_frac>,p2_only_compress:<p2_frac>",
                "--out",
                str(out_dir / "internal_passkey_under_load.json"),
            ],
            ready=False,
            notes=["Fill calibrated 32K mem-fracs. Report in the fixed order: completion rate, valid rate, EM."],
        ),
        CommandSpec(
            title="32K passkey under load / vLLM",
            argv=[
                "python",
                str(BENCH_DIR / "benchmark_vllm_passkey_under_load.py"),
                "--model-name",
                QWEN_MODEL_PATH,
                "--input-len",
                str(PASSKEY_UNDER_LOAD_32K["input_len"]),
                "--max-new-tokens",
                str(PASSKEY_UNDER_LOAD_32K["max_new_tokens"]),
                "--depths",
                ",".join(str(x) for x in PASSKEY_UNDER_LOAD_32K["depths"]),
                "--keys-per-depth",
                str(PASSKEY_UNDER_LOAD_32K["keys_per_depth"]),
                "--concurrency-list",
                ",".join(str(x) for x in PASSKEY_UNDER_LOAD_32K["concurrency_list"]),
                "--gpu-memory-utilization",
                "<vllm_util>",
                "--out",
                str(out_dir / "vllm_passkey_under_load.json"),
            ],
            ready=False,
            notes=["Fill <vllm_util> with the calibration winner before running."],
        ),
    ]


def _sharegpt_commands(_: str) -> List[CommandSpec]:
    out_dir = _results_dir("sharegpt_qwen")
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        CommandSpec(
            title="ShareGPT serving / internal methods",
            argv=[
                "python",
                str(BENCH_DIR / "benchmark_sharegpt_serving.py"),
                "--model-name",
                QWEN_MODEL_PATH,
                "--dataset",
                str(SHAREGPT_SERVING["dataset_name"]),
                "--methods",
                "off_compress,p2_only_compress",
                "--sample-count",
                str(SHAREGPT_SERVING["sample_count"]),
                "--concurrency-list",
                ",".join(str(x) for x in SHAREGPT_SERVING["concurrency_list"]),
                "--prompt-len-min",
                str(SHAREGPT_SERVING["prompt_len_range"][0]),
                "--prompt-len-max",
                str(SHAREGPT_SERVING["prompt_len_range"][1]),
                "--target-len-min",
                str(SHAREGPT_SERVING["target_len_clip"][0]),
                "--target-len-max",
                str(SHAREGPT_SERVING["target_len_clip"][1]),
                "--repeats",
                str(SHAREGPT_SERVING["repeats"]),
                "--gpu-mem-frac-map",
                "off_compress:<off_compress_frac>,p2_only_compress:<p2_frac>",
                "--out",
                str(out_dir / "internal_sharegpt.json"),
            ],
            ready=False,
            notes=["Fill calibrated mem-fracs before running."],
        ),
        CommandSpec(
            title="ShareGPT serving / vLLM",
            argv=[
                "python",
                str(BENCH_DIR / "benchmark_sharegpt_serving.py"),
                "--model-name",
                QWEN_MODEL_PATH,
                "--dataset",
                str(SHAREGPT_SERVING["dataset_name"]),
                "--methods",
                "vllm",
                "--sample-count",
                str(SHAREGPT_SERVING["sample_count"]),
                "--concurrency-list",
                ",".join(str(x) for x in SHAREGPT_SERVING["concurrency_list"]),
                "--prompt-len-min",
                str(SHAREGPT_SERVING["prompt_len_range"][0]),
                "--prompt-len-max",
                str(SHAREGPT_SERVING["prompt_len_range"][1]),
                "--target-len-min",
                str(SHAREGPT_SERVING["target_len_clip"][0]),
                "--target-len-max",
                str(SHAREGPT_SERVING["target_len_clip"][1]),
                "--repeats",
                str(SHAREGPT_SERVING["repeats"]),
                "--vllm-gpu-memory-utilization",
                "<vllm_util>",
                "--out",
                str(out_dir / "vllm_sharegpt.json"),
            ],
            ready=False,
            notes=["Fill <vllm_util> with the calibration winner before running."],
        ),
    ]


def _longbench_commands(_: str) -> List[CommandSpec]:
    out_dir = _results_dir("longbench_qwen")
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        CommandSpec(
            title="LongBench sanity subset / internal methods",
            argv=[
                "python",
                str(BENCH_DIR / "benchmark_longbench_concurrency.py"),
                "--model-name",
                QWEN_MODEL_PATH,
                "--tasks",
                ",".join(str(x) for x in LONGBENCH["tasks"]),
                "--samples-per-task",
                str(LONGBENCH["sanity_samples_per_task"]),
                "--max-prompt-tokens",
                str(LONGBENCH["max_prompt_tokens"]),
                "--max-new-tokens",
                str(LONGBENCH["max_new_tokens"]),
                "--concurrency-list",
                ",".join(str(x) for x in LONGBENCH["concurrency_list"]),
                "--methods",
                "off_compress,p2_only_compress",
                "--gpu-mem-frac-map",
                "off_compress:<off_compress_frac>,p2_only_compress:<p2_frac>",
                "--out",
                str(out_dir / "internal_sanity.json"),
            ],
            ready=False,
            notes=["Run this first to verify task loading, scoring, and score@16 - score@1 logic before the full LongBench run."],
        ),
        CommandSpec(
            title="LongBench sanity subset / vLLM",
            argv=[
                "python",
                str(BENCH_DIR / "benchmark_longbench_concurrency.py"),
                "--model-name",
                QWEN_MODEL_PATH,
                "--tasks",
                ",".join(str(x) for x in LONGBENCH["tasks"]),
                "--samples-per-task",
                str(LONGBENCH["sanity_samples_per_task"]),
                "--max-prompt-tokens",
                str(LONGBENCH["max_prompt_tokens"]),
                "--max-new-tokens",
                str(LONGBENCH["max_new_tokens"]),
                "--concurrency-list",
                ",".join(str(x) for x in LONGBENCH["concurrency_list"]),
                "--methods",
                "vllm",
                "--vllm-gpu-memory-utilization",
                "<vllm_util>",
                "--out",
                str(out_dir / "vllm_sanity.json"),
            ],
            ready=False,
            notes=["Run this first to verify task loading, scoring, and score@16 - score@1 logic before the full LongBench run."],
        ),
        CommandSpec(
            title="LongBench full run / internal methods",
            argv=[
                "python",
                str(BENCH_DIR / "benchmark_longbench_concurrency.py"),
                "--model-name",
                QWEN_MODEL_PATH,
                "--tasks",
                ",".join(str(x) for x in LONGBENCH["tasks"]),
                "--samples-per-task",
                str(LONGBENCH["full_samples_per_task"]),
                "--max-prompt-tokens",
                str(LONGBENCH["max_prompt_tokens"]),
                "--max-new-tokens",
                str(LONGBENCH["max_new_tokens"]),
                "--concurrency-list",
                ",".join(str(x) for x in LONGBENCH["concurrency_list"]),
                "--methods",
                "off_compress,p2_only_compress",
                "--gpu-mem-frac-map",
                "off_compress:<off_compress_frac>,p2_only_compress:<p2_frac>",
                "--out",
                str(out_dir / "internal_full.json"),
            ],
            ready=False,
            notes=["Only run after the sanity subset is confirmed clean."],
        ),
        CommandSpec(
            title="LongBench full run / vLLM",
            argv=[
                "python",
                str(BENCH_DIR / "benchmark_longbench_concurrency.py"),
                "--model-name",
                QWEN_MODEL_PATH,
                "--tasks",
                ",".join(str(x) for x in LONGBENCH["tasks"]),
                "--samples-per-task",
                str(LONGBENCH["full_samples_per_task"]),
                "--max-prompt-tokens",
                str(LONGBENCH["max_prompt_tokens"]),
                "--max-new-tokens",
                str(LONGBENCH["max_new_tokens"]),
                "--concurrency-list",
                ",".join(str(x) for x in LONGBENCH["concurrency_list"]),
                "--methods",
                "vllm",
                "--vllm-gpu-memory-utilization",
                "<vllm_util>",
                "--out",
                str(out_dir / "vllm_full.json"),
            ],
            ready=False,
            notes=["Only run after the sanity subset is confirmed clean."],
        ),
    ]


PHASES: Dict[str, PhaseSpec] = {
    "single_seq_frozen": PhaseSpec(
        name="single_seq_frozen",
        title="Phase 1 / Frozen Single-Sequence Results",
        summary="Single-sequence longest-passkey results are frozen and no new single-seq points should be added to the main paper table.",
        commands_fn=_single_seq_frozen_commands,
        outputs=[
            "/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/results/paper/off_raw_rerun_fixed/off_raw_memfrac_0.38.json",
            "/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/results/paper/single_seq_qwen_neighbors_rerun_fixed/off_compress_memfrac_0.12_stepdown.json",
        ],
        notes=[
            "The main paper table should contain exactly two rows: off_raw and off_compress.",
            "Appendix keeps the pre-fix historical runs and the repaired depth table.",
        ],
    ),
    "dry_run": PhaseSpec(
        name="dry_run",
        title="Preflight / Dry Runs",
        summary="Dry runs for 32K x 16 x 512 and 8K x 64 x 1024 before calibration.",
        commands_fn=_dry_run_commands,
        outputs=[str(_results_dir("dry_run"))],
    ),
    "calibration32k_qwen": PhaseSpec(
        name="calibration32k_qwen",
        title="Phase 2 / 32K Serving Memory-Knob Calibration",
        summary="Calibrate vLLM, off_compress, and P2 mainline for 32K serving.",
        commands_fn=_calibration_commands,
        outputs=[str(_results_dir("calibration32k_qwen"))],
        notes=[
            "Rough sweep uses repeat=2; the top-2 candidates each receive one extra confirmation run.",
            "Selection rule: filter OOM/failure count > 0 first, then compare valid completion, completion, tokens/s, and TTFT p99.",
        ],
    ),
    "synthetic_qwen": PhaseSpec(
        name="synthetic_qwen",
        title="Phase 3 / Synthetic Fixed-Point Main Table",
        summary="Three fixed points: 8K x 64 x 1024, 16K x 32 x 1024, and 32K x 16 x 512.",
        commands_fn=_synthetic_commands,
        outputs=[str(_results_dir("synthetic_qwen"))],
    ),
    "frontier32k_qwen": PhaseSpec(
        name="frontier32k_qwen",
        title="Phase 4 / 32K Concurrency Frontier",
        summary="Concurrency sweep at 32K: 1, 2, 4, 8, 12, 16, 24, 32.",
        commands_fn=_frontier_commands,
        outputs=[str(_results_dir("frontier32k_qwen"))],
    ),
    "passkey32k_load_qwen": PhaseSpec(
        name="passkey32k_load_qwen",
        title="Phase 5 / 32K Passkey Under Load",
        summary="Passkey quality under concurrency with reporting order: completion rate, valid rate, EM.",
        commands_fn=_passkey_commands,
        outputs=[str(_results_dir("passkey32k_load_qwen"))],
    ),
    "sharegpt_qwen": PhaseSpec(
        name="sharegpt_qwen",
        title="Phase 6A / ShareGPT Serving",
        summary="Real-workload serving benchmark.",
        commands_fn=_sharegpt_commands,
        outputs=[str(_results_dir("sharegpt_qwen"))],
    ),
    "longbench_qwen": PhaseSpec(
        name="longbench_qwen",
        title="Phase 6B / LongBench",
        summary="LongBench sanity subset first, then the full run.",
        commands_fn=_longbench_commands,
        outputs=[str(_results_dir("longbench_qwen"))],
    ),
}


def _phase_sequence(name: str) -> List[str]:
    if name in PHASES:
        return [name]
    if name in ("qwen_main", "all"):
        return list(PHASE_ORDER)
    raise KeyError(name)


def _render_phase(phase: PhaseSpec, python_bin: str) -> List[str]:
    lines = [f"## {phase.title}", "", phase.summary, ""]
    if phase.notes:
        lines.append("Notes:")
        for note in phase.notes:
            lines.append(f"- {note}")
        lines.append("")
    if phase.name == "single_seq_frozen":
        lines.extend(_fmt_table(FROZEN_SINGLE_SEQ_ROWS))
        lines.append("")
        lines.append("Appendix-only tables:")
        for item in APPENDIX_TABLES:
            lines.append(f"- `{item['title']}`: {item['purpose']}")
        lines.append("")
    commands = phase.commands_fn(python_bin)
    if commands:
        lines.append("Commands:")
        for idx, cmd in enumerate(commands, start=1):
            status = "READY" if cmd.ready else "MANUAL/PENDING"
            lines.append(f"{idx}. `{status}` {cmd.title}")
            lines.append("")
            lines.append("```bash")
            lines.append(cmd.shell())
            lines.append("```")
            if cmd.notes:
                for note in cmd.notes:
                    lines.append(f"- {note}")
            lines.append("")
    else:
        lines.append("Commands: none. This phase is already frozen.")
        lines.append("")
    if phase.outputs:
        lines.append("Outputs:")
        for out in phase.outputs:
            lines.append(f"- `{out}`")
        lines.append("")
    return lines


def render_runbook(phase_names: List[str], python_bin: str) -> str:
    lines = [
        "# Paper Experiments Runbook",
        "",
        "This runbook reflects the finalized paper plan and the frozen single-sequence results.",
        "",
        "## Frozen Single-Sequence Results",
        "",
        *_fmt_table(FROZEN_SINGLE_SEQ_ROWS),
        "",
        "## Status Definitions",
        "",
    ]
    for key, value in STATUS_DEFINITIONS.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Valid Definitions", ""])
    for key, value in VALID_DEFINITIONS.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Metric Definitions", ""])
    for key, value in METRIC_DEFINITIONS.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Synthetic Main Table Fields")
    lines.append("")
    for field in SYNTHETIC_MAIN_TABLE_FIELDS:
        lines.append(f"- `{field}`")
    lines.append("")
    lines.append("## P2 Mainline")
    lines.append("")
    lines.append(f"- group: `{P2_MAINLINE['group']}`")
    lines.append(f"- variant: `{P2_MAINLINE['variant']}`")
    lines.append(
        f"- config: `retain={P2_MAINLINE['retain_ratio']}`, `budget_max={P2_MAINLINE['offload_budget_blocks_max']}`, "
        f"`chunk={P2_MAINLINE['chunk_size']}`, `sink={P2_MAINLINE['sink_len']}`, `recent={P2_MAINLINE['p2_recent_tokens']}`"
    )
    lines.append("")
    for phase_name in phase_names:
        lines.extend(_render_phase(PHASES[phase_name], python_bin))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the finalized paper experiment runbook.")
    parser.add_argument("--phase", type=str, default="qwen_main")
    parser.add_argument("--python-bin", type=str, default="python")
    parser.add_argument("--write", type=str, default="")
    args = parser.parse_args()

    phase_names = _phase_sequence(str(args.phase).strip())
    text = render_runbook(phase_names, str(args.python_bin))
    if str(args.write).strip():
        out_path = Path(args.write)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote runbook to {out_path}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
