# Paper Experiments

This directory encodes the finalized paper experiment plan through:

- [benchmark_paper_suite.py](/C:/Users/mamengkui/OneDrive/文档/Playground/autodl_mirror/benchmarks/benchmark_paper_suite.py)
- [configs/paper_plan.py](/C:/Users/mamengkui/OneDrive/文档/Playground/autodl_mirror/benchmarks/configs/paper_plan.py)

Generate the finalized runbook:

```bash
python benchmark_paper_suite.py --phase qwen_main
```

Write the runbook to disk:

```bash
python benchmark_paper_suite.py --phase qwen_main --write results/paper/paper_runbook.md
```

## Frozen Single-Sequence Results

These rows are frozen for the main paper table and should not be expanded with more single-seq points:

| method | best `gpu_mem_frac` | `L_quality_max` | `tokens/s @ L_quality_max` | `p95 @ L_quality_max` |
|---|---:|---:|---:|---:|
| `off_raw` | `0.38` | `49000` | `0.4578` | `208.740 ms` |
| `off_compress` | `0.12` | `115000` | `0.1739` | `82.303 ms` |

Appendix keeps:

- pre-fix historical runs
- the repaired depth table

## Final Phase Order

1. Freeze single-sequence results
2. `32K serving memory-knob calibration`
3. Synthetic fixed-point main table
4. `32K concurrency frontier`
5. `32K passkey under load`
6. `ShareGPT / LongBench`

## Metric and Validity Definitions

Locked metric definitions:

- `TTFT`: request arrival to first token emission
- `ITL`: per-request average inter-token latency, then aggregate p95/p99 across requests
- `completion rate`: fraction of requests that terminate normally
- `valid completion rate`: fraction of requests that terminate normally and satisfy the phase-specific valid definition
- `OOM/failure count`: OOM, timeout, runtime failure, or other system-level failures
- `min free blocks`: minimum free KV blocks observed during the run
- `min free block ratio`: `min_free_blocks / N_total`, stored in internal logs only
- `wall-clock total runtime`: stored in internal logs only

Phase-specific `valid` definitions:

- Synthetic fixed-point / concurrency frontier / ShareGPT:
  - request completed normally and output format is legal
- Passkey under load:
  - request completed normally and the target answer field is parseable
- LongBench:
  - request completed normally and the task scoring path can score the sample

Status definitions:

- `Success`: `valid completion rate = 1.0` and no OOM/runtime failure
- `Degraded`: `completion rate > 0` but `valid completion rate < 1.0`, without catastrophic system failure
- `Failed/OOM`: `completion rate = 0`, or any run terminated by OOM/timeout/runtime failure such that the point is not stably serviceable

## Dry Run Policy

Before calibration, run two dry runs:

- `32K x 16 x 512`
- `8K x 64 x 1024`

Each dry run must confirm the system produces:

- `Status`
- `completion rate`
- `valid completion rate`
- `OOM/failure count`
- `TTFT/ITL` when available
- For the current Synthetic main table, latency columns are temporarily omitted until internal and vLLM latency fields are rerun under a unified TTFT/ITL schema.
- `min free blocks`
- `min free block ratio`
- `wall-clock total runtime`

## Calibration Policy

Calibration is fixed at:

- `input_len = 32768`
- `max_new_tokens = 512`
- `concurrency = 16`

Methods:

- `vLLM`
- `off_compress`
- `P2 mainline`

Selection rule:

1. filter out points with `OOM/failure count > 0`
2. maximize `valid completion rate`
3. maximize `completion rate`
4. maximize `tokens/s`
5. minimize `TTFT p99`

Repeat policy:

- rough sweep: `repeat = 2`
- top-2 confirm: add `1` extra repeat to each top candidate

## Frozen 32K Formal Knobs

Formal phases from Synthetic fixed-point onward use frozen knobs and must not retune per point:

- internal methods:
  - `gpu_mem_frac = 0.20`
  - `prefill_batch_size = 4`
  - `chunk_size = 512`
- `vLLM`:
  - `gpu_memory_utilization = 0.90`

These values are backed by:

- corrected 32K calibration
- internal knob tuning
- fresh verify at `0.20 / 4 / 512` for both `off_compress` and `P2 mainline`

## Current Runner Coverage

Implemented in-repo runners:

- [benchmark_exp1_capacity_table.py](/C:/Users/mamengkui/OneDrive/文档/Playground/autodl_mirror/benchmarks/benchmark_exp1_capacity_table.py)
  - synthetic and fixed-point style internal runs
  - internal concurrency sweeps
  - group-specific mem-frac maps
  - derived status, completion, and memory fields
- [benchmark_passkey_under_load.py](/C:/Users/mamengkui/OneDrive/文档/Playground/autodl_mirror/benchmarks/benchmark_passkey_under_load.py)
  - internal 32K passkey under load for `off_compress` and `P2 mainline`
- [benchmark_vllm_dry_run.py](/C:/Users/mamengkui/OneDrive/文档/Playground/autodl_mirror/benchmarks/benchmark_vllm_dry_run.py)
  - vLLM dry-run point using the OpenAI-compatible server path
- [benchmark_vllm_calibration.py](/C:/Users/mamengkui/OneDrive/文档/Playground/autodl_mirror/benchmarks/benchmark_vllm_calibration.py)
  - single calibration point for vLLM
- [benchmark_vllm_fixedpoint.py](/C:/Users/mamengkui/OneDrive/文档/Playground/autodl_mirror/benchmarks/benchmark_vllm_fixedpoint.py)
  - single fixed-point synthetic evaluation for vLLM
- [benchmark_vllm_frontier.py](/C:/Users/mamengkui/OneDrive/文档/Playground/autodl_mirror/benchmarks/benchmark_vllm_frontier.py)
  - 32K concurrency frontier for vLLM
- [benchmark_vllm_passkey_under_load.py](/C:/Users/mamengkui/OneDrive/文档/Playground/autodl_mirror/benchmarks/benchmark_vllm_passkey_under_load.py)
  - 32K passkey under load for vLLM
- [benchmark_sharegpt_serving.py](/C:/Users/mamengkui/OneDrive/文档/Playground/autodl_mirror/benchmarks/benchmark_sharegpt_serving.py)
  - ShareGPT serving benchmark for internal methods and vLLM
- [benchmark_longbench_concurrency.py](/C:/Users/mamengkui/OneDrive/文档/Playground/autodl_mirror/benchmarks/benchmark_longbench_concurrency.py)
  - LongBench sanity and full runs for internal methods and vLLM

Known limitations:

- vLLM runners do not expose KV block pool metrics, so `min_free_blocks` and `min_free_block_ratio` are set to `-1`
- LongBench scoring currently uses built-in normalized exact-match and token-F1 proxies for the selected tasks
- the internal synthetic/frontier/passkey runners expose TTFT/ITL as engine-side practical proxies; compare latency columns only after the dedicated latency rerun is complete
