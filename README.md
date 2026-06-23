# UnifiedKV / BP-KV

This repository contains the research prototype used for BP-KV, a fixed-budget
block-paged KV-cache runtime for long-context LLM inference under memory
constraints. The code is organized around:

- `core/`: block-paged KV pool, scheduler, compression/writeback path,
  offload/prefetch support, and Page16 direct decoding backend.
- `benchmarks/`: LongBench, Passkey, ShareGPT online serving, offline
  fixed-batch capacity, and ablation runners used in the paper.
- `reproducibility/`: public workload manifests and notes for reconstructing
  fixed evaluation subsets without committing raw datasets or model weights.

## Scope

The repository does not include model weights, raw ShareGPT/LongBench data, or
large experiment outputs. Download the corresponding public datasets and models
separately, then use the scripts under `benchmarks/` with the released manifests
under `reproducibility/`.

## Environment

The experiments were developed for a single NVIDIA RTX 4090 24GB GPU with
PyTorch, Hugging Face Transformers, Triton, FlashAttention, and optional vLLM
baselines. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Some packages, especially `flash-attn` and `vllm`, may require CUDA- and
PyTorch-compatible wheels. Use versions compatible with your local CUDA/PyTorch
stack.

## Workloads

ShareGPT workload manifests are stored in `reproducibility/sharegpt/`. The
manifests contain record indices, turn indices, token lengths, generation limits,
and request order. They intentionally omit prompt text; reconstruct requests from
the original ShareGPT corpus with the same preprocessing script and tokenizer.

## Notes

This is a research prototype rather than a packaged serving engine. The code is
intended to reproduce the paper's experiments and to expose the BP-KV runtime
mechanisms for inspection.
