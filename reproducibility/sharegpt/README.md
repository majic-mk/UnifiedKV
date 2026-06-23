# ShareGPT workload manifests

These files describe the fixed ShareGPT workloads used by the BP-KV experiments
without storing raw prompt text.

- `sharegpt_clean292_llama_manifest.json`: Llama online continuous-refill
  workload, selected from the original ShareGPT corpus with long-input and
  long-output filters.
- `sharegpt_clean292_qwen_manifest.json`: Qwen online continuous-refill
  workload generated with the Qwen tokenizer and non-reasoning inference setting.
- `sharegpt_small64_manifest.json`: independently sampled fixed 64-request
  workload for block-paged KV management and direct-decoding ablations.

Each manifest stores source `record_idx`, `turn_idx`, token lengths, generation
limits, and request order. Reconstruct the prompt text from the original
ShareGPT corpus and the same tokenizer/preprocessing code; do not compare
methods on regenerated random samples.
