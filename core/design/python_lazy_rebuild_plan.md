# Python-only Lazy Per-layer Rebuild Plan

## Goal
Reduce decode-time temporary GPU memory in the compatibility rebuild path without adding C++/CUDA paged-attention kernels.

This is deliberately a Python-layer compatibility path. The paper story remains:

- the system advantage comes from block-managed KV plus paged decode/offload policy;
- the current full `DynamicCache` rebuild is a conservative HF-compatible fallback;
- any lazy rebuild optimization must preserve the non-invasive middleware premise.

## Two Hard Risks To Validate First

### Risk 1: HF `Cache` semantics may not support layer-local materialization

Before implementing a lazy cache, verify the exact model implementation being used.

For Llama in the current environment, the required behavior is:

1. each decoder layer calls `past_key_values.update(key_states, value_states, layer_idx, cache_kwargs)`;
2. the attention layer immediately consumes the `(K_full, V_full)` returned by that same `update()` call;
3. the forward pass does not require old full-KV tensors for previous layers to remain stored in the cache object;
4. the model returns the same cache object at the end, but does not need to iterate a complete tuple-style cache after forward;
5. post-forward block-pool writeback can be driven from the newly produced per-layer token KV, not from a permanently stored all-layer full cache.

A very small prototype must pass before touching the formal runner:

```bash
/root/miniconda3/bin/python benchmarks/probes/probe_llama_cache_update_chain.py \
  --json-out benchmarks/results/probes/llama_cache_update_chain.json
```

The prototype must fail loudly if Llama tries to read cross-layer full cache state through `__getitem__` or iteration.

### Risk 2: per-layer rebuild can still create a large single-layer memory spike

Lazy per-layer rebuild only removes the all-layer peak. It is not enough if each layer still does:

- `index_select` into temporary pages;
- `permute/reshape` temporary views/copies;
- per-sequence `pad` tensors;
- `torch.cat(Ks/Vs)`;
- `DynamicCache.update()`, which can itself concatenate old and new cache tensors.

Therefore the lazy path must also remove two internal amplification points.

## Required Sub-optimizations

### A. Do not call `DynamicCache.update(cat(...))` for rebuilt past KV

The lazy cache should return the materialized full K/V for the current layer directly from its own `update()` implementation.

The returned tensors should be the tensors consumed by Llama attention for that layer. They should not be inserted into a `DynamicCache` that keeps full-KV for all layers.

### B. Prefer preallocation plus direct fill over list/pad/cat

For each materialized layer, allocate final tensors once:

```python
K_full = empty(batch, kv_heads, max_seq_len + new_tokens, head_dim)
V_full = empty(batch, kv_heads, max_seq_len + new_tokens, head_dim)
```

Then fill each request slice directly from block pages into the final destination, leaving padded tail positions unused/masked. Avoid building `Ks`, `Vs`, padded copies, and final `cat`.

## Proposed Implementation Stages

### Stage 0: Safety fallback already needed

Catch `torch.cuda.OutOfMemoryError` in the current rebuild path and split the decode microbatch. This is a low-risk guardrail and does not change model outputs.

### Stage 1: Cache API prototype

Run `benchmarks/probes/probe_llama_cache_update_chain.py` on the installed Transformers version.

Pass criteria:

- `update_order` equals `[0, 1, ..., num_layers-1]` for a single decode forward;
- `cross_layer_reads == 0`;
- `iter_reads == 0`;
- returned `past_key_values` is the same lazy cache object;
- logits are produced successfully.

If this fails, stop the lazy-cache approach and keep only microbatch/OOM fallback.

### Stage 2: Python `LazyPagedRebuildCache`

Implement a cache object that:

- subclasses or conforms to `transformers.cache_utils.Cache`;
- receives `seq_ids`, scheduler, pool, `seq_lens`, and `logical_seq_lens`;
- implements `get_seq_length()` for mask construction;
- implements `update()` to materialize only `layer_idx`;
- returns `(K_full_with_current, V_full_with_current)` for the current layer;
- records the new current-token K/V for later block-pool append;
- does not retain full past K/V tensors for previous layers after they are consumed.

### Stage 3: Direct-fill layer materialization

Replace the current per-layer rebuild shape:

```python
index_select -> view -> permute/reshape -> per-seq slices -> pad -> cat -> DynamicCache.update
```

with:

```python
preallocate final K/V -> copy block ranges directly into final slices -> append current token slice -> return final K/V
```

The first implementation can use simple Python loops over batch and logical blocks. It may be slower, but the purpose is survival and memory control. Optimize later only if needed.

### Stage 4: A/B gated runner path

Add a runner flag such as:

```bash
--decode-cache-mode rebuild|lazy_layer_rebuild
```

Default remains `rebuild` until Stage 1 and Stage 2 pass. Formal experiments should explicitly record this mode in internal logs.

## What This Does Not Claim

This is not true paged attention. It still materializes a full per-layer K/V tensor before attention, so it will not match a CUDA paged-attention kernel on latency.

The claim is narrower and defensible:

- it reduces the compatibility fallback memory peak;
- it preserves the Python/HF middleware boundary;
- it gives the P2/offload mechanism a better chance to survive high-concurrency long-decode settings without changing C++ kernels.

## Current Probe Result

Environment checked on the new server:

- `transformers = 4.57.6`
- `torch = 2.5.1+cu124`
- probe file: `benchmarks/probes/probe_llama_cache_update_chain.py`
- result file: `benchmarks/results/probes/llama_cache_update_chain.json`

Observed result:

- `update_order = [0, 1, 2]` in the tiny 3-layer Llama probe;
- `cross_layer_reads = 0`;
- `iter_reads = 0`;
- returned `past_key_values` is the same cache object;
- logits are produced successfully.

Conclusion: for the installed HF Llama implementation, a Python-layer lazy per-layer cache is interface-feasible. The next risk is memory behavior inside layer materialization, so the implementation must still remove `DynamicCache.update(cat(...))` and avoid list/pad/cat amplification.


## Implemented Prototype Status

Implemented behind an explicit engine switch:

```bash
--engine-overrides-json '{"decode_path_mode":"lazy_layer_rebuild"}'
```

Default behavior is unchanged. Formal existing runs still use the previous `auto/rebuild` path unless this override is set.

Code changes:

- `LazyPagedRebuildCache` in `core/engine.py`;
- `_rebuild_pkv_lazy_layer()` in `core/engine.py`;
- rebuild CUDA OOM now goes through retryable memory handling and decode microbatch split, instead of only catching Python `MemoryError`;
- forward-path memory failures also call CUDA cleanup before retry/split.

Validation:

1. Cache API probe passed:
   - `benchmarks/results/probes/llama_cache_update_chain.json`
   - `prototype_passed = true`
   - `update_order = [0, 1, 2]`
   - `cross_layer_reads = 0`
   - `iter_reads = 0`

2. Tiny lazy smoke passed:
   - `benchmarks/results/probes/lazy_layer_smoke/smoke_results.json`
   - `Status = Success`
   - `completion_rate = 1.0`
   - `valid_completion_rate = 1.0`
   - `decode_path_selected = lazy_layer_rebuild`

3. Rebuild-vs-lazy deterministic comparison passed:
   - `benchmarks/results/probes/lazy_layer_compare/compare_rebuild_vs_lazy.json`
   - `outputs_equal = true`
   - `token_ids_equal = true`

4. Problem-scenario short probe passed:
   - `benchmarks/results/probes/lazy_layer_8k_c64_n16_p2_frac035/p2_lazy_8k_c64_n16_results.json`
   - setting: `8K x 64 x 16`, `P2`, `gpu_mem_frac=0.35`
   - `Status = Success`
   - `completion_rate = 1.0`
   - `valid_completion_rate = 1.0`
   - `decode_path_selected = lazy_layer_rebuild`
   - `min_free_blocks = 171`, with `wm_low = 167`
   - `p2_attempted_steps = 0`, so this run is a rebuild-memory survival probe, not an offload-trigger proof

Interpretation:

- The Python-only lazy layer path is functionally viable and reduces the all-layer `DynamicCache` peak.
- It is much slower than a true paged-attention kernel, because it still materializes a full K/V tensor per layer and uses Python-level copy loops.
- It should not replace the main formal performance path yet.
- It is useful as a gated compatibility fallback and as evidence that part of the 8K x 64 failure came from rebuild temporary memory, not only KV block capacity.
