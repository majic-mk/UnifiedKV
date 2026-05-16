import math
import time
import gc
import os
import json
from collections import deque
from dataclasses import dataclass, field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

from kv_types import PhaseState, OffloadState
from scheduler import KVScheduler
from chunked_prefill import ChunkedPrefillProcessor
from varlen import HAS_FLASH_ATTN, build_cu_seqlens


class CustomKVCache(DynamicCache):
    """Custom KV cache compatible with the current Transformers DynamicCache API."""

    def __init__(self):
        super().__init__()

    def from_tuple(self, pkv_tuple):
        # Rebuild through DynamicCache.update() so layer metadata such as
        # get_seq_length() stays correct for batched prefill.
        self.layers = []
        for layer_idx, layer_kv in enumerate(tuple(pkv_tuple or ())):
            if not isinstance(layer_kv, (tuple, list)) or len(layer_kv) < 2:
                continue
            K, V = layer_kv[0], layer_kv[1]
            if not (torch.is_tensor(K) and torch.is_tensor(V)):
                continue
            if K.numel() == 0 or V.numel() == 0:
                continue
            self.update(K, V, layer_idx)
        return self

REQ_WAITING_PREFILL = "WAITING_PREFILL"
REQ_PREFILLING = "PREFILLING"
REQ_READY_DECODE = "READY_DECODE"
REQ_DECODING = "DECODING"
REQ_DONE = "DONE"
REQ_FAILED = "FAILED"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None or str(val).strip() == "":
        return bool(default)
    return str(val).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class OnlineRequest:
    request_id: int
    prompt: str
    max_new_tokens: int
    arrival_step: int
    state: str = REQ_WAITING_PREFILL
    generated_tokens: List[int] = field(default_factory=list)
    token_logprobs: List[float] = field(default_factory=list)
    error: Optional[str] = None
    input_ids_cpu: Optional[torch.Tensor] = None
    prompt_token_len: int = 0
    prefill_cursor: int = 0
    prefill_past_kv: Any = None
    last_logits: Optional[torch.Tensor] = None
    current_token: Optional[int] = None
    finished_step: int = -1
    decode_steps: int = 0
    eos_reached: bool = False
    prefill_chunk_cap: int = 0
    prefill_chunk_success_streak: int = 0
    prefill_chunk_recovery_disabled: bool = False
    prefill_activate_retries: int = 0
    decode_state_initialized: bool = False
    submit_time_s: float = 0.0
    first_token_time_s: float = 0.0
    finish_time_s: float = 0.0
    first_token_source: str = ""


class ManagedInferenceEngine:
    def __init__(
        self,
        model_name: str = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
        block_size: int = 16,
        sink_len: int = 16,
        snapkv_observation_len: int = 16,
        retain_ratio: float = 0.20,
        retain_budget_tokens: int = 0,
        selected_writeback_enabled: bool = False,
        chunk_size: int = 2048,
        max_new_tokens: int = 512,
        num_gpus: int = 1,
        cpu_mem_gb: float = 48.0,
        local_model_path: str = None,
        gpu_mem_frac: float = 0.75,
        prefill_batch_size: int = 4,
        decode_micro_batch_size: int = 0,
        stable_greedy_tie_eps: float = 1e-3,
        p2_sink_tokens: int = 16,
        p2_recent_tokens: int = 16,
        decode_window_tiered: bool = True,
        decode_window_thrash_window_steps: int = 16,
        decode_window_thrash_low: float = 0.30,
        decode_window_thrash_high: float = 0.80,
        decode_window_pressure_steps: int = 8,
        decode_window_recover_steps: int = 24,
        ready_decode_eviction_threshold: int = 64,
        p2_target_free_blocks: int = 0,
        wm_low_ratio: float = 0.15,
        wm_high_ratio: float = 0.40,
        p2_enabled: bool = True,
        p2_min_reclaim_blocks: int = 32,
        p2_gain_window_steps: int = 8,
        p2_gain_fail_cooldown_steps: int = 16,
        offload_budget_blocks: int = 256,
        prefetch_budget_blocks: int = 256,
        offload_budget_blocks_base: int = 64,
        offload_budget_blocks_max: int = 256,
        prefetch_budget_blocks_base: int = 64,
        prefetch_budget_blocks_max: int = 256,
        p2_partial_offload_chunk_blocks: int = 128,
        p2_cuda_pressure_min_gb: float = 1.0,
        kv_min_resident_ratio: float = 0.20,
        p2_ready_reclaim_margin_blocks: int = 128,
        p2_max_ready_sequences_per_step: int = 4,
        p2_max_ready_offload_blocks_per_step: int = 320,
        kv_admission_enabled: bool = False,
        kv_admission_margin_blocks: int = 128,
        kv_admission_output_reserve_tokens: int = 0,
        kv_admission_include_low_watermark: bool = True,
        decode_active_cap_initial: int = 0,
        decode_active_cap_min: int = 2,
        decode_active_cap_downscale: float = 0.80,
        decode_active_cap_recover_step: int = 1,
        decode_active_cap_pressure_windows: int = 2,
        decode_active_cap_floor_ratio: float = 0.50,
        decode_retry_priority_quota: float = 0.5,
        per_sequence_max_retry: int = 32,
        max_waiting_requests: int = 512,
        max_prefill_active: int = 8,
        prefill_token_budget_per_step: int = 4096,
        prefill_past_bucket_tokens: int = 0,
        online_prefill_admission_enabled: bool = False,
        online_prefill_short_threshold_tokens: int = 4096,
        online_prefill_mid_threshold_tokens: int = 8192,
        online_prefill_cap_short: int = 4,
        online_prefill_cap_mid: int = 2,
        online_prefill_cap_long: int = 1,
        online_prefill_admission_lookahead: int = 8,
        online_prefill_cuda_headroom_gb: float = 1.0,
        online_prefill_min_effective_chunk: int = 128,
        online_prefill_active_token_budget: int = 0,
        max_decode_active_cap: int = 0,
        scheduler_tick_ms: int = 0,
        decode_memory_guard_hard_guard_gb: float = 2.0,
        decode_paged_flash_enabled: bool = False,
        decode_paged_flash_strict: bool = False,
        decode_page16_native_strict: bool = False,
        decode_path_mode: str = "auto",
    ):
        self.max_new_tokens = max_new_tokens
        self.num_gpus = num_gpus
        self.chunk_size = chunk_size
        self.prefill_batch_size = max(1, prefill_batch_size)
        self.decode_micro_batch_size = max(0, int(decode_micro_batch_size))
        self.stable_greedy_tie_eps = max(0.0, float(stable_greedy_tie_eps))
        self.decode_active_cap_initial = max(0, int(decode_active_cap_initial))
        self.decode_active_cap_min = max(1, int(decode_active_cap_min))
        self.decode_active_cap_downscale = float(max(0.1, min(1.0, decode_active_cap_downscale)))
        self.decode_active_cap_recover_step = max(1, int(decode_active_cap_recover_step))
        self.decode_active_cap_pressure_windows = max(1, int(decode_active_cap_pressure_windows))
        self.decode_active_cap_floor_ratio = float(max(0.0, min(1.0, decode_active_cap_floor_ratio)))
        self.decode_retry_priority_quota = float(max(0.0, min(1.0, decode_retry_priority_quota)))
        self.per_sequence_max_retry = max(1, int(per_sequence_max_retry))
        self._retry_trace_seq_ids: Set[int] = set()
        trace_seq_text = os.environ.get("KV_MIDDLEWARE_TRACE_RETRY_SEQS", "")
        for part in trace_seq_text.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                self._retry_trace_seq_ids.add(int(part))
            except ValueError:
                continue
        self._retry_trace_path = os.environ.get("KV_MIDDLEWARE_RETRY_TRACE_PATH", "").strip()
        self._retry_trace_enabled = bool(self._retry_trace_seq_ids and self._retry_trace_path)
        if self._retry_trace_enabled:
            try:
                trace_dir = os.path.dirname(self._retry_trace_path)
                if trace_dir:
                    os.makedirs(trace_dir, exist_ok=True)
                with open(self._retry_trace_path, "w", encoding="utf-8"):
                    pass
            except Exception:
                self._retry_trace_enabled = False
        self.max_waiting_requests = max(1, int(max_waiting_requests))
        self.max_prefill_active = max(1, _env_int("KV_MIDDLEWARE_MAX_PREFILL_ACTIVE", int(max_prefill_active)))
        self.prefill_token_budget_per_step = max(0, int(prefill_token_budget_per_step))
        self.prefill_past_bucket_tokens = max(0, int(prefill_past_bucket_tokens))
        self.online_prefill_admission_enabled = bool(online_prefill_admission_enabled) or _env_flag("KV_MIDDLEWARE_ONLINE_PREFILL_ADMISSION", False)
        self.online_prefill_short_threshold_tokens = max(1, _env_int("KV_MIDDLEWARE_ONLINE_PREFILL_SHORT_THRESHOLD_TOKENS", int(online_prefill_short_threshold_tokens)))
        self.online_prefill_mid_threshold_tokens = max(
            self.online_prefill_short_threshold_tokens + 1,
            _env_int("KV_MIDDLEWARE_ONLINE_PREFILL_MID_THRESHOLD_TOKENS", int(online_prefill_mid_threshold_tokens)),
        )
        self.online_prefill_cap_short = max(1, _env_int("KV_MIDDLEWARE_ONLINE_PREFILL_CAP_SHORT", int(online_prefill_cap_short)))
        self.online_prefill_cap_mid = max(1, _env_int("KV_MIDDLEWARE_ONLINE_PREFILL_CAP_MID", int(online_prefill_cap_mid)))
        self.online_prefill_cap_long = max(1, _env_int("KV_MIDDLEWARE_ONLINE_PREFILL_CAP_LONG", int(online_prefill_cap_long)))
        self.online_prefill_admission_lookahead = max(1, _env_int("KV_MIDDLEWARE_ONLINE_PREFILL_ADMISSION_LOOKAHEAD", int(online_prefill_admission_lookahead)))
        self.online_prefill_cuda_headroom_gb = max(0.0, _env_float("KV_MIDDLEWARE_ONLINE_PREFILL_CUDA_HEADROOM_GB", float(online_prefill_cuda_headroom_gb)))
        self.online_prefill_min_effective_chunk = max(1, _env_int("KV_MIDDLEWARE_ONLINE_PREFILL_MIN_EFFECTIVE_CHUNK", int(online_prefill_min_effective_chunk)))
        self.online_prefill_active_token_budget = max(0, _env_int("KV_MIDDLEWARE_ONLINE_PREFILL_ACTIVE_TOKEN_BUDGET", int(online_prefill_active_token_budget)))
        self.p2_ready_reclaim_margin_blocks = max(0, _env_int("KV_MIDDLEWARE_P2_READY_RECLAIM_MARGIN_BLOCKS", int(p2_ready_reclaim_margin_blocks)))
        self.p2_max_ready_sequences_per_step = max(1, _env_int("KV_MIDDLEWARE_P2_MAX_READY_SEQUENCES_PER_STEP", int(p2_max_ready_sequences_per_step)))
        self.p2_max_ready_offload_blocks_per_step = max(1, _env_int("KV_MIDDLEWARE_P2_MAX_READY_OFFLOAD_BLOCKS_PER_STEP", int(p2_max_ready_offload_blocks_per_step)))
        self.kv_admission_enabled = bool(kv_admission_enabled) or _env_flag("KV_MIDDLEWARE_KV_ADMISSION", False)
        self.kv_admission_margin_blocks = max(0, _env_int("KV_MIDDLEWARE_KV_ADMISSION_MARGIN_BLOCKS", int(kv_admission_margin_blocks)))
        self.kv_admission_output_reserve_tokens = max(0, _env_int("KV_MIDDLEWARE_KV_ADMISSION_OUTPUT_RESERVE_TOKENS", int(kv_admission_output_reserve_tokens)))
        env_include_low = os.environ.get("KV_MIDDLEWARE_KV_ADMISSION_INCLUDE_LOW_WATERMARK", None)
        if env_include_low is None:
            self.kv_admission_include_low_watermark = bool(kv_admission_include_low_watermark)
        else:
            self.kv_admission_include_low_watermark = str(env_include_low).strip().lower() in {"1", "true", "yes", "on"}
        self.p2_post_offload_cooldown_steps = 2
        self.max_decode_active_cap = max(0, int(max_decode_active_cap))
        self.scheduler_tick_ms = max(0, int(scheduler_tick_ms))
        self.decode_memory_guard_hard_guard_gb = max(0.0, float(decode_memory_guard_hard_guard_gb))
        mode = str(decode_path_mode).strip().lower()
        if mode not in ("auto", "paged_direct", "paged_materialize", "page16_native", "rebuild"):
            mode = "auto"
        self.decode_path_mode = mode
        # Keep legacy flag for compatibility, but auto/paged_direct mode implies paged path probing.
        self.decode_paged_flash_enabled = bool(
            decode_paged_flash_enabled or self.decode_path_mode in ("auto", "paged_direct", "paged_materialize")
        )
        self.decode_paged_flash_strict = bool(decode_paged_flash_strict)
        self.decode_page16_native_strict = bool(decode_page16_native_strict)
        self.decode_paged_flash_active = False
        self.decode_paged_flash_reason = "disabled"
        self.decode_no_progress_watchdog_steps = 4
        self.prefill_no_progress_watchdog_steps = max(8, _env_int("KV_MIDDLEWARE_PREFILL_NO_PROGRESS_WATCHDOG_STEPS", 64))
        self.decode_backpressure_pause_steps = 4
        self._decode_rr_cursor = 0
        self._prefill_rr_cursor = 0
        self._next_request_id = 0

        model_path = local_model_path if local_model_path else model_name
        print(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = 'auto' if num_gpus > 1 else 'cuda'
        self.flash_attn_enabled = False
        load_kwargs = {
            'torch_dtype': torch.float16,
            'device_map': device_map,
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
        }
        if HAS_FLASH_ATTN:
            load_kwargs['attn_implementation'] = 'flash_attention_2'
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs,
            )
            self.flash_attn_enabled = load_kwargs.get('attn_implementation') == 'flash_attention_2'
        except Exception as exc:
            if 'attn_implementation' in load_kwargs:
                print(f"flash_attention_2 unavailable, fallback to default attention: {exc}")
                load_kwargs.pop('attn_implementation', None)
                # Release partial allocations from the failed flash-attn load path.
                try:
                    if hasattr(self, "model"):
                        del self.model
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **load_kwargs,
                )
                self.flash_attn_enabled = False
            else:
                raise
        self.model.eval()

        torch.cuda.synchronize()
        model_used = torch.cuda.memory_allocated()
        total_gpu = torch.cuda.get_device_properties(0).total_memory
        reserve = int(0.10 * total_gpu)
        available = total_gpu - model_used - reserve
        actual_frac = gpu_mem_frac if gpu_mem_frac < 1.0 else max(0.1, available / total_gpu)

        print(f"Model memory: {model_used/1024**3:.1f}GB  KV-available: {available/1024**3:.1f}GB")

        # Runtime is P2-only. Keep decode pressure signals, but retire
        # decode-window pruning / P3 runtime configuration.

        self.scheduler = KVScheduler(
            model=self.model,
            block_size=block_size,
            sink_len=sink_len,
            snapkv_observation_len=snapkv_observation_len,
            retain_ratio=retain_ratio,
            retain_budget_tokens=retain_budget_tokens,
            selected_writeback_enabled=selected_writeback_enabled,
            gpu_mem_frac=actual_frac,
            cpu_mem_gb=cpu_mem_gb,
            p2_sink_tokens=p2_sink_tokens,
            p2_recent_tokens=p2_recent_tokens,
            decode_window_tiered=decode_window_tiered,
            decode_window_thrash_window_steps=decode_window_thrash_window_steps,
            decode_window_thrash_low=decode_window_thrash_low,
            decode_window_thrash_high=decode_window_thrash_high,
            decode_window_pressure_steps=decode_window_pressure_steps,
            decode_window_recover_steps=decode_window_recover_steps,
            ready_decode_eviction_threshold=ready_decode_eviction_threshold,
            p2_target_free_blocks=p2_target_free_blocks,
            wm_low_ratio=wm_low_ratio,
            wm_high_ratio=wm_high_ratio,
            p2_enabled=p2_enabled,
            p2_min_reclaim_blocks=p2_min_reclaim_blocks,
            p2_gain_window_steps=p2_gain_window_steps,
            p2_gain_fail_cooldown_steps=p2_gain_fail_cooldown_steps,
            offload_budget_blocks=offload_budget_blocks,
            prefetch_budget_blocks=prefetch_budget_blocks,
            p2_cuda_pressure_min_gb=p2_cuda_pressure_min_gb,
            kv_min_resident_ratio=kv_min_resident_ratio,
        )
        self.scheduler.offloader.configure_transfer_budget_limits(
            offload_base=offload_budget_blocks_base,
            offload_max=offload_budget_blocks_max or offload_budget_blocks,
            prefetch_base=prefetch_budget_blocks_base,
            prefetch_max=prefetch_budget_blocks_max or prefetch_budget_blocks,
            partial_chunk=p2_partial_offload_chunk_blocks,
        )
        self.scheduler.offloader.set_step_transfer_budgets(
            offload_budget=offload_budget_blocks_base,
            prefetch_budget=prefetch_budget_blocks_base,
        )
        if self.prefill_past_bucket_tokens <= 0:
            self.prefill_past_bucket_tokens = max(1, int(self.scheduler.pool.B))
        support_ok, support_reason = self.scheduler.paged_flash_decode_support()
        if self.decode_paged_flash_enabled and num_gpus == 1 and support_ok:
            self.decode_paged_flash_active = True
            self.decode_paged_flash_reason = "ok"
        elif self.decode_paged_flash_enabled and num_gpus != 1:
            self.decode_paged_flash_reason = "multi_gpu_not_supported"
        elif self.decode_paged_flash_enabled:
            self.decode_paged_flash_reason = support_reason
        else:
            self.decode_paged_flash_reason = "disabled"
        self.chunker = ChunkedPrefillProcessor(chunk_size=chunk_size)
        self.multi_gpu = None

        if num_gpus > 1:
            from multi_gpu import MultiGPUCoordinator

            cfg = self.model.config
            self.multi_gpu = MultiGPUCoordinator(
                num_gpus=num_gpus,
                block_size=block_size,
                num_layers=cfg.num_hidden_layers,
                num_kv_heads_total=getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads),
                head_dim=cfg.hidden_size // cfg.num_attention_heads,
                dtype=torch.float16,
                cpu_mem_gb_per_gpu=cpu_mem_gb / num_gpus,
            )

        print(
            f"Init done: N_total={self.scheduler.pool.N_total} blocks  "
            f"flash_attn={'yes' if self.flash_attn_enabled else 'no (fallback)'}  "
            f"decode_path_mode={self.decode_path_mode}"
        )
        if self.decode_paged_flash_enabled:
            print(
                "paged_flash_decode="
                f"{'enabled' if self.decode_paged_flash_active else 'disabled'}"
                f" ({self.decode_paged_flash_reason})"
            )
        self._reset_online_runtime(clear_request_counter=False)

    @staticmethod
    def _percentile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        data = sorted(values)
        pos = (len(data) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(data) - 1)
        if lo == hi:
            return float(data[lo])
        frac = pos - lo
        return float(data[lo] * (1.0 - frac) + data[hi] * frac)

    @staticmethod
    def _stats_delta(after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
        keys = set(after.keys()) | set(before.keys())
        return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in sorted(keys)}

    def _reset_online_runtime(self, clear_request_counter: bool = False):
        self._requests: Dict[int, OnlineRequest] = {}
        self.waiting_queue: Deque[int] = deque()
        self.prefill_active: Deque[int] = deque()
        self.ready_decode: Deque[int] = deque()
        self.decode_active: Deque[int] = deque()
        self.finished_queue: Deque[Dict[str, Any]] = deque()
        self._online_step = 0
        self._online_decode_started = False
        self._online_total_t0 = 0.0
        self._online_prefill_ms = 0.0
        self._online_decode_step_lat_ms: List[float] = []
        self._online_generated_tokens = 0
        self._online_decode_min_n_free = int(self.scheduler.pool.n_free)
        self._online_prefill_min_n_free = int(self.scheduler.pool.n_free)
        self._online_global_min_n_free = int(self.scheduler.pool.n_free)
        self._online_kv_total_blocks = int(self.scheduler.pool.N_total)
        self._online_kv_peak_used_blocks = max(0, int(self.scheduler.pool.N_total) - int(self.scheduler.pool.n_free))
        self._online_decode_active_cap = 0
        self._online_decode_active_cap_boot = 0
        self._online_decode_active_cap_min_seen = 0
        self._online_decode_cap_pressure_streak = 0
        self._online_decode_cap_recover_streak = 0
        self._online_decode_microbatch_sizes: List[int] = []
        self._online_decode_append_fail_count = 0
        self._online_decode_backpressure_events = 0
        self._online_decode_memory_cap_events = 0
        self._online_decode_memory_cap_min_batch = 0
        self._online_decode_memory_guard_target_batch_last = 0
        self._online_decode_memory_guard_source_batch_last = 0
        self._online_decode_memory_guard_free_gb_last = 0.0
        self._online_decode_memory_guard_budget_gb_last = 0.0
        self._online_decode_memory_guard_reserve_gb_last = 0.0
        self._online_decode_memory_guard_hard_guard_gb = float(self.decode_memory_guard_hard_guard_gb)
        self._online_guard_seen_count = 0
        self._online_guard_effective_shrink_count = 0
        self._online_guard_strong_shrink_count = 0
        self._online_guard_target_batch_min = 0
        self._online_guard_source_batch_max = 0
        self._online_p2_reject_deferred = 0
        self._online_p2_reject_no_resident = 0
        self._online_p2_reject_protected = 0
        self._online_p2_ready_protected_ignored = 0
        self._online_p2_reject_active_floor = 0
        self._online_p2_reject_plan_empty = 0
        self._online_kv_admission_blocked_steps = 0
        self._online_kv_admission_blocked_requests = 0
        self._online_kv_admission_last_free_blocks = 0
        self._online_kv_admission_last_required_blocks = 0
        self._online_kv_admission_last_workload_demand_blocks = 0
        self._online_kv_admission_last_reserved_blocks = 0
        self._online_kv_admission_last_margin_blocks = 0
        self._online_kv_admission_last_total_required_blocks = 0
        self._online_kv_admission_last_prompt_blocks = 0
        self._online_kv_admission_last_prompt_resident_blocks = 0
        self._online_kv_admission_last_request_blocks = 0
        self._online_kv_admission_last_pending_output_blocks = 0
        self._online_kv_admission_last_pending_prompt_blocks = 0
        self._online_kv_admission_last_output_reserve_tokens = 0
        self._online_kv_admission_last_output_reserve_blocks = 0
        self._online_kv_admission_last_allowed = 1
        self._online_kv_admission_last_blocked_step = -1
        self._online_prefill_admission_blocked_steps = 0
        self._online_prefill_admission_blocked_requests = 0
        self._online_prefill_admission_last_blocked_step = -1
        self._online_prefill_admission_last_reason = ""
        self._online_prefill_admission_last_prompt_len = 0
        self._online_prefill_admission_last_bucket = ""
        self._online_prefill_admission_last_cuda_free_gb = 0.0
        self._online_prefill_admission_last_active_short = 0
        self._online_prefill_admission_last_active_mid = 0
        self._online_prefill_admission_last_active_long = 0
        self._online_prefill_admission_last_cap = 0
        self._online_prefill_admission_last_active_tokens = 0
        self._online_prefill_admission_last_projected_tokens = 0
        self._online_prefill_admission_last_token_budget = 0
        self._online_prefill_admission_token_budget_blocked_steps = 0
        self._online_prefill_admission_token_budget_blocked_requests = 0
        self._online_prefill_admission_token_budget_last_blocked_step = -1
        self._online_prefill_admission_last_allowed = 1
        self._online_prefill_chunk_floor_pause_steps = 0
        self._online_prefill_chunk_floor_last_chunk_cap = 0
        self._online_decode_memory_est_peak_max_gb = 0.0
        self._online_decode_memory_est_peak_maxlen_gb = 0.0
        self._online_decode_memory_est_peak_sumlen_gb = 0.0
        self._online_decode_memory_aware_cap_enabled = 1
        self._online_decode_memory_aware_margin_gb = max(1.0, float(getattr(self.scheduler, 'p2_cuda_pressure_min_gb', 1.0) or 1.0))
        self._online_decode_memory_aware_peak_factor = 1.25
        self._online_decode_length_bucketed_steps = 0
        self._online_decode_length_bucket_subbatch_count = 0
        self._online_decode_length_bucket_singleton_count = 0
        self._online_decode_length_bucket_max_trigger_ratio = 0.0
        self._online_prefill_backpressure_events = 0
        self._online_prefill_batch_failed_steps = 0
        self._online_prefill_chunk_failed_steps = 0
        self._online_prefill_activate_failed_steps = 0
        self._online_prefill_batch_merge_peak_gb = 0.0
        self._online_prefill_batch_input_peak_gb = 0.0
        self._online_prefill_batch_forward_peak_gb = 0.0
        self._online_prefill_batch_slice_peak_gb = 0.0
        self._online_prefill_chunk_input_peak_gb = 0.0
        self._online_prefill_chunk_forward_peak_gb = 0.0
        self._online_prefill_no_progress_steps = 0
        self._online_prefill_no_progress_streak = 0
        self._online_prefill_no_progress_peak = 0
        self._online_prefill_no_progress_fail_count = 0
        self._online_prefill_no_progress_last_cursor = -1
        self._online_decode_retry_timeout_fail_count = 0
        self._online_decode_retry_timeout_seq_ids: List[int] = []
        self._online_decode_no_progress_steps = 0
        self._online_decode_no_progress_streak = 0
        self._online_decode_no_progress_peak = 0
        self._online_decode_path_selected_counts: Dict[str, int] = {}
        self._online_decode_path_last_selected: str = "none"
        self._online_decode_path_fallback_count = 0
        self._online_decode_path_fallback_reasons: Dict[str, int] = {}
        self._online_decode_paged_direct_steps = 0
        self._online_decode_page16_native_steps = 0
        self._online_decode_rebuild_steps = 0
        self._online_decode_materialize_kv_bytes = 0
        self._online_decode_paged_direct_blocked_reason = ""
        self._online_decode_paged_direct_resident_miss_steps = 0
        self._online_decode_page16_native_blocked_reason = ""
        self._online_decode_page16_native_resident_miss_steps = 0
        self._online_decode_page16_native_kernel_ms = 0.0
        self._online_last_missing_blocks_scheduled = 0
        self._online_last_materialized_blocks_scheduled = 0
        self._online_prefill_pause_steps = 0
        self._priority_retry: Deque[int] = deque()
        self._priority_retry_set: Set[int] = set()
        self._consecutive_retry_count: Dict[int, int] = {}
        self._p2_offload_cooldown_until: Dict[int, int] = {}
        self._online_offload_stats_before = self.scheduler.offloader.get_stats(reset=False)
        self._online_offload_stats_last = dict(self._online_offload_stats_before)
        if torch.cuda.is_available():
            try:
                free_b, total_b = torch.cuda.mem_get_info()
            except Exception:
                total_b = int(torch.cuda.get_device_properties(0).total_memory)
                free_b = max(0, total_b - int(torch.cuda.memory_reserved()))
        else:
            free_b, total_b = 0, 0
        self._online_cuda_total_bytes = int(total_b)
        self._online_cuda_free_min_bytes = int(free_b)
        self._online_cuda_alloc_peak_bytes = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
        self._online_cuda_reserved_peak_bytes = int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
        free_gb_now = (float(free_b) / 1024**3) if torch.cuda.is_available() else 0.0
        self._online_cuda_free_post_cleanup_gb = free_gb_now
        self._online_cuda_free_post_cleanup_last_gb = free_gb_now
        self._online_decode_cuda_free_post_cleanup_recent_min_gb = float("inf")
        self.scheduler.reset_runtime_state(cuda_free_gb=free_gb_now)
        self._return_details_online = False
        self._decode_rr_cursor = 0
        self._prefill_rr_cursor = 0
        if clear_request_counter:
            self._next_request_id = 0

    def _update_online_pool_stats(self, phase: str = "") -> None:
        n_free = int(self.scheduler.pool.n_free)
        n_total = int(self.scheduler.pool.N_total)
        self._online_global_min_n_free = min(int(self._online_global_min_n_free), n_free)
        if phase == "prefill":
            self._online_prefill_min_n_free = min(int(self._online_prefill_min_n_free), n_free)
        elif phase == "decode":
            self._online_decode_min_n_free = min(int(self._online_decode_min_n_free), n_free)
        self._online_kv_total_blocks = n_total
        self._online_kv_peak_used_blocks = max(int(self._online_kv_peak_used_blocks), max(0, n_total - n_free))
        self._update_online_cuda_stats()

    def _update_online_cuda_stats(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            free_b, total_b = torch.cuda.mem_get_info()
        except Exception:
            total_b = int(torch.cuda.get_device_properties(0).total_memory)
            free_b = max(0, total_b - int(torch.cuda.memory_reserved()))
        alloc_b = int(torch.cuda.memory_allocated())
        reserved_b = int(torch.cuda.memory_reserved())
        self._online_cuda_total_bytes = max(int(self._online_cuda_total_bytes), int(total_b))
        self._online_cuda_free_min_bytes = min(int(self._online_cuda_free_min_bytes), int(free_b))
        self._online_cuda_alloc_peak_bytes = max(int(self._online_cuda_alloc_peak_bytes), int(alloc_b))
        self._online_cuda_reserved_peak_bytes = max(int(self._online_cuda_reserved_peak_bytes), int(reserved_b))

    def _sample_cuda_free_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        try:
            free_b, _ = torch.cuda.mem_get_info()
        except Exception:
            total_b = int(torch.cuda.get_device_properties(0).total_memory)
            free_b = max(0, total_b - int(torch.cuda.memory_reserved()))
        return float(free_b) / 1024**3

    def _record_post_cleanup_cuda_state(self, active_seq_count: int) -> None:
        self._update_online_cuda_stats()
        free_gb = float(self._sample_cuda_free_gb())
        self._online_cuda_free_post_cleanup_last_gb = free_gb
        self._online_cuda_free_post_cleanup_gb = min(float(self._online_cuda_free_post_cleanup_gb), free_gb)
        self.scheduler.set_cuda_free_post_cleanup_gb(free_gb)
        self.scheduler.update_decode_pressure(active_seq_count=active_seq_count)
        if int(active_seq_count) > 0:
            recent_min = float(self.scheduler.decode_cuda_free_post_cleanup_recent_min_gb())
            if math.isfinite(recent_min):
                self._online_decode_cuda_free_post_cleanup_recent_min_gb = min(
                    float(self._online_decode_cuda_free_post_cleanup_recent_min_gb),
                    recent_min,
                )

    def _passes_active_decode_resident_floor(self, info: Dict[str, int]) -> bool:
        if int(info.get('residency_protected', 0)) > 0:
            return False
        logical_blocks = max(
            0,
            int(info.get('materialized_blocks', 0) or info.get('logical_blocks', 0)),
        )
        resident_blocks = max(0, int(info.get('resident_blocks', 0)))
        if logical_blocks <= 0:
            return True
        min_blocks = int(self.scheduler.min_resident_blocks_required(logical_blocks))
        return resident_blocks > min_blocks

    def _p2_protected_block_indices(self, logical_blocks: int) -> Set[int]:
        sched = self.scheduler
        B = max(1, int(sched.pool.B))
        sink_tokens = max(int(getattr(sched, 'p2_sink_tokens', 0)), int(getattr(sched, 'sink_len', 0)))
        recent_tokens = max(0, int(getattr(sched, 'p2_recent_tokens', 0)))
        sink_blocks = max(0, int(math.ceil(float(sink_tokens) / float(B)))) if sink_tokens > 0 else 0
        recent_blocks = max(0, int(math.ceil(float(recent_tokens) / float(B)))) if recent_tokens > 0 else 0
        sink_blocks = min(logical_blocks, sink_blocks)
        recent_blocks = min(logical_blocks, recent_blocks)
        protected: Set[int] = set(range(sink_blocks))
        if recent_blocks > 0:
            tail_start = max(0, logical_blocks - recent_blocks)
            protected.update(range(tail_start, logical_blocks))
        return protected

    def _plan_p2_offload_blocks(self, seq_id: int, offload_all_resident: bool = False) -> List[int]:
        sched = self.scheduler
        entry = sched.offloader._get_seq_layer0_entry(int(seq_id))
        if entry is None:
            return []
        sched.offloader._ensure_entry_maps(entry)
        logical_blocks = len(entry.gpu_block_map)
        if logical_blocks <= 0:
            return []
        resident_indices = [idx for idx, bid in enumerate(entry.gpu_block_map) if int(bid) >= 0]
        if not resident_indices:
            return []
        if bool(offload_all_resident):
            return sorted(resident_indices)
        resident_blocks = int(len(resident_indices))
        min_resident = int(sched.min_resident_blocks_required(logical_blocks))
        max_drop = max(0, resident_blocks - min_resident)
        if max_drop <= 0:
            return []
        protected = self._p2_protected_block_indices(logical_blocks)
        candidates = [idx for idx in resident_indices if idx not in protected]
        if not candidates:
            return []
        candidates.sort()
        return candidates[:max_drop]

    def _compute_p2_offload_budget(
        self,
        ready_plans: Dict[int, List[int]],
        decode_plans: Dict[int, List[int]],
        ready_reclaim_blocks: Optional[Dict[int, int]] = None,
        target_reclaim_blocks: int = 0,
    ) -> int:
        sched = self.scheduler
        offloader = sched.offloader
        ready_planned_blocks = int(
            sum(int((ready_reclaim_blocks or {}).get(sid, len(v))) for sid, v in ready_plans.items())
        )
        decode_planned_blocks = int(sum(len(plan) for plan in decode_plans.values()))
        desired = max(
            int(offloader.offload_budget_blocks_base),
            int(target_reclaim_blocks),
            int(ready_planned_blocks),
            int(decode_planned_blocks),
        )
        capped = min(int(offloader.offload_budget_blocks_max), int(desired))
        return max(int(offloader.offload_budget_blocks_base), int(capped))

    def _mark_p2_offload_cooldown(self, seq_id: int) -> None:
        if self.p2_post_offload_cooldown_steps <= 0:
            return
        sid = int(seq_id)
        now = int(self.scheduler.decode_step_count)
        until = now + int(self.p2_post_offload_cooldown_steps)
        prev = int(self._p2_offload_cooldown_until.get(sid, 0))
        if until > prev:
            self._p2_offload_cooldown_until[sid] = until

    def _is_p2_offload_deferred(self, seq_id: int) -> bool:
        sid = int(seq_id)
        until = int(self._p2_offload_cooldown_until.get(sid, 0))
        if until <= 0:
            return False
        now = int(self.scheduler.decode_step_count)
        if now >= until:
            self._p2_offload_cooldown_until.pop(sid, None)
            return False
        return True

    def _p2_last_access_age(self, seq_id: int, info: Optional[Dict[str, int]] = None) -> int:
        if info is None:
            info = self.scheduler.offloader.get_sequence_residency(int(seq_id))
        last_access = int(info.get('last_access', -1))
        now = int(self.scheduler.decode_step_count)
        if last_access < 0:
            return max(0, now + 1)
        return max(0, now - last_access)

    @staticmethod
    def _p2_candidate_sort_key(candidate: Dict[str, int]) -> Tuple[int, int, int]:
        return (
            int(candidate.get('last_access', -1)),
            -int(candidate.get('releasable_blocks', 0)),
            int(candidate.get('seq_id', 0)),
        )

    def _build_p2_candidate(
        self,
        seq_id: int,
        require_active_floor: bool,
    ) -> Optional[Dict[str, int]]:
        sched = self.scheduler
        sid = int(seq_id)
        ready_candidate = not bool(require_active_floor)
        if (not ready_candidate) and self._is_p2_offload_deferred(sid):
            self._online_p2_reject_deferred += 1
            return None
        info = sched.offloader.get_sequence_residency(sid)
        if int(info.get('resident_blocks', 0)) <= 0:
            self._online_p2_reject_no_resident += 1
            return None
        if (not ready_candidate) and int(info.get('residency_protected', 0)) > 0:
            self._online_p2_reject_protected += 1
            return None
        if require_active_floor and not self._passes_active_decode_resident_floor(info):
            self._online_p2_reject_active_floor += 1
            return None
        plan = self._plan_p2_offload_blocks(sid, offload_all_resident=ready_candidate)
        releasable = int(len(plan))
        if releasable <= 0:
            self._online_p2_reject_plan_empty += 1
            return None
        return {
            'seq_id': sid,
            'last_access': int(info.get('last_access', -1)),
            'last_access_age': int(self._p2_last_access_age(sid, info)),
            'releasable_blocks': releasable,
            'plan': plan,
            'ready_decode_candidate': int(ready_candidate),
        }

    def _layer1_ready_decode_candidates(self) -> List[Dict[str, int]]:
        candidates: List[Dict[str, int]] = []
        for sid in list(self.ready_decode):
            cand = self._build_p2_candidate(int(sid), require_active_floor=False)
            if cand is None:
                continue
            candidates.append(cand)
        candidates.sort(key=lambda c: (-int(c['releasable_blocks']), int(c['seq_id'])))
        return candidates

    def _select_p2_ready_offload_candidates(
        self,
        candidates: Sequence[Dict[str, int]],
        target_reclaim_blocks: int,
    ) -> Tuple[List[Dict[str, int]], int, str]:
        selected: List[Dict[str, int]] = []
        planned_blocks = 0
        max_sequences = max(1, int(self.p2_max_ready_sequences_per_step))
        max_blocks = max(1, int(self.p2_max_ready_offload_blocks_per_step))
        target = max(0, int(target_reclaim_blocks))
        if target <= 0:
            return [], 0, "not_needed"
        if not candidates:
            return [], 0, "no_ready_candidate"
        for cand in candidates:
            cand_blocks = max(0, int(cand.get('releasable_blocks', 0)))
            if cand_blocks <= 0:
                continue
            if len(selected) >= max_sequences:
                return selected, planned_blocks, "sequence_cap_reached"
            if planned_blocks + cand_blocks > max_blocks:
                if selected:
                    return selected, planned_blocks, "block_cap_reached"
                return [], 0, "block_cap_reached"
            selected.append(dict(cand))
            planned_blocks += cand_blocks
            if planned_blocks >= target:
                return selected, planned_blocks, "target_reached"
        return selected, planned_blocks, "no_ready_candidate" if not selected else "candidate_exhausted"

    def _layer2_active_candidates(self, protected_seq_ids: Sequence[int]) -> List[Dict[str, int]]:
        protected = set(int(x) for x in protected_seq_ids)
        candidates: List[Dict[str, int]] = []
        for sid in list(self.decode_active):
            sid_i = int(sid)
            if sid_i in protected:
                continue
            cand = self._build_p2_candidate(sid_i, require_active_floor=True)
            if cand is None or int(cand['last_access_age']) < 2:
                continue
            candidates.append(cand)
        candidates.sort(key=self._p2_candidate_sort_key)
        return candidates

    def _layer3_cold_active_fallback_candidates(self, protected_seq_ids: Sequence[int]) -> List[Dict[str, int]]:
        protected = set(int(x) for x in protected_seq_ids)
        candidates: List[Dict[str, int]] = []
        for sid in list(self.decode_active):
            sid_i = int(sid)
            if sid_i in protected:
                continue
            cand = self._build_p2_candidate(sid_i, require_active_floor=True)
            if cand is None or int(cand['last_access_age']) < 4:
                continue
            candidates.append(cand)
        candidates.sort(key=self._p2_candidate_sort_key)
        return candidates

    def _maybe_relieve_p2_pressure(self, protected_seq_ids: Sequence[int]) -> Dict[str, int]:
        sched = self.scheduler
        if not bool(getattr(sched, "p2_enabled", True)):
            sched.record_p2_attempt(attempted=False, success=False, no_candidate=False, candidate_count=0)
            return {'attempted': 0, 'success': 0, 'no_candidate': 0, 'offloaded': 0, 'candidate_count': 0}
        low_threshold = int(sched.p2_low_threshold())
        target_free = max(
            int(sched.p2_target_free_threshold()),
            int(low_threshold) + int(self.p2_ready_reclaim_margin_blocks),
        )
        pool_pressure = int(sched.pool.n_free) <= int(sched.p2_low_threshold())
        has_decode_work = bool(self.decode_active or self.ready_decode)
        cuda_pressure = bool(sched.p2_cuda_pressure_signal(has_decode_work=has_decode_work))
        pressure_deficit = max(0, low_threshold - int(sched.pool.n_free))
        reclaim_gate = max(
            1,
            int(getattr(sched, 'p2_min_reclaim_blocks', 0) or 0),
        )
        free_before = int(sched.pool.n_free)
        cuda_free_before = float(sched.cuda_free_post_cleanup_gb())
        if (not bool(sched._p2_managed_active)) and (not pool_pressure) and not cuda_pressure:
            sched.record_p2_attempt(attempted=False, success=False, no_candidate=False, candidate_count=0)
            return {'attempted': 0, 'success': 0, 'no_candidate': 0, 'offloaded': 0, 'candidate_count': 0}
        if sched.p2_gain_fail_cooldown_active():
            sched.record_p2_attempt(attempted=False, success=False, no_candidate=False, candidate_count=0)
            return {'attempted': 0, 'success': 0, 'no_candidate': 0, 'offloaded': 0, 'candidate_count': 0, 'offload_budget': 0}
        all_seq_ids = list(self.ready_decode) + list(self.decode_active)
        for sid in all_seq_ids:
            try:
                sched.offloader._try_finalize_offload(int(sid), force_sync=False)
                sched.offloader._try_finalize_prefetch(int(sid), force_sync=False)
            except Exception:
                pass
        pool_pressure = int(sched.pool.n_free) <= int(sched.p2_low_threshold())
        cuda_pressure = bool(sched.p2_cuda_pressure_signal(has_decode_work=has_decode_work))
        if int(sched.pool.n_free) >= target_free and not cuda_pressure:
            sched.record_p2_attempt(attempted=False, success=True, no_candidate=False, candidate_count=0)
            return {'attempted': 0, 'success': 1, 'no_candidate': 0, 'offloaded': 0, 'candidate_count': 0}

        layer1_all = self._layer1_ready_decode_candidates()
        # Keep online P2 serviceable: only ready_decode is eligible in this stage.
        layer2: List[Dict[str, int]] = []
        layer3: List[Dict[str, int]] = []

        offloader = sched.offloader
        target_reclaim_blocks = max(0, int(target_free) - int(sched.pool.n_free))
        decode_plans: Dict[int, List[int]] = {}
        total_ready_reclaim_blocks = int(sum(int(c.get('releasable_blocks', 0)) for c in layer1_all))
        expected_reclaim_blocks = int(total_ready_reclaim_blocks)
        sched.record_p2_candidate_mix(
            ready_candidate_count=len(layer1_all),
            decode_candidate_count=0,
            expected_reclaim_blocks=int(expected_reclaim_blocks),
        )
        candidate_count = len(layer1_all)
        no_candidate = candidate_count == 0
        if no_candidate:
            if hasattr(sched, 'record_p2_ready_selection'):
                sched.record_p2_ready_selection(0, 0, int(target_reclaim_blocks), 0, "no_ready_candidate")
            sched.record_p2_attempt(attempted=False, success=False, no_candidate=True, candidate_count=0)
            return {'attempted': 0, 'success': 0, 'no_candidate': 1, 'offloaded': 0, 'candidate_count': 0, 'offload_budget': 0}
        if int(target_reclaim_blocks) <= 0:
            if hasattr(sched, 'record_p2_ready_selection'):
                sched.record_p2_ready_selection(0, 0, int(target_reclaim_blocks), 0, "not_needed")
            sched.record_p2_attempt(attempted=False, success=True, no_candidate=False, candidate_count=int(candidate_count))
            return {'attempted': 0, 'success': 1, 'no_candidate': 0, 'offloaded': 0, 'candidate_count': int(candidate_count), 'offload_budget': 0}
        hard_pool_pressure = bool(
            (pool_pressure and int(low_threshold - int(sched.pool.n_free)) >= int(reclaim_gate))
            or bool(cuda_pressure)
        )
        low_benefit_floor = min(32, max(1, int(math.ceil(float(target_reclaim_blocks) * 0.25))))
        if (not bool(hard_pool_pressure)) and int(total_ready_reclaim_blocks) < int(low_benefit_floor):
            sched.record_p2_low_benefit_skip(int(expected_reclaim_blocks))
            if hasattr(sched, 'record_p2_ready_selection'):
                sched.record_p2_ready_selection(0, 0, int(target_reclaim_blocks), 0, "low_benefit_skip")
            sched.record_p2_attempt(attempted=False, success=False, no_candidate=False, candidate_count=int(candidate_count))
            return {'attempted': 0, 'success': 0, 'no_candidate': 0, 'offloaded': 0, 'candidate_count': int(candidate_count), 'offload_budget': 0}
        layer1, planned_ready_blocks, ready_stop_reason = self._select_p2_ready_offload_candidates(layer1_all, int(target_reclaim_blocks))
        ready_plans = {int(c['seq_id']): list(c['plan']) for c in layer1}
        ready_reclaim_blocks = {int(c['seq_id']): int(c['releasable_blocks']) for c in layer1}
        if not layer1:
            if hasattr(sched, 'record_p2_ready_selection'):
                sched.record_p2_ready_selection(0, 0, int(target_reclaim_blocks), 0, str(ready_stop_reason))
            sched.record_p2_attempt(attempted=False, success=False, no_candidate=False, candidate_count=int(candidate_count))
            return {'attempted': 0, 'success': 0, 'no_candidate': 0, 'offloaded': 0, 'candidate_count': int(candidate_count), 'offload_budget': 0}
        offload_budget = self._compute_p2_offload_budget(ready_plans, decode_plans, ready_reclaim_blocks, target_reclaim_blocks=int(target_reclaim_blocks))
        offloader.set_step_transfer_budgets(offload_budget=offload_budget)

        attempted = False
        offloaded = 0
        ready_offloaded = 0
        ready_offload_steps = 0
        issued_async = False
        for cand in list(layer1):
            if offloaded >= offload_budget:
                break
            sid = int(cand['seq_id'])
            chunk = list(cand['plan'])
            if not chunk:
                continue
            if int(offloaded) + int(len(chunk)) > int(offload_budget):
                ready_stop_reason = "block_cap_reached"
                break
            attempted = True
            try:
                _ = offloader.offload_sequence_blocks(int(sid), chunk)
                ok = bool(offloader._sequence_has_state(int(sid), OffloadState.OFFLOAD_INFLIGHT))
                if not ok:
                    ok = bool(offloader._try_finalize_offload(int(sid), force_sync=False))
            except Exception:
                ok = False
            if not ok:
                continue
            issued_async = True
            offloaded += len(chunk)
            ready_offloaded += len(chunk)
            ready_offload_steps += 1
            self._record_post_cleanup_cuda_state(active_seq_count=max(1, len(protected_seq_ids)))
        if hasattr(sched, 'record_p2_ready_selection'):
            sched.record_p2_ready_selection(int(len(layer1)), int(planned_ready_blocks), int(target_reclaim_blocks), int(ready_offloaded), str(ready_stop_reason))
        success = int(sched.pool.n_free) >= target_free
        if cuda_pressure:
            success = bool(success and not bool(sched.p2_cuda_pressure_signal(has_decode_work=has_decode_work)))
        progress_made = bool(
            int(offloaded) > 0 and (
                int(sched.pool.n_free) > int(free_before)
                or float(sched.cuda_free_post_cleanup_gb()) > float(cuda_free_before)
            )
        )
        effective_success = bool(success or progress_made or issued_async)
        if bool(attempted):
            sched.record_p2_gain_result(bool(effective_success))
        if int(ready_offloaded) > 0:
            sched.record_p2_ready_offload_blocks(int(ready_offloaded), int(ready_offload_steps))
        sched.record_p2_attempt(
            attempted=attempted,
            success=bool(effective_success),
            no_candidate=bool(no_candidate),
            candidate_count=int(candidate_count),
        )
        return {
            'attempted': int(attempted),
            'success': int(effective_success),
            'no_candidate': int(no_candidate),
            'offloaded': int(offloaded),
            'ready_offloaded': int(ready_offloaded),
            'candidate_count': int(candidate_count),
            'offload_budget': int(offload_budget),
        }

    def _pending_request_count(self) -> int:
        return len(self.waiting_queue) + len(self.prefill_active) + len(self.ready_decode) + len(self.decode_active)

    def _kv_admission_prompt_resident_blocks_for_len(self, prompt_tokens: int) -> int:
        block_size = max(1, int(self.scheduler.pool.B))
        prompt_blocks = max(0, int(math.ceil(float(max(0, int(prompt_tokens))) / float(block_size))))
        if prompt_blocks <= 0:
            return 0
        snapkv = getattr(self.scheduler, 'snapkv', None)
        retain_budget_tokens = int(getattr(snapkv, 'retain_budget_tokens', 0) or 0)
        if retain_budget_tokens > 0:
            retained_prompt_tokens = min(max(0, int(prompt_tokens)), max(1, int(retain_budget_tokens)))
            return max(1, int(math.ceil(float(retained_prompt_tokens) / float(block_size))))
        retain_ratio = float(getattr(snapkv, 'retain_ratio', 1.0) or 1.0)
        return max(1, int(math.ceil(float(prompt_blocks) * max(0.0, min(1.0, retain_ratio)))))

    def _kv_admission_pending_prompt_blocks(self) -> int:
        # Waiting/PREFILLING requests have not committed their selected prompt KV
        # into the physical pool yet, so pool.n_free does not reflect their future
        # fixed-budget prompt footprint. Count them here to avoid admitting too many
        # requests at once in continuous refill. READY/DECODING requests are already
        # materialized in the pool, so counting them would double count.
        total = 0
        for req in list(getattr(self, '_requests', {}).values()):
            if req is None or req.state not in (REQ_WAITING_PREFILL, REQ_PREFILLING):
                continue
            total += int(self._kv_admission_prompt_resident_blocks_for_len(int(getattr(req, 'prompt_token_len', 0) or 0)))
        return int(total)

    def _kv_admission_output_reserve_tokens_for_remaining(self, remaining_tokens: int) -> int:
        remaining = max(0, int(remaining_tokens))
        reserve = max(0, int(getattr(self, 'kv_admission_output_reserve_tokens', 0) or 0))
        if reserve > 0:
            return min(remaining, reserve)
        return remaining

    def _kv_admission_pending_output_blocks(self) -> int:
        block_size = max(1, int(self.scheduler.pool.B))
        total = 0
        for req in list(getattr(self, '_requests', {}).values()):
            if req is None or req.state in (REQ_DONE, REQ_FAILED):
                continue
            remaining = max(0, int(req.max_new_tokens) - int(len(req.generated_tokens or [])))
            reserved = int(self._kv_admission_output_reserve_tokens_for_remaining(remaining))
            total += int(math.ceil(float(reserved) / float(block_size))) if reserved > 0 else 0
        return int(total)

    def can_admit_prompt_tokens(self, prompt_tokens: int, max_new_tokens: int = 0) -> Dict[str, int]:
        block_size = max(1, int(self.scheduler.pool.B))
        prompt_blocks = max(0, int(math.ceil(float(max(0, int(prompt_tokens))) / float(block_size))))
        prompt_resident_blocks = int(self._kv_admission_prompt_resident_blocks_for_len(int(prompt_tokens)))
        output_reserve_tokens = int(self._kv_admission_output_reserve_tokens_for_remaining(max(0, int(max_new_tokens))))
        new_output_blocks = max(0, int(math.ceil(float(output_reserve_tokens) / float(block_size))))
        request_blocks = int(prompt_resident_blocks) + int(new_output_blocks)
        pending_output_blocks = int(self._kv_admission_pending_output_blocks())
        pending_prompt_blocks = int(self._kv_admission_pending_prompt_blocks())
        reserve_blocks = int(self.scheduler.p2_low_threshold()) if bool(getattr(self, "kv_admission_include_low_watermark", True)) else 0
        margin_blocks = int(self.kv_admission_margin_blocks)
        workload_demand_blocks = int(request_blocks) + int(pending_prompt_blocks) + int(pending_output_blocks)
        total_required_blocks = int(workload_demand_blocks) + int(reserve_blocks) + int(margin_blocks)
        free_blocks = int(self.scheduler.pool.n_free)
        bootstrap = self._pending_request_count() <= 0
        allowed = (not bool(self.kv_admission_enabled)) or bool(bootstrap) or free_blocks >= total_required_blocks
        self._online_kv_admission_last_free_blocks = int(free_blocks)
        # required_blocks is the workload demand. The admission threshold is
        # total_required_blocks = demand + reserved low-watermark + margin.
        self._online_kv_admission_last_required_blocks = int(workload_demand_blocks)
        self._online_kv_admission_last_workload_demand_blocks = int(workload_demand_blocks)
        self._online_kv_admission_last_reserved_blocks = int(reserve_blocks)
        self._online_kv_admission_last_margin_blocks = int(margin_blocks)
        self._online_kv_admission_last_total_required_blocks = int(total_required_blocks)
        self._online_kv_admission_last_prompt_blocks = int(prompt_blocks)
        self._online_kv_admission_last_prompt_resident_blocks = int(prompt_resident_blocks)
        self._online_kv_admission_last_request_blocks = int(request_blocks)
        self._online_kv_admission_last_pending_prompt_blocks = int(pending_prompt_blocks)
        self._online_kv_admission_last_pending_output_blocks = int(pending_output_blocks)
        self._online_kv_admission_last_output_reserve_tokens = int(output_reserve_tokens)
        self._online_kv_admission_last_output_reserve_blocks = int(new_output_blocks)
        self._online_kv_admission_last_allowed = int(bool(allowed))
        if bool(self.kv_admission_enabled) and not bool(allowed):
            self._online_kv_admission_blocked_requests += 1
            step = int(getattr(self, '_online_step', 0))
            if int(self._online_kv_admission_last_blocked_step) != step:
                self._online_kv_admission_blocked_steps += 1
                self._online_kv_admission_last_blocked_step = step
        return {
            'allowed': int(bool(allowed)),
            'enabled': int(bool(self.kv_admission_enabled)),
            'free_blocks': int(free_blocks),
            'required_blocks': int(workload_demand_blocks),
            'workload_demand_blocks': int(workload_demand_blocks),
            'reserved_blocks': int(reserve_blocks),
            'margin_blocks': int(margin_blocks),
            'total_required_blocks': int(total_required_blocks),
            'prompt_blocks': int(prompt_blocks),
            'prompt_resident_blocks': int(prompt_resident_blocks),
            'request_blocks': int(request_blocks),
            'pending_prompt_blocks': int(pending_prompt_blocks),
            'pending_output_blocks': int(pending_output_blocks),
            'output_reserve_tokens': int(output_reserve_tokens),
            'output_reserve_blocks': int(new_output_blocks),
            'include_low_watermark': int(bool(getattr(self, 'kv_admission_include_low_watermark', True))),
            'bootstrap': int(bool(bootstrap)),
        }

    def has_pending_requests(self) -> bool:
        return self._pending_request_count() > 0

    def _apply_decode_backpressure(self, reason: str, count_event: bool = True):
        cap_now = max(self.decode_active_cap_min, int(self._online_decode_active_cap or self.decode_active_cap_min))
        cap_next = max(
            self.decode_active_cap_min,
            int(math.floor(float(cap_now) * self.decode_active_cap_downscale)),
        )
        if cap_next < cap_now:
            self._online_decode_active_cap = cap_next
        else:
            self._online_decode_active_cap = max(self.decode_active_cap_min, cap_now - 1)
        if count_event:
            self._online_decode_backpressure_events += 1
        self._online_prefill_pause_steps = max(
            self._online_prefill_pause_steps,
            int(self.decode_backpressure_pause_steps),
        )

    def _apply_prefill_backpressure(self, reason: str):
        self._online_prefill_backpressure_events += 1
        if reason == "prefill_batch_failed":
            self._online_prefill_batch_failed_steps += 1
        elif reason == "prefill_chunk_failed":
            self._online_prefill_chunk_failed_steps += 1
        elif reason == "prefill_activate_failed":
            self._online_prefill_activate_failed_steps += 1
        self._online_prefill_pause_steps = max(
            self._online_prefill_pause_steps,
            int(self.decode_backpressure_pause_steps),
        )

    def _sample_cuda_alloc_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        try:
            return float(torch.cuda.memory_allocated()) / 1024**3
        except Exception:
            return 0.0

    def _note_prefill_peak(self, attr_name: str) -> None:
        try:
            cur = float(self._sample_cuda_alloc_gb())
            prev = float(getattr(self, attr_name, 0.0) or 0.0)
            if cur > prev:
                setattr(self, attr_name, cur)
        except Exception:
            pass

    @staticmethod
    def _is_retryable_memory_error(exc: Exception) -> bool:
        if isinstance(exc, MemoryError):
            return True
        torch_oom = getattr(torch, 'OutOfMemoryError', None)
        if torch_oom is not None and isinstance(exc, torch_oom):
            return True
        cuda_mod = getattr(torch, 'cuda', None)
        cuda_oom = getattr(cuda_mod, 'OutOfMemoryError', None) if cuda_mod is not None else None
        if cuda_oom is not None and isinstance(exc, cuda_oom):
            return True
        msg = str(exc).lower()
        if 'cuda out of memory' in msg:
            return True
        if 'out of memory' in msg:
            return True
        if 'cublas_status_alloc_failed' in msg:
            return True
        if 'tried to allocate' in msg and ('cuda' in msg or 'memory' in msg):
            return True
        return False

    def _record_decode_memory_estimate(self, est_maxlen_gb: float, est_sumlen_gb: float) -> None:
        est_maxlen_gb = float(max(0.0, est_maxlen_gb))
        est_sumlen_gb = float(max(0.0, est_sumlen_gb))
        self._online_decode_memory_est_peak_maxlen_gb = max(float(self._online_decode_memory_est_peak_maxlen_gb), est_maxlen_gb)
        self._online_decode_memory_est_peak_sumlen_gb = max(float(self._online_decode_memory_est_peak_sumlen_gb), est_sumlen_gb)
        self._online_decode_memory_est_peak_max_gb = max(float(self._online_decode_memory_est_peak_max_gb), est_maxlen_gb)

    def _record_decode_memory_cap_event(self, capped_batch_size: int, est_maxlen_gb: float, est_sumlen_gb: float) -> None:
        self._online_decode_memory_cap_events += 1
        capped_batch_size = max(1, int(capped_batch_size))
        if self._online_decode_memory_cap_min_batch > 0:
            self._online_decode_memory_cap_min_batch = min(int(self._online_decode_memory_cap_min_batch), capped_batch_size)
        else:
            self._online_decode_memory_cap_min_batch = capped_batch_size
        self._record_decode_memory_estimate(est_maxlen_gb, est_sumlen_gb)

    def _logical_seq_len_for_decode_guard(self, sid: int) -> int:
        sched = self.scheduler
        req = self._requests.get(int(sid))
        req_logical_len = 0
        if req is not None:
            prompt_len = int(getattr(req, 'prompt_token_len', 0) or 0)
            generated_len = len(getattr(req, 'generated_tokens', []) or [])
            decode_steps = int(getattr(req, 'decode_steps', 0) or 0)
            req_logical_len = max(prompt_len + generated_len, prompt_len + decode_steps, prompt_len)
        entry_logical_len = 0
        entry_seq_len = 0
        entry = sched.offloader._get_seq_layer0_entry(int(sid))
        if entry is not None:
            entry_logical_len = int(getattr(entry, 'logical_seq_len', 0) or 0)
            entry_seq_len = int(getattr(entry, 'seq_len', 0) or 0)
        return max(1, req_logical_len, entry_logical_len, entry_seq_len)

    @staticmethod
    def _decode_length_bucket_index(logical_len: int) -> int:
        logical_len = int(max(0, logical_len))
        if logical_len < 2048:
            return 0
        if logical_len < 4096:
            return 1
        if logical_len < 8192:
            return 2
        if logical_len < 16384:
            return 3
        if logical_len < 32768:
            return 4
        return 5

    @staticmethod
    def _median_decode_guard_len(lengths: Sequence[int]) -> float:
        vals = sorted(int(max(0, x)) for x in lengths)
        if not vals:
            return 0.0
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            return float(vals[mid])
        return 0.5 * float(vals[mid - 1] + vals[mid])

    def _bucket_decode_batch_by_length(self, seq_ids: List[int]) -> List[List[int]]:
        batch = [int(sid) for sid in seq_ids]
        disable_bucket = str(os.environ.get("KV_MIDDLEWARE_DISABLE_DECODE_LENGTH_BUCKET", "")).strip().lower()
        if disable_bucket in {"1", "true", "yes", "on"}:
            return [batch]
        if len(batch) < 4:
            return [batch]
        logical_seq_lens = [self._logical_seq_len_for_decode_guard(int(sid)) for sid in batch]
        if len(logical_seq_lens) != len(batch) or not logical_seq_lens:
            return [batch]
        max_len = max(int(x) for x in logical_seq_lens)
        median_len = float(self._median_decode_guard_len(logical_seq_lens))
        trigger_ratio = float(max_len) / max(1.0, median_len)
        if trigger_ratio < 2.0:
            return [batch]
        buckets: Dict[int, List[int]] = {}
        first_pos: Dict[int, int] = {}
        for pos, (sid, logical_len) in enumerate(zip(batch, logical_seq_lens)):
            bucket_idx = self._decode_length_bucket_index(int(logical_len))
            buckets.setdefault(bucket_idx, []).append(int(sid))
            first_pos.setdefault(bucket_idx, int(pos))
        if len(buckets) <= 1:
            return [batch]
        ordered_bucket_ids = sorted(first_pos.keys(), key=lambda idx: int(first_pos[idx]))
        out: List[List[int]] = []
        micro_batch = int(self.decode_micro_batch_size or 0)
        for bucket_idx in ordered_bucket_ids:
            group = list(buckets.get(bucket_idx, []) or [])
            if not group:
                continue
            if micro_batch > 0:
                for start in range(0, len(group), micro_batch):
                    chunk = group[start:start + micro_batch]
                    if not chunk:
                        continue
                    out.append(chunk)
                    self._online_decode_length_bucket_subbatch_count += 1
                    if len(chunk) == 1:
                        self._online_decode_length_bucket_singleton_count += 1
            else:
                out.append(group)
                self._online_decode_length_bucket_subbatch_count += 1
                if len(group) == 1:
                    self._online_decode_length_bucket_singleton_count += 1
        if len(out) <= 1:
            return [batch]
        self._online_decode_length_bucketed_steps += 1
        self._online_decode_length_bucket_max_trigger_ratio = max(
            float(self._online_decode_length_bucket_max_trigger_ratio),
            float(trigger_ratio),
        )
        return out

    def _estimate_decode_rebuild_peak_gb(self, seq_ids: List[int]) -> Tuple[float, float]:
        if not seq_ids:
            return 0.0, 0.0
        sched = self.scheduler
        logical_seq_lens = [self._logical_seq_len_for_decode_guard(int(sid)) for sid in seq_ids]
        if not logical_seq_lens:
            return 0.0, 0.0
        dtype = getattr(self.model, 'dtype', torch.float16)
        try:
            dtype_bytes = int(torch.empty((), dtype=dtype).element_size())
        except Exception:
            dtype_bytes = 2
        bytes_per_token_all_layers = (
            int(sched.num_layers)
            * int(sched.num_kv_heads)
            * int(sched.head_dim)
            * 2
            * int(dtype_bytes)
        )
        batch = len(logical_seq_lens)
        lmax = max(logical_seq_lens)
        lsum = sum(logical_seq_lens)
        peak_factor = float(self._online_decode_memory_aware_peak_factor or 1.25)
        base_overhead_gb = 0.25
        scale = float(bytes_per_token_all_layers) / float(1024 ** 3)
        est_maxlen = float(batch * lmax) * scale * peak_factor + base_overhead_gb
        est_sumlen = float(lsum) * scale * peak_factor + base_overhead_gb
        return float(est_maxlen), float(est_sumlen)

    def _maybe_apply_decode_memory_guard(self, seq_ids: List[int]) -> Tuple[List[int], List[int]]:
        batch = list(seq_ids)
        disable_guard = str(os.environ.get("KV_MIDDLEWARE_DISABLE_DECODE_MEMORY_GUARD", "")).strip().lower()
        if disable_guard in {"1", "true", "yes", "on"}:
            return batch, []
        if len(batch) <= 1 or not torch.cuda.is_available():
            return batch, []
        sched = self.scheduler
        if (not bool(getattr(sched, '_p2_managed_active', False))) and int(sched.pool.n_free) > int(sched.p2_low_threshold()):
            return batch, []
        source_batch = int(len(batch))
        self._online_decode_memory_guard_source_batch_last = source_batch
        self._online_guard_source_batch_max = max(int(self._online_guard_source_batch_max), source_batch)
        total_gb = 0.0
        if self._online_cuda_total_bytes:
            total_gb = float(self._online_cuda_total_bytes) / float(1024 ** 3)
        if total_gb <= 0.0:
            try:
                total_gb = float(torch.cuda.get_device_properties(0).total_memory) / float(1024 ** 3)
            except Exception:
                total_gb = 0.0
        reserve_gb = max(
            float(self._online_decode_memory_aware_margin_gb or 1.0),
            0.10 * max(0.0, total_gb),
        )
        hard_guard_gb = float(self.decode_memory_guard_hard_guard_gb or 0.0)
        free_gb = float(self._sample_cuda_free_gb())
        budget_gb = max(0.0, free_gb - reserve_gb)
        self._online_decode_memory_guard_free_gb_last = float(free_gb)
        self._online_decode_memory_guard_budget_gb_last = float(budget_gb)
        self._online_decode_memory_guard_reserve_gb_last = float(reserve_gb)
        self._online_decode_memory_guard_hard_guard_gb = float(hard_guard_gb)
        if free_gb >= (reserve_gb + hard_guard_gb):
            return batch, []
        est_maxlen_gb, est_sumlen_gb = self._estimate_decode_rebuild_peak_gb(batch)
        self._record_decode_memory_estimate(est_maxlen_gb, est_sumlen_gb)
        if est_maxlen_gb <= budget_gb:
            return batch, []
        gc.collect()
        torch.cuda.empty_cache()
        self._record_post_cleanup_cuda_state(active_seq_count=max(1, len(batch)))
        free_gb = float(self._sample_cuda_free_gb())
        budget_gb = max(0.0, free_gb - reserve_gb)
        self._online_decode_memory_guard_free_gb_last = float(free_gb)
        self._online_decode_memory_guard_budget_gb_last = float(budget_gb)
        self._online_decode_memory_guard_reserve_gb_last = float(reserve_gb)
        self._online_decode_memory_guard_hard_guard_gb = float(hard_guard_gb)
        if est_maxlen_gb <= budget_gb:
            return batch, []
        per_seq_est_gb = est_maxlen_gb / max(1, len(batch))
        if per_seq_est_gb <= 0.0:
            return batch, []
        target_batch = int(math.floor(budget_gb / per_seq_est_gb))
        target_batch = max(1, min(len(batch), target_batch))
        self._online_decode_memory_guard_target_batch_last = int(target_batch)
        self._online_guard_seen_count += 1
        if self._online_guard_target_batch_min > 0:
            self._online_guard_target_batch_min = min(int(self._online_guard_target_batch_min), int(target_batch))
        else:
            self._online_guard_target_batch_min = int(target_batch)
        if target_batch >= len(batch):
            return batch, []
        self._online_guard_effective_shrink_count += 1
        shrink_delta = int(len(batch) - target_batch)
        strong_gate = max(2, int(math.ceil(0.1 * float(len(batch)))))
        if shrink_delta >= strong_gate:
            self._online_guard_strong_shrink_count += 1
        cand = batch[:target_batch]
        est_maxlen_gb, est_sumlen_gb = self._estimate_decode_rebuild_peak_gb(cand)
        self._record_decode_memory_cap_event(target_batch, est_maxlen_gb, est_sumlen_gb)
        return cand, batch[target_batch:]

    @staticmethod
    def _slice_past_key_values(past_key_values: Any, batch_idx: int):
        if past_key_values is None:
            return None
        sliced = []
        for layer_kv in past_key_values:
            if not isinstance(layer_kv, (tuple, list)) or len(layer_kv) < 2:
                return None
            k_full, v_full = layer_kv[0], layer_kv[1]
            if not (torch.is_tensor(k_full) and torch.is_tensor(v_full)):
                return None
            if k_full.dim() < 4 or v_full.dim() < 4:
                return None
            if batch_idx >= k_full.shape[0] or batch_idx >= v_full.shape[0]:
                return None
            sliced.append(
                (
                    k_full[batch_idx: batch_idx + 1].detach(),
                    v_full[batch_idx: batch_idx + 1].detach(),
                )
            )
        return CustomKVCache().from_tuple(tuple(sliced))

    @staticmethod
    def _past_kv_len(past_key_values: Any) -> int:
        if past_key_values is None:
            return 0
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                return int(get_seq_length())
            except Exception:
                pass
        try:
            if len(past_key_values) <= 0:
                return 0
        except Exception:
            return 0
        try:
            layer0 = past_key_values[0]
        except Exception:
            return 0
        if not isinstance(layer0, (tuple, list)) or len(layer0) < 2:
            return 0
        k0 = layer0[0]
        if not torch.is_tensor(k0) or k0.dim() < 4:
            return 0
        return int(k0.shape[2])
    @staticmethod
    def _merge_past_key_values(past_list: Sequence[Any]):
        if not past_list:
            return None
        num_layers = len(past_list[0])
        merged = []
        for layer_id in range(num_layers):
            k_parts = []
            v_parts = []
            for pkv in past_list:
                k_i, v_i = pkv[layer_id][0], pkv[layer_id][1]
                k_parts.append(k_i)
                v_parts.append(v_i)
            merged.append((torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)))
        return CustomKVCache().from_tuple(tuple(merged))

    def _current_prefill_admit_cap(self) -> int:
        # Admit cap controls how many requests may stay in PREFILLING state.
        # Batch size only controls how many of them are processed together in one
        # prefill batch; tying the two artificially suppresses overlap and makes
        # it hard for online workloads to form resident decode pressure.
        return max(1, int(self.max_prefill_active))

    def _prefill_prompt_bucket(self, prompt_len: int) -> str:
        length = max(0, int(prompt_len))
        if length < int(self.online_prefill_short_threshold_tokens):
            return "short"
        if length < int(self.online_prefill_mid_threshold_tokens):
            return "mid"
        return "long"

    def _prefill_bucket_cap(self, bucket: str) -> int:
        if bucket == "short":
            return max(1, int(self.online_prefill_cap_short))
        if bucket == "mid":
            return max(1, int(self.online_prefill_cap_mid))
        return max(1, int(self.online_prefill_cap_long))

    def _prefill_active_bucket_counts(self) -> Dict[str, int]:
        counts = {"short": 0, "mid": 0, "long": 0}
        for rid in list(self.prefill_active):
            req = self._requests.get(int(rid))
            if req is None or req.state != REQ_PREFILLING:
                continue
            bucket = self._prefill_prompt_bucket(int(getattr(req, 'prompt_token_len', 0) or 0))
            counts[bucket] = int(counts.get(bucket, 0)) + 1
        return counts

    def _prefill_active_prompt_token_sum(self) -> int:
        total = 0
        for rid in list(self.prefill_active):
            req = self._requests.get(int(rid))
            if req is None or req.state != REQ_PREFILLING:
                continue
            total += max(0, int(getattr(req, 'prompt_token_len', 0) or 0))
        return int(total)

    def _record_prefill_admission_decision(
        self,
        req: Optional[OnlineRequest],
        allowed: bool,
        reason: str,
        bucket: str,
        cap: int,
        counts: Dict[str, int],
        free_gb: float,
        active_tokens: int = 0,
        projected_tokens: int = 0,
        token_budget: int = 0,
    ) -> None:
        self._online_prefill_admission_last_reason = str(reason or "")
        self._online_prefill_admission_last_prompt_len = int(getattr(req, 'prompt_token_len', 0) or 0) if req is not None else 0
        self._online_prefill_admission_last_bucket = str(bucket or "")
        self._online_prefill_admission_last_cuda_free_gb = float(free_gb)
        self._online_prefill_admission_last_active_short = int(counts.get("short", 0))
        self._online_prefill_admission_last_active_mid = int(counts.get("mid", 0))
        self._online_prefill_admission_last_active_long = int(counts.get("long", 0))
        self._online_prefill_admission_last_cap = int(cap)
        self._online_prefill_admission_last_active_tokens = int(active_tokens)
        self._online_prefill_admission_last_projected_tokens = int(projected_tokens)
        self._online_prefill_admission_last_token_budget = int(token_budget)
        self._online_prefill_admission_last_allowed = int(bool(allowed))
        if bool(self.online_prefill_admission_enabled) and not bool(allowed) and str(reason or "") == "token_budget":
            self._online_prefill_admission_token_budget_blocked_requests += 1
            step = int(getattr(self, '_online_step', 0))
            if int(self._online_prefill_admission_token_budget_last_blocked_step) != step:
                self._online_prefill_admission_token_budget_blocked_steps += 1
                self._online_prefill_admission_token_budget_last_blocked_step = step
        if bool(self.online_prefill_admission_enabled) and not bool(allowed):
            self._online_prefill_admission_blocked_requests += 1
            step = int(getattr(self, '_online_step', 0))
            if int(self._online_prefill_admission_last_blocked_step) != step:
                self._online_prefill_admission_blocked_steps += 1
                self._online_prefill_admission_last_blocked_step = step

    def _prefill_admission_allows(self, req: OnlineRequest) -> bool:
        if not bool(self.online_prefill_admission_enabled):
            return True
        prompt_len = max(0, int(getattr(req, 'prompt_token_len', 0) or 0))
        counts = self._prefill_active_bucket_counts()
        bucket = self._prefill_prompt_bucket(prompt_len)
        cap = self._prefill_bucket_cap(bucket)
        free_gb = float(self._sample_cuda_free_gb())
        active_tokens = int(self._prefill_active_prompt_token_sum())
        projected_tokens = int(active_tokens + prompt_len)
        token_budget = max(0, int(self.online_prefill_active_token_budget))
        if int(counts.get(bucket, 0)) >= int(cap):
            self._record_prefill_admission_decision(req, False, "bucket_cap", bucket, cap, counts, free_gb, active_tokens, projected_tokens, token_budget)
            return False
        # Token budget lets many tiny prompts prefill together, but prevents a burst
        # of several 3-8K prompts from holding dense prefill KV at the same time. A
        # single long prompt is still admitted when no other prefill is active.
        if token_budget > 0 and active_tokens > 0 and projected_tokens > token_budget:
            self._record_prefill_admission_decision(req, False, "token_budget", bucket, cap, counts, free_gb, active_tokens, projected_tokens, token_budget)
            return False
        # Headroom gate only throttles additional pressure. If the engine is otherwise
        # idle, admit one request so a single prompt is allowed to make progress or
        # fail explicitly instead of waiting forever.
        has_prefill_pressure = bool(self.prefill_active or self.ready_decode or self.decode_active)
        if has_prefill_pressure and free_gb < float(self.online_prefill_cuda_headroom_gb):
            self._record_prefill_admission_decision(req, False, "cuda_headroom", bucket, cap, counts, free_gb, active_tokens, projected_tokens, token_budget)
            return False
        self._record_prefill_admission_decision(req, True, "allowed", bucket, cap, counts, free_gb, active_tokens, projected_tokens, token_budget)
        return True

    def _prefill_next_chunk_cap_after_oom(self, chunk: int) -> int:
        next_cap = max(1, int(chunk) // 2)
        if bool(self.online_prefill_admission_enabled) and next_cap < int(self.online_prefill_min_effective_chunk):
            next_cap = max(1, int(self.online_prefill_min_effective_chunk))
            self._online_prefill_chunk_floor_pause_steps += 1
        self._online_prefill_chunk_floor_last_chunk_cap = int(next_cap)
        return int(next_cap)

    def _prefill_is_paused(self) -> bool:
        if self._online_prefill_pause_steps > 0:
            return True
        return bool(self.decode_active and int(self.scheduler.pool.n_free) <= int(self.scheduler.pool.N_wm_low))

    def submit(
        self,
        prompt: str,
        request_id: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> int:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if self._pending_request_count() >= self.max_waiting_requests:
            raise RuntimeError(
                f"Too many pending requests ({self._pending_request_count()}), "
                f"max_waiting_requests={self.max_waiting_requests}"
            )

        rid = int(self._next_request_id if request_id is None else request_id)
        if rid in self._requests:
            raise ValueError(f"request_id already exists: {rid}")
        if request_id is None:
            self._next_request_id += 1
        else:
            self._next_request_id = max(self._next_request_id, rid + 1)

        submit_time_s = time.perf_counter()
        token_ids = self.tokenizer(prompt, return_tensors='pt', truncation=False).input_ids.squeeze(0).cpu()
        req = OnlineRequest(
            request_id=rid,
            prompt=prompt,
            max_new_tokens=max(1, int(self.max_new_tokens if max_new_tokens is None else max_new_tokens)),
            arrival_step=self._online_step,
            state=REQ_WAITING_PREFILL,
            input_ids_cpu=token_ids,
            prompt_token_len=int(token_ids.numel()),
            submit_time_s=float(submit_time_s),
        )
        self._requests[rid] = req
        self.waiting_queue.append(rid)
        if self._online_total_t0 <= 0.0:
            self._online_total_t0 = submit_time_s
        return rid

    def collect_finished(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        while self.finished_queue:
            out.append(self.finished_queue.popleft())
        return out

    @staticmethod
    def _build_decode_attention_mask(seq_lens: List[int], device: torch.device) -> torch.Tensor:
        if not seq_lens:
            return torch.empty((0, 0), dtype=torch.long, device=device)
        max_len = max(seq_lens)
        mask = torch.zeros((len(seq_lens), max_len + 1), dtype=torch.long, device=device)
        for i, l in enumerate(seq_lens):
            if l > 0:
                mask[i, :l] = 1
            mask[i, max_len] = 1
        return mask

    @staticmethod
    def _build_decode_position_ids(logical_seq_lens: List[int], device: torch.device) -> torch.Tensor:
        if not logical_seq_lens:
            return torch.empty((0, 1), dtype=torch.long, device=device)
        vals = [max(0, int(x)) for x in logical_seq_lens]
        return torch.tensor(vals, dtype=torch.long, device=device).unsqueeze(1)

    def _prefill_batch(
        self,
        batch_items: Sequence[Tuple[int, torch.Tensor]],
        all_last_logits: List[torch.Tensor],
    ):
        sched = self.scheduler
        seq_ids = [sid for sid, _ in batch_items]
        seq_lens = [int(ids.numel()) for _, ids in batch_items]
        batch = len(batch_items)
        max_len = max(seq_lens)

        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long, device='cuda')
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long, device='cuda')

        for i, (_, ids_cpu) in enumerate(batch_items):
            l = seq_lens[i]
            input_ids[i, :l] = ids_cpu.to('cuda')
            attention_mask[i, :l] = 1

        if HAS_FLASH_ATTN:
            _ = build_cu_seqlens(seq_lens, device=input_ids.device)

        sched.active_seqs = seq_ids
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

        for i, sid in enumerate(seq_ids):
            l = seq_lens[i]
            all_last_logits[sid] = out.logits[i: i + 1, l - 1: l, :].detach()
            sched.capture_prefill_kv(
                seq_id=sid,
                past_key_values=out.past_key_values,
                batch_idx=i,
                seq_len=l,
            )

        del out, input_ids, attention_mask
        torch.cuda.empty_cache()

    def _prefill_single(self, sid: int, ids_cpu: torch.Tensor, all_last_logits: List[torch.Tensor]):
        sched = self.scheduler
        sched.active_seqs = [sid]

        enc = ids_cpu.unsqueeze(0).to('cuda')
        L = enc.shape[1]
        if L > self.chunk_size:
            last_logits, past_kv = self.chunker.run(self.model, enc, self.chunk_size)
            sched.capture_prefill_kv(seq_id=sid, past_key_values=past_kv, batch_idx=0, seq_len=L)
        else:
            with torch.no_grad():
                out = self.model(enc, use_cache=True)
            last_logits = out.logits[:, -1:, :]
            sched.capture_prefill_kv(seq_id=sid, past_key_values=out.past_key_values, batch_idx=0, seq_len=L)
            del out

        all_last_logits[sid] = last_logits.detach()
        del enc
        torch.cuda.empty_cache()

    def _prefill_batch_online(
        self,
        batch_items: Sequence[Tuple[int, torch.Tensor]],
        past_key_values_list: Optional[Sequence[Any]] = None,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, Any]]:
        sched = self.scheduler
        seq_ids = [sid for sid, _ in batch_items]
        seq_lens = [int(ids.numel()) for _, ids in batch_items]
        batch = len(batch_items)
        max_len = max(seq_lens)
        merged_past = None
        if past_key_values_list is not None:
            if len(past_key_values_list) != batch:
                raise RuntimeError("past_key_values_list size mismatch for prefill batch")
            if any(p is None for p in past_key_values_list):
                raise RuntimeError("past_key_values_list contains None")
            past_lens = [self._past_kv_len(p) for p in past_key_values_list]
            if len(set(past_lens)) != 1:
                raise RuntimeError("prefill batch requires equal past length")
            merged_past = self._merge_past_key_values(past_key_values_list)
            self._note_prefill_peak('_online_prefill_batch_merge_peak_gb')

        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long, device='cuda')
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long, device='cuda')
        for i, (_, ids_cpu) in enumerate(batch_items):
            l = seq_lens[i]
            input_ids[i, :l] = ids_cpu.to('cuda')
            attention_mask[i, :l] = 1
        self._note_prefill_peak('_online_prefill_batch_input_peak_gb')

        if HAS_FLASH_ATTN:
            _ = build_cu_seqlens(seq_lens, device=input_ids.device)

        sched.phase = PhaseState.PREFILL
        sched.active_seqs = seq_ids
        with torch.no_grad():
            if merged_past is None:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            else:
                out = self.model(input_ids=input_ids, past_key_values=merged_past, use_cache=True)
        self._note_prefill_peak('_online_prefill_batch_forward_peak_gb')

        last_logits_map: Dict[int, torch.Tensor] = {}
        past_kv_map: Dict[int, Any] = {}
        for i, sid in enumerate(seq_ids):
            l = seq_lens[i]
            last_logits_map[sid] = out.logits[i: i + 1, l - 1: l, :].detach()
            past_kv_i = self._slice_past_key_values(out.past_key_values, i)
            if past_kv_i is None:
                raise RuntimeError(f"failed_to_slice_past_key_values seq={sid}")
            past_kv_map[sid] = past_kv_i
        self._note_prefill_peak('_online_prefill_batch_slice_peak_gb')

        del out, input_ids, attention_mask, merged_past
        return last_logits_map, past_kv_map

    def _prefill_chunk_online(self, req: OnlineRequest, chunk_tokens: int) -> bool:
        if req.input_ids_cpu is None:
            raise RuntimeError(f"request {req.request_id} has no input_ids")
        if req.prefill_cursor >= req.prompt_token_len:
            return True

        sched = self.scheduler
        start = int(req.prefill_cursor)
        end = min(req.prompt_token_len, start + int(chunk_tokens))
        if end <= start:
            return req.prefill_cursor >= req.prompt_token_len

        chunk_ids = req.input_ids_cpu[start:end].unsqueeze(0).to('cuda')
        self._note_prefill_peak('_online_prefill_chunk_input_peak_gb')
        sched.phase = PhaseState.PREFILL
        sched.active_seqs = [req.request_id]
        with torch.no_grad():
            out = self.model(input_ids=chunk_ids, past_key_values=req.prefill_past_kv, use_cache=True)
        self._note_prefill_peak('_online_prefill_chunk_forward_peak_gb')

        req.prefill_past_kv = out.past_key_values
        req.prefill_cursor = end
        req.last_logits = out.logits[:, -1:, :].detach()

        del out, chunk_ids
        return req.prefill_cursor >= req.prompt_token_len

    def _activate_decoding_request(self, req: OnlineRequest, reset_decode_state: bool):
        sched = self.scheduler
        if req.last_logits is None:
            raise RuntimeError(f"request {req.request_id} missing prefill logits")
        has_staged_prefill = any(
            int(req.request_id) in sched.raw_kv_cache[layer_id]
            for layer_id in range(sched.num_layers)
        )
        if req.prefill_past_kv is None and not has_staged_prefill:
            raise RuntimeError(f'request {req.request_id} missing prefill cache')

        should_reset = bool(reset_decode_state and not req.decode_state_initialized)
        if req.prefill_past_kv is not None:
            prefill_past_kv = req.prefill_past_kv
            sched.capture_prefill_kv(
                seq_id=req.request_id,
                past_key_values=prefill_past_kv,
                batch_idx=0,
                seq_len=req.prompt_token_len,
            )
            # Once staged on CPU inside the scheduler, the original GPU cache can
            # be released before block-pool registration to reduce activation peak.
            req.prefill_past_kv = None
            del prefill_past_kv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        sched.post_prefill_compress([req.request_id], reset_decode_state=should_reset)
        self._sync_multigpu_after_prefill([req.request_id])
        req.decode_state_initialized = True

        req.prefill_past_kv = None
        req.state = REQ_READY_DECODE
        next_logits = req.last_logits[:, -1, :]
        next_tok = self._select_next_tokens(next_logits)
        first_tok = int(next_tok[0].item())
        req.current_token = first_tok
        req.generated_tokens.append(first_tok)
        if req.first_token_time_s <= 0.0:
            req.first_token_time_s = time.perf_counter()
            req.first_token_source = "prefill_activation"
        self._online_generated_tokens += 1
        if self._return_details_online:
            next_logprobs = torch.log_softmax(next_logits, dim=-1)
            req.token_logprobs.append(float(next_logprobs[0, first_tok].item()))
            del next_logprobs
        del next_logits, next_tok
        req.last_logits = None

        eos_hit = first_tok == self.tokenizer.eos_token_id
        max_hit = len(req.generated_tokens) >= int(req.max_new_tokens)
        if eos_hit or max_hit:
            req.eos_reached = bool(eos_hit)
            self._finish_request(req)
            return

        if req.request_id not in self.ready_decode:
            self.ready_decode.append(req.request_id)

    def _activate_ready_decode_requests(self) -> int:
        ready_ids = []
        for rid in self.ready_decode:
            req = self._requests.get(rid)
            if req is None or req.state != REQ_READY_DECODE:
                continue
            ready_ids.append(int(rid))
        self.ready_decode = deque(ready_ids)
        if not ready_ids:
            return 0

        total_candidates = len(self.decode_active) + len(ready_ids)
        cap_boot_now = self._initial_decode_active_cap(total_candidates)
        if self.max_decode_active_cap > 0:
            cap_boot_now = min(cap_boot_now, self.max_decode_active_cap)
        if self._online_decode_active_cap <= 0:
            self._online_decode_active_cap = max(self.decode_active_cap_min, cap_boot_now)
            self._online_decode_active_cap_boot = int(self._online_decode_active_cap)
            self._online_decode_active_cap_min_seen = int(self._online_decode_active_cap)
        else:
            self._online_decode_active_cap_boot = max(self._online_decode_active_cap_boot, int(cap_boot_now))

        cap_now = int(self._online_decode_active_cap)
        if self.max_decode_active_cap > 0:
            cap_now = min(cap_now, self.max_decode_active_cap)
        slots = max(0, cap_now - len(self.decode_active))
        if slots <= 0:
            return 0

        activated = 0
        deferred_ids: List[int] = []
        while self.ready_decode and activated < slots:
            rid = int(self.ready_decode.popleft())
            req = self._requests.get(rid)
            if req is None or req.state != REQ_READY_DECODE:
                continue
            if self._is_p2_offload_deferred(rid):
                deferred_ids.append(rid)
                continue
            req.state = REQ_DECODING
            if rid not in self.decode_active:
                self.decode_active.append(rid)
            activated += 1
        for rid in deferred_ids:
            if rid not in self.ready_decode:
                self.ready_decode.append(rid)
        if activated > 0:
            self._online_decode_started = True
        return activated

    def _queue_residency_stats(self, seq_ids: Sequence[int]) -> Dict[str, int]:
        seq_list = [int(x) for x in seq_ids]
        stats = {
            'seq_total': 0,
            'seq_on_gpu': 0,
            'seq_on_cpu': 0,
            'seq_pinned': 0,
            'seq_offload_inflight': 0,
            'seq_prefetch_inflight': 0,
            'seq_missing': 0,
            'logical_blocks': 0,
            'materialized_blocks': 0,
            'resident_blocks': 0,
        }
        if not seq_list:
            return stats

        page_table = self.scheduler.offloader.page_table
        B = max(1, int(self.scheduler.pool.B))
        for sid in seq_list:
            rep = None
            for layer_id in range(self.scheduler.num_layers):
                entry = page_table.get((sid, layer_id))
                if entry is not None:
                    rep = entry
                    break
            if rep is None:
                stats['seq_missing'] += 1
                continue
            stats['seq_total'] += 1
            self.scheduler.offloader._ensure_entry_maps(rep)
            logical_len = int(getattr(rep, 'logical_seq_len', 0) or rep.seq_len or 0)
            logical_blocks = int(math.ceil(float(max(0, logical_len)) / float(B)))
            materialized_blocks = int(getattr(rep, 'materialized_blocks', 0) or len(getattr(rep, 'gpu_block_map', []) or getattr(rep, 'block_ids', []) or []))
            stats['logical_blocks'] += logical_blocks
            stats['materialized_blocks'] += materialized_blocks
            state = rep.state
            resident_blocks = 0
            gpu_map = list(getattr(rep, 'gpu_block_map', []) or getattr(rep, 'block_ids', []) or [])
            if state == OffloadState.ON_GPU:
                stats['seq_on_gpu'] += 1
                resident_blocks = sum(1 for bid in gpu_map if int(bid) >= 0) if gpu_map else len(rep.block_ids)
            elif state == OffloadState.ON_CPU:
                stats['seq_on_cpu'] += 1
                resident_blocks = 0
            elif state == OffloadState.MIXED:
                stats['seq_on_gpu'] += 1
                stats['seq_on_cpu'] += 1
                resident_blocks = sum(1 for bid in gpu_map if int(bid) >= 0)
            elif state == OffloadState.PINNED:
                stats['seq_pinned'] += 1
                resident_blocks = len(rep.pending_block_ids or rep.block_ids or [])
            elif state == OffloadState.OFFLOAD_INFLIGHT:
                stats['seq_offload_inflight'] += 1
                resident_blocks = len(rep.block_ids or [])
            elif state == OffloadState.PREFETCH_INFLIGHT:
                stats['seq_prefetch_inflight'] += 1
                resident_blocks = len(rep.pending_block_ids or rep.block_ids or [])
            stats['resident_blocks'] += int(resident_blocks)
        return stats

    def _request_timing_fields(self, req: OnlineRequest) -> Dict[str, Any]:
        finish_s = float(req.finish_time_s if req.finish_time_s > 0.0 else time.perf_counter())
        submit_s = float(req.submit_time_s if req.submit_time_s > 0.0 else (self._online_total_t0 or finish_s))
        first_s = float(req.first_token_time_s)
        wall_ms = max(0.0, (finish_s - submit_s) * 1000.0)
        ttft_ms = max(0.0, (first_s - submit_s) * 1000.0) if first_s > 0.0 else 0.0
        gen_tokens = int(len(req.generated_tokens))
        if first_s > 0.0 and gen_tokens > 1:
            avg_itl_ms = max(0.0, (finish_s - first_s) * 1000.0) / max(1, gen_tokens - 1)
        else:
            avg_itl_ms = 0.0
        return {
            'submit_time_s': round(float(submit_s), 6),
            'first_token_time_s': round(float(first_s), 6) if first_s > 0.0 else 0.0,
            'finish_time_s': round(float(finish_s), 6),
            'ttft_ms': round(float(ttft_ms), 3),
            'wall_ms': round(float(wall_ms), 3),
            'avg_itl_ms': round(float(avg_itl_ms), 3),
            'first_token_source': str(req.first_token_source or ''),
        }

    def _mark_request_failed(self, req: OnlineRequest, error: Exception):
        if req.state == REQ_FAILED:
            return
        req.state = REQ_FAILED
        req.error = str(error)
        req.finished_step = self._online_step
        req.finish_time_s = time.perf_counter()
        self._clear_retry_tracking(req.request_id)
        self._p2_offload_cooldown_until.pop(int(req.request_id), None)
        rid = int(req.request_id)
        self.waiting_queue = deque([x for x in self.waiting_queue if x != rid])
        self.prefill_active = deque([x for x in self.prefill_active if x != rid])
        self.ready_decode = deque([x for x in self.ready_decode if x != rid])
        self.decode_active = deque([x for x in self.decode_active if x != rid])
        try:
            self.scheduler.release_sequence(req.request_id)
        except Exception:
            pass
        self.finished_queue.append(
            {
                'request_id': int(req.request_id),
                'state': req.state,
                'output': self.tokenizer.decode(req.generated_tokens, skip_special_tokens=True),
                'token_ids': list(req.generated_tokens),
                'token_logprobs': list(req.token_logprobs) if self._return_details_online else None,
                'error': req.error,
                'arrival_step': int(req.arrival_step),
                'finished_step': int(req.finished_step),
                'decode_steps': int(req.decode_steps),
                **self._request_timing_fields(req),
            }
        )

    def _finish_request(self, req: OnlineRequest):
        if req.state == REQ_DONE:
            return
        req.state = REQ_DONE
        req.finished_step = self._online_step
        req.finish_time_s = time.perf_counter()
        self._clear_retry_tracking(req.request_id)
        self._p2_offload_cooldown_until.pop(int(req.request_id), None)
        rid = int(req.request_id)
        self.waiting_queue = deque([x for x in self.waiting_queue if x != rid])
        self.prefill_active = deque([x for x in self.prefill_active if x != rid])
        self.ready_decode = deque([x for x in self.ready_decode if x != rid])
        self.decode_active = deque([x for x in self.decode_active if x != rid])
        self.scheduler.release_sequence(req.request_id)
        if self.multi_gpu is not None:
            self.multi_gpu.release_all(req.request_id)
        self.finished_queue.append(
            {
                'request_id': int(req.request_id),
                'state': req.state,
                'output': self.tokenizer.decode(req.generated_tokens, skip_special_tokens=True),
                'token_ids': list(req.generated_tokens),
                'token_logprobs': list(req.token_logprobs) if self._return_details_online else None,
                'error': req.error,
                'arrival_step': int(req.arrival_step),
                'finished_step': int(req.finished_step),
                'decode_steps': int(req.decode_steps),
                **self._request_timing_fields(req),
            }
        )

    def _sync_multigpu_after_prefill(self, seq_ids: List[int]):
        if self.multi_gpu is None:
            return

        sched = self.scheduler
        for sid in seq_ids:
            for layer_id in range(sched.num_layers):
                entry = sched.offloader.page_table.get((sid, layer_id))
                if entry is None:
                    continue
                if entry.state != OffloadState.ON_GPU or not entry.block_ids:
                    continue
                K, V = sched.pool.read_kv_from_blocks(layer_id, entry.block_ids, entry.seq_len)
                self.multi_gpu.register_layer_from_full_kv(sid, layer_id, K, V)

    def _sync_multigpu_decode_token(
        self,
        seq_id: int,
        layer_id: int,
        k_tok: torch.Tensor,
        v_tok: torch.Tensor,
    ):
        if self.multi_gpu is None:
            return
        self.multi_gpu.append_decode_token_all_gpus(seq_id, layer_id, k_tok, v_tok)

    def _select_round_robin_active_ids(self, active_ids: List[int], cap: int) -> List[int]:
        if not active_ids:
            return []
        if cap <= 0 or cap >= len(active_ids):
            return list(active_ids)
        # Fairness: cursor rotates every step, so skipped sequences are
        # pulled into the next rounds instead of being starved.
        start = self._decode_rr_cursor % len(active_ids)
        ordered = active_ids[start:] + active_ids[:start]
        picked = ordered[:cap]
        self._decode_rr_cursor = (start + cap) % len(active_ids)
        return picked

    def _queue_priority_retry(self, seq_id: int):
        sid = int(seq_id)
        if sid in self._priority_retry_set:
            return
        self._priority_retry.append(sid)
        self._priority_retry_set.add(sid)

    def _clear_retry_tracking(self, seq_id: int):
        sid = int(seq_id)
        self._consecutive_retry_count.pop(sid, None)
        if sid in self._priority_retry_set:
            self._priority_retry = deque([x for x in self._priority_retry if x != sid])
            self._priority_retry_set.discard(sid)

    def _trace_retry_event(
        self,
        event: str,
        seq_id: Optional[int] = None,
        seq_ids: Optional[Sequence[int]] = None,
        cur_batch: Optional[Sequence[int]] = None,
        scheduled_ids: Optional[Sequence[int]] = None,
        **extra: Any,
    ) -> None:
        if not bool(getattr(self, "_retry_trace_enabled", False)):
            return
        try:
            tracked: Set[int] = set()
            if seq_id is not None:
                sid0 = int(seq_id)
                if sid0 in self._retry_trace_seq_ids:
                    tracked.add(sid0)
            for source in (seq_ids, cur_batch, scheduled_ids):
                if source is None:
                    continue
                for sid in source:
                    sid_int = int(sid)
                    if sid_int in self._retry_trace_seq_ids:
                        tracked.add(sid_int)
            if not tracked:
                return

            cur_batch_list = [int(x) for x in (cur_batch or [])]
            scheduled_list = [int(x) for x in (scheduled_ids or [])]
            tracked_list = sorted(tracked)

            logical_lens: Dict[str, int] = {}
            for sid in sorted(set(cur_batch_list) | tracked):
                try:
                    logical_lens[str(int(sid))] = int(self._logical_seq_len_for_decode_guard(int(sid)))
                except Exception:
                    logical_lens[str(int(sid))] = 0

            generated_tokens: Dict[str, int] = {}
            for sid in tracked_list:
                req = self._requests.get(int(sid))
                generated_tokens[str(sid)] = int(len(getattr(req, "generated_tokens", []) or [])) if req is not None else 0

            row: Dict[str, Any] = {
                "ts": float(time.time()),
                "online_step": int(getattr(self, "_online_step", 0)),
                "decode_step": int(getattr(self.scheduler, "decode_step_count", 0)),
                "event": str(event),
                "seq_id": int(seq_id) if seq_id is not None else -1,
                "tracked_seq_ids": tracked_list,
                "retry_counts": {str(sid): int(self._consecutive_retry_count.get(int(sid), 0)) for sid in tracked_list},
                "generated_tokens": generated_tokens,
                "cur_batch_size": int(len(cur_batch_list)),
                "cur_batch_ids": cur_batch_list[:128],
                "scheduled_count": int(len(scheduled_list)),
                "scheduled_ids": scheduled_list[:128],
                "decode_active_count": int(len(getattr(self, "decode_active", []))),
                "ready_decode_count": int(len(getattr(self, "ready_decode", []))),
                "prefill_active_count": int(len(getattr(self, "prefill_active", []))),
                "waiting_count": int(len(getattr(self, "waiting_queue", []))),
                "decode_active_cap": int(getattr(self, "_online_decode_active_cap", 0)),
                "n_free": int(getattr(self.scheduler.pool, "n_free", 0)),
                "p2_low_threshold": int(self.scheduler.p2_low_threshold()),
                "cuda_free_post_cleanup_gb": float(getattr(self, "_online_cuda_free_post_cleanup_last_gb", 0.0)),
                "decode_memory_cap_events": int(getattr(self, "_online_decode_memory_cap_events", 0)),
                "decode_backpressure_events": int(getattr(self, "_online_decode_backpressure_events", 0)),
                "decode_no_progress_streak": int(getattr(self, "_online_decode_no_progress_streak", 0)),
                "p2_managed_active": bool(getattr(self.scheduler, "p2_managed_active", False)),
                "logical_seq_lens": logical_lens,
            }
            row.update(extra)
            with open(self._retry_trace_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        except Exception:
            return

    def _record_retryable_sequence(self, seq_id: int, reason: str) -> bool:
        sid = int(seq_id)
        prev = int(self._consecutive_retry_count.get(sid, 0))
        cur = prev + 1
        self._consecutive_retry_count[sid] = cur
        self._queue_priority_retry(sid)
        self._trace_retry_event(
            "retry_recorded",
            seq_id=sid,
            reason=str(reason),
            retry_count=int(cur),
            previous_retry_count=int(prev),
        )
        if prev == 0:
            self._apply_decode_backpressure(reason, count_event=True)
        else:
            self._online_prefill_pause_steps = max(
                self._online_prefill_pause_steps,
                int(self.decode_backpressure_pause_steps),
            )
        if cur < int(self.per_sequence_max_retry):
            return True

        self._online_decode_retry_timeout_fail_count += 1
        self._online_decode_retry_timeout_seq_ids.append(sid)
        self._trace_retry_event(
            "retry_timeout",
            seq_id=sid,
            reason=str(reason),
            retry_count=int(cur),
        )
        req = self._requests.get(sid)
        if req is not None and req.state not in (REQ_DONE, REQ_FAILED):
            self._mark_request_failed(
                req,
                RuntimeError(
                    f"decode_retry_timeout seq={sid} retries={cur} reason={reason}"
                ),
            )
        self._clear_retry_tracking(sid)
        return False

    def _select_decode_step_ids(self, active_ids: List[int], cap: int) -> List[int]:
        if not active_ids or cap <= 0:
            return []
        non_deferred_ids = [sid for sid in active_ids if not self._is_p2_offload_deferred(sid)]
        if non_deferred_ids:
            active_ids = non_deferred_ids
        cap = min(int(cap), len(active_ids))
        active_set = set(int(x) for x in active_ids)
        picked: List[int] = []

        retry_cap = int(math.floor(float(cap) * self.decode_retry_priority_quota))
        if self._priority_retry and retry_cap <= 0:
            retry_cap = 1
        retry_cap = max(0, min(cap, retry_cap))

        while self._priority_retry and len(picked) < retry_cap:
            sid = int(self._priority_retry.popleft())
            self._priority_retry_set.discard(sid)
            req = self._requests.get(sid)
            if req is None or req.state != REQ_DECODING:
                self._consecutive_retry_count.pop(sid, None)
                continue
            if sid not in active_set:
                continue
            picked.append(sid)

        remain = cap - len(picked)
        if remain <= 0:
            return picked
        picked_set = set(picked)
        pending_priority = set(self._priority_retry_set)
        rr_source = [
            sid for sid in active_ids
            if sid not in picked_set and sid not in pending_priority
        ]
        rr_pick = self._select_round_robin_active_ids(rr_source, remain)
        return picked + rr_pick

    def _initial_decode_active_cap(self, active_count: int) -> int:
        if active_count <= 0:
            return 0
        if self.decode_active_cap_initial > 0:
            return max(self.decode_active_cap_min, min(active_count, self.decode_active_cap_initial))
        return max(self.decode_active_cap_min, active_count)

    def _iter_decode_batches(self, active_ids: List[int]) -> List[List[int]]:
        if not active_ids:
            return []
        mb = self.decode_micro_batch_size if self.decode_micro_batch_size > 0 else len(active_ids)
        return [active_ids[i: i + mb] for i in range(0, len(active_ids), mb)]

    def _select_next_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=logits.device)
        if logits.shape[-1] == 1:
            return torch.zeros((logits.shape[0],), dtype=torch.long, device=logits.device)
        if self.stable_greedy_tie_eps <= 0:
            return logits.argmax(dim=-1)

        top2_vals, top2_ids = torch.topk(logits, k=2, dim=-1)
        gap = top2_vals[:, 0] - top2_vals[:, 1]
        primary = top2_ids[:, 0]
        secondary = top2_ids[:, 1]
        tie_mask = gap <= self.stable_greedy_tie_eps
        tie_choice = torch.minimum(primary, secondary)
        return torch.where(tie_mask, tie_choice, primary)

    def _admit_waiting_requests(self) -> int:
        admitted = 0
        sched = self.scheduler
        if self._prefill_is_paused():
            return 0
        prefill_cap = self._current_prefill_admit_cap()
        if not bool(self.online_prefill_admission_enabled):
            while self.waiting_queue and len(self.prefill_active) < prefill_cap:
                if int(sched.pool.n_free) <= int(sched.pool.N_wm_low):
                    break
                rid = self.waiting_queue.popleft()
                req = self._requests.get(rid)
                if req is None or req.state in (REQ_DONE, REQ_FAILED):
                    continue
                req.state = REQ_PREFILLING
                self.prefill_active.append(rid)
                admitted += 1
            return admitted

        while self.waiting_queue and len(self.prefill_active) < prefill_cap:
            if int(sched.pool.n_free) <= int(sched.pool.N_wm_low):
                break
            queue_ids = list(self.waiting_queue)
            lookahead = min(len(queue_ids), max(1, int(self.online_prefill_admission_lookahead)))
            selected_index = -1
            selected_req: Optional[OnlineRequest] = None
            stale_index = -1
            first_blocked_req: Optional[OnlineRequest] = None
            for idx in range(lookahead):
                rid = int(queue_ids[idx])
                req = self._requests.get(rid)
                if req is None or req.state in (REQ_DONE, REQ_FAILED):
                    stale_index = idx
                    break
                if self._prefill_admission_allows(req):
                    selected_index = idx
                    selected_req = req
                    break
                if first_blocked_req is None:
                    first_blocked_req = req
            if stale_index >= 0:
                queue_ids.pop(stale_index)
                self.waiting_queue = deque(queue_ids)
                continue
            if selected_index < 0 or selected_req is None:
                if first_blocked_req is not None:
                    # Refresh the last blocked reason for the visible head of the blocked window.
                    self._prefill_admission_allows(first_blocked_req)
                break
            rid = int(queue_ids.pop(selected_index))
            self.waiting_queue = deque(queue_ids)
            selected_req.state = REQ_PREFILLING
            self.prefill_active.append(rid)
            admitted += 1
        return admitted

    def _prefill_cursor_total(self) -> int:
        total = 0
        for rid in list(self.prefill_active):
            req = self._requests.get(int(rid))
            if req is not None and req.state == REQ_PREFILLING:
                total += int(getattr(req, 'prefill_cursor', 0) or 0)
        return int(total)

    def _update_prefill_no_progress_watchdog(self, prefill_stats: Dict[str, int]) -> None:
        active_ids = [int(rid) for rid in list(self.prefill_active) if self._requests.get(int(rid)) is not None and self._requests[int(rid)].state == REQ_PREFILLING]
        if not active_ids:
            self._online_prefill_no_progress_streak = 0
            self._online_prefill_no_progress_last_cursor = -1
            return
        cursor_total = self._prefill_cursor_total()
        progressed = int(prefill_stats.get('prefill_tokens', 0) or 0) > 0 or cursor_total != int(self._online_prefill_no_progress_last_cursor)
        self._online_prefill_no_progress_last_cursor = int(cursor_total)
        if progressed:
            self._online_prefill_no_progress_streak = 0
            return
        self._online_prefill_no_progress_steps += 1
        self._online_prefill_no_progress_streak += 1
        self._online_prefill_no_progress_peak = max(int(self._online_prefill_no_progress_peak), int(self._online_prefill_no_progress_streak))
        if self._online_prefill_no_progress_streak < int(self.prefill_no_progress_watchdog_steps):
            return
        self._online_prefill_no_progress_fail_count += len(active_ids)
        free_gb = float(self._sample_cuda_free_gb())
        for rid in active_ids:
            req = self._requests.get(int(rid))
            if req is None or req.state != REQ_PREFILLING:
                continue
            err = RuntimeError(
                'prefill_no_progress_watchdog:'
                f' cursor={int(getattr(req, "prefill_cursor", 0) or 0)}/{int(getattr(req, "prompt_token_len", 0) or 0)}'
                f' chunk_cap={int(getattr(req, "prefill_chunk_cap", 0) or 0)}'
                f' free_gb={free_gb:.4f}'
            )
            self._mark_request_failed(req, err)
        self._online_prefill_no_progress_streak = 0
        self._online_prefill_no_progress_last_cursor = -1

    def _run_prefill_budget(self) -> Dict[str, int]:
        if not self.prefill_active:
            return {'prefill_tokens': 0, 'prefill_finished': 0, 'prefill_failed': 0}
        if self._prefill_is_paused():
            return {'prefill_tokens': 0, 'prefill_finished': 0, 'prefill_failed': 0}

        self._update_online_pool_stats(phase="prefill")
        budget_left = self.prefill_token_budget_per_step if self.prefill_token_budget_per_step > 0 else 10**12
        used = 0
        finished = 0
        failed = 0
        remove_ids: List[int] = []

        ids = list(self.prefill_active)
        if ids:
            start = self._prefill_rr_cursor % len(ids)
            ordered = ids[start:] + ids[:start]
        else:
            start = 0
            ordered = []
        processed_batch_ids: Set[int] = set()

        # Batch-first path for online prefill:
        # bucket by past length (block-aligned by default), then batch same-past requests.
        if budget_left > 0 and self.prefill_batch_size > 1 and len(ordered) >= 2:
            grouped: Dict[Tuple[int, int], List[Tuple[int, OnlineRequest, int, int]]] = {}
            group_order: List[Tuple[int, int]] = []
            for rid in ordered:
                req = self._requests.get(rid)
                if req is None or req.state != REQ_PREFILLING or req.input_ids_cpu is None:
                    continue
                remain = int(req.prompt_token_len - req.prefill_cursor)
                if remain <= 0:
                    continue
                chunk_cap = int(req.prefill_chunk_cap) if int(req.prefill_chunk_cap) > 0 else int(self.chunk_size)
                past_len = int(self._past_kv_len(req.prefill_past_kv))
                bucket_base = int(max(1, self.prefill_past_bucket_tokens))
                bucket_id = int(past_len // bucket_base)
                gk = (bucket_id, past_len)
                if gk not in grouped:
                    grouped[gk] = []
                    group_order.append(gk)
                grouped[gk].append((rid, req, remain, chunk_cap))

            for gk in group_order:
                if budget_left <= 0:
                    break
                group = grouped.get(gk, [])
                if len(group) < 2:
                    continue
                group = group[:int(self.prefill_batch_size)]
                base_chunk = min(int(remain) for _, _, remain, _ in group)
                base_chunk = min(base_chunk, min(int(chunk_cap) for _, _, _, chunk_cap in group))
                if base_chunk <= 0:
                    continue
                max_take = int(budget_left // max(1, base_chunk))
                bucket_take = min(len(group), int(self.prefill_batch_size), int(max_take))
                if bucket_take < 2:
                    continue
                selected = group[:bucket_take]
                shared_chunk = min(
                    min(int(remain) for _, _, remain, _ in selected),
                    min(int(chunk_cap) for _, _, _, chunk_cap in selected),
                    int(budget_left // max(1, bucket_take)),
                )
                if shared_chunk <= 0:
                    continue

                batch_items = []
                past_list: List[Any] = []
                has_past = int(gk[1]) > 0
                for rid, req, _, _ in selected:
                    s = int(req.prefill_cursor)
                    e = s + int(shared_chunk)
                    batch_items.append((rid, req.input_ids_cpu[s:e]))
                    if has_past:
                        past_list.append(req.prefill_past_kv)
                t0 = time.perf_counter()
                try:
                    last_logits_map, past_kv_map = self._prefill_batch_online(
                        batch_items,
                        past_key_values_list=(past_list if has_past else None),
                    )
                except Exception as exc:
                    if self._is_retryable_memory_error(exc):
                        for _, req, _, _ in selected:
                            req.prefill_chunk_cap = self._prefill_next_chunk_cap_after_oom(int(shared_chunk))
                            req.prefill_chunk_success_streak = 0
                            req.prefill_chunk_recovery_disabled = True
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self._apply_prefill_backpressure("prefill_batch_failed")
                        budget_left = 0
                        break
                    for rid, req, _, _ in selected:
                        self._mark_request_failed(req, exc)
                        failed += 1
                        remove_ids.append(rid)
                    continue

                self._online_prefill_ms += (time.perf_counter() - t0) * 1000.0
                self._update_online_pool_stats(phase="prefill")
                for rid, req, _, chunk_cap in selected:
                    req.prefill_past_kv = past_kv_map.get(rid)
                    req.prefill_cursor = int(req.prefill_cursor + int(shared_chunk))
                    req.last_logits = last_logits_map.get(rid)
                    processed_batch_ids.add(int(rid))
                    used += int(shared_chunk)
                    budget_left -= int(shared_chunk)

                    if (
                        req.prefill_chunk_cap > 0
                        and not bool(req.prefill_chunk_recovery_disabled)
                        and int(shared_chunk) >= int(chunk_cap)
                        and req.prefill_chunk_cap < self.chunk_size
                    ):
                        req.prefill_chunk_success_streak += 1
                        if req.prefill_chunk_success_streak >= 3:
                            grow = max(1, min(256, int(req.prefill_chunk_cap) // 2))
                            req.prefill_chunk_cap = min(self.chunk_size, int(req.prefill_chunk_cap) + int(grow))
                            req.prefill_chunk_success_streak = 0
                    else:
                        req.prefill_chunk_success_streak = 0

                    if req.prefill_cursor < req.prompt_token_len:
                        continue
                    try:
                        self._activate_decoding_request(
                            req,
                            reset_decode_state=(not req.decode_state_initialized),
                        )
                        finished += 1
                        remove_ids.append(rid)
                    except Exception as exc:
                        if self._is_retryable_memory_error(exc):
                            req.prefill_activate_retries += 1
                            try:
                                self.scheduler.release_sequence(req.request_id)
                            except Exception:
                                pass
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            self._apply_prefill_backpressure("prefill_activate_failed")
                            budget_left = 0
                            break
                        self._mark_request_failed(req, exc)
                        failed += 1
                        remove_ids.append(rid)

        for rid in ordered:
            if budget_left <= 0:
                break
            if rid in processed_batch_ids:
                continue
            req = self._requests.get(rid)
            if req is None or req.state != REQ_PREFILLING:
                remove_ids.append(rid)
                continue

            remain = int(req.prompt_token_len - req.prefill_cursor)
            if remain <= 0:
                try:
                    reset_decode_state = not req.decode_state_initialized
                    self._activate_decoding_request(req, reset_decode_state=reset_decode_state)
                    finished += 1
                except Exception as exc:
                    if self._is_retryable_memory_error(exc):
                        req.prefill_activate_retries += 1
                        try:
                            self.scheduler.release_sequence(req.request_id)
                        except Exception:
                            pass
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self._apply_prefill_backpressure("prefill_activate_failed")
                        budget_left = 0
                        break
                    self._mark_request_failed(req, exc)
                    failed += 1
                    remove_ids.append(rid)
                    continue
                remove_ids.append(rid)
                continue

            chunk_cap = int(req.prefill_chunk_cap) if int(req.prefill_chunk_cap) > 0 else int(self.chunk_size)
            chunk = max(1, min(chunk_cap, remain, budget_left))
            t0 = time.perf_counter()
            try:
                done = self._prefill_chunk_online(req, chunk_tokens=chunk)
            except Exception as exc:
                if self._is_retryable_memory_error(exc):
                    req.prefill_chunk_cap = self._prefill_next_chunk_cap_after_oom(int(chunk))
                    req.prefill_chunk_success_streak = 0
                    req.prefill_chunk_recovery_disabled = True
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._apply_prefill_backpressure("prefill_chunk_failed")
                    budget_left = 0
                    break
                self._mark_request_failed(req, exc)
                failed += 1
                remove_ids.append(rid)
                continue

            self._online_prefill_ms += (time.perf_counter() - t0) * 1000.0
            self._update_online_pool_stats(phase="prefill")
            used += chunk
            budget_left -= chunk
            if req.prefill_chunk_cap > 0 and not bool(req.prefill_chunk_recovery_disabled) and chunk >= req.prefill_chunk_cap and req.prefill_chunk_cap < self.chunk_size:
                req.prefill_chunk_success_streak += 1
                if req.prefill_chunk_success_streak >= 3:
                    grow = max(1, min(256, int(req.prefill_chunk_cap) // 2))
                    req.prefill_chunk_cap = min(self.chunk_size, int(req.prefill_chunk_cap) + int(grow))
                    req.prefill_chunk_success_streak = 0
            else:
                req.prefill_chunk_success_streak = 0

            if done:
                try:
                    reset_decode_state = not req.decode_state_initialized
                    self._activate_decoding_request(req, reset_decode_state=reset_decode_state)
                    finished += 1
                except Exception as exc:
                    if self._is_retryable_memory_error(exc):
                        req.prefill_activate_retries += 1
                        try:
                            self.scheduler.release_sequence(req.request_id)
                        except Exception:
                            pass
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self._apply_prefill_backpressure("prefill_activate_failed")
                        budget_left = 0
                        break
                    self._mark_request_failed(req, exc)
                    failed += 1
                    remove_ids.append(rid)
                    continue
                remove_ids.append(rid)

        if remove_ids:
            remove_set = set(remove_ids)
            self.prefill_active = deque([rid for rid in self.prefill_active if rid not in remove_set])
        if self.prefill_active:
            self._prefill_rr_cursor = (start + 1) % len(self.prefill_active)
        else:
            self._prefill_rr_cursor = 0

        return {'prefill_tokens': int(used), 'prefill_finished': int(finished), 'prefill_failed': int(failed)}

    def _update_decode_cap_state(self, active_ids: List[int], scheduled_ids: List[int]):
        if not scheduled_ids:
            return
        sched = self.scheduler
        self._maybe_relieve_p2_pressure(scheduled_ids)
        sched.update_decode_pressure(active_seq_count=len(scheduled_ids), refresh_thrash=False)
        self._update_online_pool_stats(phase="decode")
        self._update_online_cuda_stats()

        pressure_needed = max(1, int(self.decode_active_cap_pressure_windows))
        low_pressure = (
            int(sched.pool.n_free) <= int(sched.p2_low_threshold())
            or bool(sched.p2_cuda_pressure_signal(has_decode_work=bool(active_ids)))
            or sched.get_thrash_win16() >= float(sched.decode_window_thrash_low)
        )
        if low_pressure:
            self._online_decode_cap_pressure_streak += 1
            self._online_decode_cap_recover_streak = 0
        else:
            self._online_decode_cap_pressure_streak = max(0, self._online_decode_cap_pressure_streak - 1)
            recover_cond = (
                int(sched.pool.n_free) >= int(sched.pool.N_wm_high)
                and not bool(sched.p2_cuda_pressure_signal(has_decode_work=bool(active_ids)))
                and sched.get_thrash_win16() < float(sched.decode_window_thrash_low)
            )
            if recover_cond:
                self._online_decode_cap_recover_streak += 1
            else:
                self._online_decode_cap_recover_streak = 0

        if (
            self._online_decode_cap_pressure_streak >= pressure_needed
            and self._online_decode_active_cap > self.decode_active_cap_min
        ):
            cap_floor = max(
                self.decode_active_cap_min,
                int(math.ceil(float(max(1, self._online_decode_active_cap_boot)) * float(self.decode_active_cap_floor_ratio))),
            )
            self._online_decode_active_cap = max(
                cap_floor,
                int(math.floor(float(self._online_decode_active_cap) * self.decode_active_cap_downscale)),
            )
            self._online_decode_cap_pressure_streak = 0
            self._online_decode_cap_recover_streak = 0

        cap_upper = max(self.decode_active_cap_min, self._online_decode_active_cap_boot)
        if self.max_decode_active_cap > 0:
            cap_upper = min(cap_upper, self.max_decode_active_cap)
        if (
            self._online_decode_cap_recover_streak >= int(sched.decode_window_recover_steps)
            and self._online_decode_active_cap < cap_upper
        ):
            self._online_decode_active_cap = min(
                cap_upper,
                self._online_decode_active_cap + self.decode_active_cap_recover_step,
            )
            self._online_decode_cap_recover_streak = 0

        if self._online_decode_active_cap > 0:
            self._online_decode_active_cap_min_seen = (
                min(self._online_decode_active_cap_min_seen, self._online_decode_active_cap)
                if self._online_decode_active_cap_min_seen > 0
                else int(self._online_decode_active_cap)
            )

    def _record_decode_path_selected(self, path: str):
        key = str(path or "unknown")
        self._online_decode_path_selected_counts[key] = (
            int(self._online_decode_path_selected_counts.get(key, 0)) + 1
        )
        self._online_decode_path_last_selected = key

    def _record_decode_path_fallback(self, reason: str):
        key = str(reason or "unknown")
        self._online_decode_path_fallback_count += 1
        self._online_decode_path_fallback_reasons[key] = (
            int(self._online_decode_path_fallback_reasons.get(key, 0)) + 1
        )

    def _decode_path_selected_label(self) -> str:
        if not self._online_decode_path_selected_counts:
            return "none"
        ranked = sorted(
            self._online_decode_path_selected_counts.items(),
            key=lambda kv: (-int(kv[1]), kv[0]),
        )
        if len(ranked) > 1 and int(ranked[0][1]) == int(ranked[1][1]):
            return "mixed"
        return str(ranked[0][0])

    def _decode_path_fallback_reason_topk(self, topk: int = 3) -> Dict[str, int]:
        if topk <= 0:
            return {}
        ranked = sorted(
            self._online_decode_path_fallback_reasons.items(),
            key=lambda kv: (-int(kv[1]), kv[0]),
        )
        return {str(k): int(v) for k, v in ranked[:topk]}

    def _record_decode_materialize_kv_bytes(self, *tensors: torch.Tensor) -> None:
        total = 0
        for tensor in tensors:
            if torch.is_tensor(tensor):
                total += int(tensor.numel()) * int(tensor.element_size())
        self._online_decode_materialize_kv_bytes += int(total)

    def _record_decode_paged_direct_blocked(self, reason: str) -> None:
        text = str(reason or "unknown")
        self._online_decode_paged_direct_blocked_reason = text
        if any(token in text.lower() for token in ("resident", "missing", "not_on_gpu", "mixed")):
            self._online_decode_paged_direct_resident_miss_steps += 1

    def _record_decode_page16_native_blocked(self, reason: str) -> None:
        text = str(reason or "unknown")
        self._online_decode_page16_native_blocked_reason = text
        if any(token in text.lower() for token in ("resident", "missing", "not_on_gpu", "mixed")):
            self._online_decode_page16_native_resident_miss_steps += 1

    def _decode_backend_label(self) -> str:
        selected = self._decode_path_selected_label()
        if selected in ("page16_native", "paged_direct_flash", "paged_materialize", "rebuild_dense"):
            return selected
        if selected == "rebuild":
            return "rebuild_dense"
        if selected == "paged_direct":
            return "paged_materialize"
        return selected

    def _resolve_decode_path_for_batch(self, seq_ids: Sequence[int]) -> Tuple[str, str]:
        mode = str(self.decode_path_mode or "auto").lower()
        if mode == "rebuild":
            return "rebuild_dense", ""
        if mode == "page16_native":
            if self.num_gpus != 1:
                return "rebuild_dense", "multi_gpu_not_supported"
            if not seq_ids:
                return "rebuild_dense", "empty_batch"
            ok, reason = self.scheduler.page16_native_decode_support()
            if not ok:
                return "rebuild_dense", str(reason or "page16_native_unavailable")
            return "page16_native", ""
        if self.num_gpus != 1:
            return "rebuild_dense", "multi_gpu_not_supported"
        if not self.decode_paged_flash_enabled:
            return "rebuild_dense", "paged_flash_disabled"
        if not self.decode_paged_flash_active:
            return "rebuild_dense", str(self.decode_paged_flash_reason or "support_not_ok")
        if not seq_ids:
            return "rebuild_dense", "empty_batch"
        if mode == "paged_materialize":
            return "paged_materialize", ""
        return "paged_direct_flash", ""

    @staticmethod
    def _classify_decode_path_runtime_error(exc: Exception) -> str:
        msg = str(exc).lower()
        if isinstance(exc, MemoryError) or "oom" in msg or "out of memory" in msg:
            return "oom"
        if "shape" in msg or "size mismatch" in msg or "stride" in msg:
            return "shape_not_supported"
        return "runtime_error"

    def _run_decode_batch(self) -> Dict[str, int]:
        sched = self.scheduler
        active_ids = [rid for rid in self.decode_active if self._requests[rid].state == REQ_DECODING]
        self.decode_active = deque(active_ids)
        if not active_ids:
            return {
                'decode_scheduled': 0,
                'decode_tokens': 0,
                'decode_microbatches': 0,
                'decode_path_selected': self._decode_path_selected_label(),
                'decode_backend': self._decode_backend_label(),
                'decode_paged_direct_steps': int(self._online_decode_paged_direct_steps),
                'decode_page16_native_steps': int(self._online_decode_page16_native_steps),
                'decode_rebuild_steps': int(self._online_decode_rebuild_steps),
                'decode_materialize_kv_bytes': int(self._online_decode_materialize_kv_bytes),
                'decode_paged_direct_blocked_reason': str(self._online_decode_paged_direct_blocked_reason),
                'decode_paged_direct_resident_miss_steps': int(self._online_decode_paged_direct_resident_miss_steps),
                'decode_page16_native_blocked_reason': str(self._online_decode_page16_native_blocked_reason),
                'decode_page16_native_resident_miss_steps': int(self._online_decode_page16_native_resident_miss_steps),
                'decode_page16_native_kernel_ms': round(float(self._online_decode_page16_native_kernel_ms), 3),
                'decode_path_fallback_count': int(self._online_decode_path_fallback_count),
                'decode_path_fallback_reason_topk': self._decode_path_fallback_reason_topk(),
            }

        cap_boot_now = self._initial_decode_active_cap(len(active_ids))
        if self.max_decode_active_cap > 0:
            cap_boot_now = min(cap_boot_now, self.max_decode_active_cap)
        if self._online_decode_active_cap <= 0:
            self._online_decode_active_cap = max(self.decode_active_cap_min, cap_boot_now)
            self._online_decode_active_cap_boot = int(self._online_decode_active_cap)
            self._online_decode_active_cap_min_seen = int(self._online_decode_active_cap)
        else:
            self._online_decode_active_cap_boot = max(self._online_decode_active_cap_boot, int(cap_boot_now))

        sched.phase = PhaseState.DECODE
        sched.decode_step_count += 1
        sched.offloader.set_decode_step(int(sched.decode_step_count))

        cap_now = int(self._online_decode_active_cap)
        if self.max_decode_active_cap > 0:
            cap_now = min(cap_now, self.max_decode_active_cap)
        scheduled_ids = self._select_decode_step_ids(active_ids, cap_now)
        self._trace_retry_event("scheduled", scheduled_ids=scheduled_ids)
        if not scheduled_ids:
            return {
                'decode_scheduled': 0,
                'decode_tokens': 0,
                'decode_microbatches': 0,
                'decode_path_selected': self._decode_path_selected_label(),
                'decode_backend': self._decode_backend_label(),
                'decode_paged_direct_steps': int(self._online_decode_paged_direct_steps),
                'decode_page16_native_steps': int(self._online_decode_page16_native_steps),
                'decode_rebuild_steps': int(self._online_decode_rebuild_steps),
                'decode_materialize_kv_bytes': int(self._online_decode_materialize_kv_bytes),
                'decode_paged_direct_blocked_reason': str(self._online_decode_paged_direct_blocked_reason),
                'decode_paged_direct_resident_miss_steps': int(self._online_decode_paged_direct_resident_miss_steps),
                'decode_page16_native_blocked_reason': str(self._online_decode_page16_native_blocked_reason),
                'decode_page16_native_resident_miss_steps': int(self._online_decode_page16_native_resident_miss_steps),
                'decode_page16_native_kernel_ms': round(float(self._online_decode_page16_native_kernel_ms), 3),
                'decode_path_fallback_count': int(self._online_decode_path_fallback_count),
                'decode_path_fallback_reason_topk': self._decode_path_fallback_reason_topk(),
            }

        self._update_decode_cap_state(active_ids, scheduled_ids)
        step_t0 = time.perf_counter()
        decode_tokens = 0
        microbatches = 0
        finished_ids: List[int] = []
        failed_ids: List[int] = []
        retry_ids: List[int] = []

        for mb_ids in self._iter_decode_batches(scheduled_ids):
            sub_batches = [mb_ids]
            while sub_batches:
                cur_batch = sub_batches.pop(0)
                if not cur_batch:
                    continue
                path, fallback_reason = self._resolve_decode_path_for_batch(cur_batch)
                direct_decode = False
                direct_ctx = None
                pkv = None
                seq_lens = None
                logical_seq_lens = None

                if path != "page16_native" and self.decode_page16_native_strict and self.decode_path_mode == "page16_native":
                    reason = str(fallback_reason or "page16_native_unavailable")
                    self._record_decode_path_fallback(reason)
                    self._record_decode_page16_native_blocked(reason)
                    raise RuntimeError(f"page16_native_blocked:{reason}")

                if path != "paged_direct_flash" and self.decode_paged_flash_strict and self.decode_path_mode == "paged_direct":
                    reason = str(fallback_reason or "paged_direct_unavailable")
                    self._record_decode_path_fallback(reason)
                    self._record_decode_paged_direct_blocked(reason)
                    raise RuntimeError(f"paged_direct_blocked:{reason}")

                if path == "page16_native":
                    try:
                        direct_ctx, seq_lens, logical_seq_lens = self._prepare_page16_native_context(cur_batch)
                        direct_decode = True
                        self._record_decode_path_selected("page16_native")
                        self._online_decode_page16_native_steps += 1
                    except Exception as exc:
                        reason = str(exc)
                        if not reason.startswith("page16_native"):
                            reason = f"page16_native_{self._classify_decode_path_runtime_error(exc)}"
                        self._record_decode_path_fallback(reason)
                        self._record_decode_page16_native_blocked(reason)
                        if self.decode_page16_native_strict:
                            raise RuntimeError(f"page16_native_blocked:{reason}") from exc
                        path = "rebuild_dense"
                        fallback_reason = "page16_native_runtime_fallback"
                        direct_decode = False
                        direct_ctx = None
                        seq_lens = None
                        logical_seq_lens = None

                if path == "paged_direct_flash":
                    try:
                        direct_ctx, seq_lens, logical_seq_lens = self._prepare_paged_direct_context(cur_batch)
                        direct_decode = True
                        self._record_decode_path_selected("paged_direct_flash")
                        self._online_decode_paged_direct_steps += 1
                    except Exception as exc:
                        reason = str(exc)
                        if not reason.startswith("paged_direct"):
                            reason = f"paged_direct_{self._classify_decode_path_runtime_error(exc)}"
                        self._record_decode_path_fallback(reason)
                        self._record_decode_paged_direct_blocked(reason)
                        if self.decode_paged_flash_strict:
                            raise RuntimeError(f"paged_direct_blocked:{reason}") from exc
                        path = "rebuild_dense"
                        fallback_reason = "paged_direct_runtime_fallback"
                        direct_decode = False
                        direct_ctx = None
                        seq_lens = None
                        logical_seq_lens = None

                if path == "paged_materialize":
                    try:
                        pkv, seq_lens, logical_seq_lens = self._rebuild_pkv_paged_direct(
                            cur_batch,
                            return_seq_lens=True,
                        )
                        self._record_decode_path_selected("paged_materialize")
                    except Exception as exc:
                        reason = f"paged_materialize_{self._classify_decode_path_runtime_error(exc)}"
                        self._record_decode_path_fallback(reason)
                        if self.decode_paged_flash_strict:
                            self._record_decode_paged_direct_blocked(reason)
                            raise RuntimeError(f"paged_direct_blocked:{reason}") from exc
                        path = "rebuild_dense"
                        fallback_reason = "paged_materialize_runtime_fallback"
                        pkv = None
                        seq_lens = None
                        logical_seq_lens = None

                if not direct_decode and path != "paged_materialize":
                    if fallback_reason:
                        self._record_decode_path_fallback(str(fallback_reason))
                    bucketed_batches = self._bucket_decode_batch_by_length(cur_batch)
                    if len(bucketed_batches) > 1:
                        for deferred_bucket in reversed(bucketed_batches[1:]):
                            sub_batches.insert(0, deferred_bucket)
                        cur_batch = list(bucketed_batches[0])
                    guarded_batch, deferred_batch = self._maybe_apply_decode_memory_guard(cur_batch)
                    if deferred_batch:
                        sub_batches.insert(0, deferred_batch)
                    cur_batch = guarded_batch
                    try:
                        pkv, seq_lens, logical_seq_lens = self._rebuild_pkv(cur_batch, return_seq_lens=True)
                        self._record_decode_path_selected("rebuild_dense")
                        self._online_decode_rebuild_steps += 1
                    except Exception as exc:
                        if not self._is_retryable_memory_error(exc):
                            raise
                        est_maxlen_gb, est_sumlen_gb = self._estimate_decode_rebuild_peak_gb(cur_batch)
                        self._record_decode_memory_cap_event(len(cur_batch), est_maxlen_gb, est_sumlen_gb)
                        self._trace_retry_event(
                            "rebuild_oom",
                            seq_ids=cur_batch,
                            cur_batch=cur_batch,
                            reason="rebuild_pkv_failed",
                            est_maxlen_gb=float(est_maxlen_gb),
                            est_sumlen_gb=float(est_sumlen_gb),
                            exception=str(exc)[:240],
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                        self._record_post_cleanup_cuda_state(active_seq_count=max(1, len(cur_batch)))
                        self._apply_decode_backpressure('decode_memory_cap')
                        if len(cur_batch) == 1:
                            sid0 = int(cur_batch[0])
                            retry_ids.append(sid0)
                            self._online_decode_append_fail_count += 1
                            if not self._record_retryable_sequence(sid0, "rebuild_pkv_failed"):
                                failed_ids.append(sid0)
                            continue
                        mid = len(cur_batch) // 2
                        sub_batches.insert(0, cur_batch[mid:])
                        sub_batches.insert(0, cur_batch[:mid])
                        continue

                input_ids = torch.tensor(
                    [int(self._requests[sid].current_token) for sid in cur_batch],
                    dtype=torch.long,
                    device='cuda',
                ).unsqueeze(1)
                attention_mask = self._build_decode_attention_mask(seq_lens, device=input_ids.device)
                position_ids = self._build_decode_position_ids(logical_seq_lens, device=input_ids.device)
                try:
                    with torch.no_grad():
                        if direct_decode:
                            sched.begin_paged_direct_decode(direct_ctx)
                            try:
                                out = self.model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    past_key_values=None,
                                    use_cache=False,
                                )
                            finally:
                                sched.end_paged_direct_decode()
                        else:
                            out = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=pkv,
                                use_cache=True,
                            )
                except Exception as exc:
                    if direct_decode:
                        backend = str((direct_ctx or {}).get('backend', 'paged_direct_flash'))
                        if backend == "page16_native":
                            reason = f"page16_native_forward_{self._classify_decode_path_runtime_error(exc)}"
                            self._record_decode_path_fallback(reason)
                            self._record_decode_page16_native_blocked(reason)
                        else:
                            reason = f"paged_direct_forward_{self._classify_decode_path_runtime_error(exc)}"
                            self._record_decode_path_fallback(reason)
                            self._record_decode_paged_direct_blocked(reason)
                        raise
                    msg = str(exc).lower()
                    is_retryable_oom = self._is_retryable_memory_error(exc)
                    is_varlen_retry = 'varlen' in msg
                    if len(cur_batch) > 1 and (is_retryable_oom or is_varlen_retry):
                        if is_retryable_oom:
                            est_maxlen_gb, est_sumlen_gb = self._estimate_decode_rebuild_peak_gb(cur_batch)
                            self._record_decode_memory_cap_event(len(cur_batch), est_maxlen_gb, est_sumlen_gb)
                            self._trace_retry_event(
                                "forward_oom",
                                seq_ids=cur_batch,
                                cur_batch=cur_batch,
                                reason="decode_forward_failed",
                                est_maxlen_gb=float(est_maxlen_gb),
                                est_sumlen_gb=float(est_sumlen_gb),
                                exception=str(exc)[:240],
                            )
                            gc.collect()
                            torch.cuda.empty_cache()
                            self._record_post_cleanup_cuda_state(active_seq_count=max(1, len(cur_batch)))
                            self._apply_decode_backpressure('decode_memory_cap')
                        mid = len(cur_batch) // 2
                        sub_batches.insert(0, cur_batch[mid:])
                        sub_batches.insert(0, cur_batch[:mid])
                        del pkv, seq_lens, logical_seq_lens, input_ids, attention_mask, position_ids, direct_ctx
                        continue
                    if len(cur_batch) == 1 and (is_retryable_oom or is_varlen_retry):
                        if is_retryable_oom:
                            est_maxlen_gb, est_sumlen_gb = self._estimate_decode_rebuild_peak_gb(cur_batch)
                            self._record_decode_memory_cap_event(len(cur_batch), est_maxlen_gb, est_sumlen_gb)
                            self._trace_retry_event(
                                "forward_oom",
                                seq_ids=cur_batch,
                                cur_batch=cur_batch,
                                reason="decode_forward_failed",
                                est_maxlen_gb=float(est_maxlen_gb),
                                est_sumlen_gb=float(est_sumlen_gb),
                                exception=str(exc)[:240],
                            )
                            gc.collect()
                            torch.cuda.empty_cache()
                            self._record_post_cleanup_cuda_state(active_seq_count=1)
                            self._apply_decode_backpressure('decode_memory_cap')
                        sid0 = int(cur_batch[0])
                        retry_ids.append(sid0)
                        self._online_decode_append_fail_count += 1
                        if not self._record_retryable_sequence(sid0, "decode_forward_failed"):
                            failed_ids.append(sid0)
                        del pkv, seq_lens, logical_seq_lens, input_ids, attention_mask, position_ids, direct_ctx
                        continue
                    if len(cur_batch) == 1:
                        req = self._requests[cur_batch[0]]
                        self._mark_request_failed(req, exc)
                        failed_ids.append(cur_batch[0])
                        del pkv, seq_lens, logical_seq_lens, input_ids, attention_mask, position_ids, direct_ctx
                        continue
                    raise

                if direct_decode:
                    sched.commit_paged_direct_decode(cur_batch)
                    self._online_decode_page16_native_kernel_ms = float(
                        getattr(sched, '_page16_native_kernel_ms_accum', 0.0)
                    )

                microbatches += 1
                self._online_decode_microbatch_sizes.append(len(cur_batch))
                next_logits = out.logits[:, -1, :]
                next_tok = self._select_next_tokens(next_logits)
                if self._return_details_online:
                    next_logprobs = torch.nn.functional.log_softmax(next_logits, dim=-1)

                for j, sid in enumerate(cur_batch):
                    if sid in failed_ids:
                        continue
                    append_failed = False
                    if not direct_decode:
                        for layer_id in range(sched.num_layers):
                            K_new, V_new = out.past_key_values[layer_id]
                            if K_new.shape[0] == 0:
                                continue
                            k_tok = K_new[j: j + 1, :, -1:, :].squeeze(0)
                            v_tok = V_new[j: j + 1, :, -1:, :].squeeze(0)
                            append_res = sched.decode_step_schedule(layer_id, sid, k_tok, v_tok)
                            if not append_res.ok:
                                retry_ids.append(sid)
                                self._online_decode_append_fail_count += 1
                                if append_res.retryable:
                                    if not self._record_retryable_sequence(
                                        sid,
                                        append_res.reason or "decode_append_retryable",
                                    ):
                                        failed_ids.append(sid)
                                else:
                                    req = self._requests[sid]
                                    self._mark_request_failed(req, RuntimeError(append_res.reason or "decode_append_failed"))
                                    failed_ids.append(sid)
                                append_failed = True
                                break
                            self._sync_multigpu_decode_token(sid, layer_id, k_tok, v_tok)
                    if append_failed:
                        continue
                    req = self._requests[sid]
                    tok = int(next_tok[j].item())
                    self._trace_retry_event(
                        "append_success",
                        seq_id=int(sid),
                        cur_batch=cur_batch,
                        token=int(tok),
                        generated_tokens_after=int(len(req.generated_tokens) + 1),
                        retry_count_before_clear=int(self._consecutive_retry_count.get(int(sid), 0)),
                    )
                    self._clear_retry_tracking(sid)
                    req.generated_tokens.append(tok)
                    if req.first_token_time_s <= 0.0:
                        req.first_token_time_s = time.perf_counter()
                        req.first_token_source = "decode"
                    req.current_token = tok
                    req.decode_steps += 1
                    decode_tokens += 1
                    self._online_generated_tokens += 1
                    if self._return_details_online:
                        req.token_logprobs.append(float(next_logprobs[j, next_tok[j]].item()))
                    eos_hit = tok == self.tokenizer.eos_token_id
                    max_hit = len(req.generated_tokens) >= int(req.max_new_tokens)
                    if eos_hit or max_hit:
                        req.eos_reached = bool(eos_hit)
                        finished_ids.append(sid)

                if self._return_details_online:
                    del next_logprobs
                del pkv, seq_lens, logical_seq_lens, input_ids, attention_mask, position_ids, out, next_logits, next_tok, direct_ctx
                for sid in cur_batch:
                    if sid in finished_ids or sid in failed_ids or sid in retry_ids:
                        continue
                    req = self._requests.get(sid)
                    if req is not None and req.state == REQ_DECODING:
                        pass

        if scheduled_ids and decode_tokens <= 0:
            self._online_decode_no_progress_steps += 1
            self._online_decode_no_progress_streak += 1
            self._online_decode_no_progress_peak = max(
                self._online_decode_no_progress_peak,
                self._online_decode_no_progress_streak,
            )
            if self._online_decode_no_progress_streak >= self.decode_no_progress_watchdog_steps:
                self._apply_decode_backpressure("decode_no_progress_watchdog")
                self._online_decode_no_progress_streak = 0
        else:
            self._online_decode_no_progress_streak = 0

        self._online_decode_step_lat_ms.append((time.perf_counter() - step_t0) * 1000.0)

        if finished_ids:
            for sid in list(dict.fromkeys(finished_ids)):
                req = self._requests.get(sid)
                if req is None or req.state != REQ_DECODING:
                    continue
                self._finish_request(req)
        if failed_ids:
            failed_set = set(failed_ids)
            self.decode_active = deque([sid for sid in self.decode_active if sid not in failed_set])
        if finished_ids:
            done_set = set(finished_ids)
            self.decode_active = deque([sid for sid in self.decode_active if sid not in done_set])

        if decode_tokens > 0 and self.has_pending_requests():
            self._update_online_pool_stats(phase="decode")
            post_decode_pressure = (
                int(sched.pool.n_free) <= int(sched.p2_low_threshold())
                or bool(sched.p2_cuda_pressure_signal(has_decode_work=bool(self.decode_active or self.ready_decode)))
            )
            if post_decode_pressure:
                # The current decode tokens have already consumed their KV for this step.
                # Age/residency gates still prevent immediate churn on just-touched sequences.
                self._maybe_relieve_p2_pressure([])

        self._record_post_cleanup_cuda_state(active_seq_count=len(scheduled_ids))

        return {
            'decode_scheduled': int(len(scheduled_ids)),
            'decode_tokens': int(decode_tokens),
            'decode_microbatches': int(microbatches),
            'decode_path_selected': self._decode_path_selected_label(),
            'decode_backend': self._decode_backend_label(),
            'decode_paged_direct_steps': int(self._online_decode_paged_direct_steps),
            'decode_page16_native_steps': int(self._online_decode_page16_native_steps),
            'decode_rebuild_steps': int(self._online_decode_rebuild_steps),
            'decode_materialize_kv_bytes': int(self._online_decode_materialize_kv_bytes),
            'decode_paged_direct_blocked_reason': str(self._online_decode_paged_direct_blocked_reason),
            'decode_paged_direct_resident_miss_steps': int(self._online_decode_paged_direct_resident_miss_steps),
            'decode_page16_native_blocked_reason': str(self._online_decode_page16_native_blocked_reason),
            'decode_page16_native_resident_miss_steps': int(self._online_decode_page16_native_resident_miss_steps),
            'decode_page16_native_kernel_ms': round(float(self._online_decode_page16_native_kernel_ms), 3),
            'decode_path_fallback_count': int(self._online_decode_path_fallback_count),
            'decode_path_fallback_reason_topk': self._decode_path_fallback_reason_topk(),
        }

    def step(self) -> Dict[str, Any]:
        self._online_step += 1
        self._online_decode_memory_guard_target_batch_last = 0
        self._online_decode_memory_guard_source_batch_last = 0
        self._online_decode_memory_guard_free_gb_last = 0.0
        self._online_decode_memory_guard_budget_gb_last = 0.0
        self._online_decode_memory_guard_reserve_gb_last = 0.0
        self._online_decode_memory_guard_hard_guard_gb = float(self.decode_memory_guard_hard_guard_gb)
        step_t0 = time.perf_counter()
        if hasattr(self.scheduler, 'reset_p2_ready_step_telemetry'):
            self.scheduler.reset_p2_ready_step_telemetry()
        admitted = self._admit_waiting_requests()
        prefill_stats = self._run_prefill_budget()
        self._update_prefill_no_progress_watchdog(prefill_stats)
        ready_activated = self._activate_ready_decode_requests()
        decode_stats = self._run_decode_batch()
        self._update_online_pool_stats()
        if self._online_prefill_pause_steps > 0:
            self._online_prefill_pause_steps = max(0, self._online_prefill_pause_steps - 1)
        if not self.has_pending_requests():
            self.scheduler.phase = PhaseState.IDLE

        step_ms = (time.perf_counter() - step_t0) * 1000.0
        offload_stats_now = self.scheduler.offloader.get_stats(reset=False)
        offload_cum_delta = self._stats_delta(offload_stats_now, self._online_offload_stats_before)
        offload_step_delta = self._stats_delta(offload_stats_now, self._online_offload_stats_last)
        self._online_offload_stats_last = dict(offload_stats_now)
        decode_window_status = self.scheduler.get_decode_window_status()
        ready_stats = self._queue_residency_stats(self.ready_decode)
        decode_stats_live = self._queue_residency_stats(self.decode_active)
        return {
            'step': int(self._online_step),
            'waiting_queue': int(len(self.waiting_queue)),
            'prefill_active': int(len(self.prefill_active)),
            'ready_decode': int(len(self.ready_decode)),
            'decode_active': int(len(self.decode_active)),
            'finished_queue': int(len(self.finished_queue)),
            'admitted': int(admitted),
            'ready_activated': int(ready_activated),
            'ready_decode_on_gpu': int(ready_stats.get('seq_on_gpu', 0)),
            'ready_decode_on_cpu': int(ready_stats.get('seq_on_cpu', 0)),
            'ready_decode_pinned': int(ready_stats.get('seq_pinned', 0)),
            'ready_decode_offload_inflight': int(ready_stats.get('seq_offload_inflight', 0)),
            'ready_decode_prefetch_inflight': int(ready_stats.get('seq_prefetch_inflight', 0)),
            'ready_decode_missing': int(ready_stats.get('seq_missing', 0)),
            'ready_decode_logical_blocks': int(ready_stats.get('logical_blocks', 0)),
            'ready_decode_materialized_blocks': int(ready_stats.get('materialized_blocks', 0)),
            'ready_decode_resident_blocks': int(ready_stats.get('resident_blocks', 0)),
            'decode_active_on_gpu': int(decode_stats_live.get('seq_on_gpu', 0)),
            'decode_active_on_cpu': int(decode_stats_live.get('seq_on_cpu', 0)),
            'decode_active_pinned': int(decode_stats_live.get('seq_pinned', 0)),
            'decode_active_offload_inflight': int(decode_stats_live.get('seq_offload_inflight', 0)),
            'decode_active_prefetch_inflight': int(decode_stats_live.get('seq_prefetch_inflight', 0)),
            'decode_active_missing': int(decode_stats_live.get('seq_missing', 0)),
            'decode_active_logical_blocks': int(decode_stats_live.get('logical_blocks', 0)),
            'decode_active_materialized_blocks': int(decode_stats_live.get('materialized_blocks', 0)),
            'decode_active_resident_blocks': int(decode_stats_live.get('resident_blocks', 0)),
            'prefill_tokens': int(prefill_stats.get('prefill_tokens', 0)),
            'prefill_finished': int(prefill_stats.get('prefill_finished', 0)),
            'prefill_failed': int(prefill_stats.get('prefill_failed', 0)),
            'decode_scheduled': int(decode_stats.get('decode_scheduled', 0)),
            'decode_tokens': int(decode_stats.get('decode_tokens', 0)),
            'decode_microbatches': int(decode_stats.get('decode_microbatches', 0)),
            'missing_blocks_scheduled': int(self._online_last_missing_blocks_scheduled),
            'materialized_blocks': int(self._online_last_materialized_blocks_scheduled),
            'decode_active_cap': int(self._online_decode_active_cap),
            'decode_active_cap_boot': int(self._online_decode_active_cap_boot),
            'decode_active_cap_min_seen': int(self._online_decode_active_cap_min_seen),
            'thrash_win16': float(self.scheduler.get_thrash_win16()),
            'decode_min_n_free': int(self._online_decode_min_n_free),
            'prefill_min_n_free': int(self._online_prefill_min_n_free),
            'global_min_n_free': int(self._online_global_min_n_free),
            'kv_total_blocks': int(self._online_kv_total_blocks),
            'kv_peak_used_blocks': int(self._online_kv_peak_used_blocks),
            'decode_append_fail_count': int(self._online_decode_append_fail_count),
            'decode_backpressure_events': int(self._online_decode_backpressure_events),
            'decode_memory_cap_events': int(self._online_decode_memory_cap_events),
            'decode_memory_cap_min_batch': int(self._online_decode_memory_cap_min_batch),
            'decode_memory_guard_target_batch_last': int(self._online_decode_memory_guard_target_batch_last),
            'decode_memory_guard_source_batch_last': int(self._online_decode_memory_guard_source_batch_last),
            'decode_memory_guard_free_gb_last': round(float(self._online_decode_memory_guard_free_gb_last), 4),
            'decode_memory_guard_budget_gb_last': round(float(self._online_decode_memory_guard_budget_gb_last), 4),
            'decode_memory_guard_reserve_gb_last': round(float(self._online_decode_memory_guard_reserve_gb_last), 4),
            'decode_memory_guard_hard_guard_gb': round(float(self._online_decode_memory_guard_hard_guard_gb), 4),
            'guard_seen_count': int(self._online_guard_seen_count),
            'guard_effective_shrink_count': int(self._online_guard_effective_shrink_count),
            'guard_strong_shrink_count': int(self._online_guard_strong_shrink_count),
            'guard_target_batch_min': int(self._online_guard_target_batch_min),
            'guard_source_batch_max': int(self._online_guard_source_batch_max),
            'decode_length_bucketed_steps': int(self._online_decode_length_bucketed_steps),
            'decode_length_bucket_subbatch_count': int(self._online_decode_length_bucket_subbatch_count),
            'decode_length_bucket_singleton_count': int(self._online_decode_length_bucket_singleton_count),
            'decode_length_bucket_max_trigger_ratio': round(float(self._online_decode_length_bucket_max_trigger_ratio), 4),
            'decode_memory_est_peak_max_gb': round(float(self._online_decode_memory_est_peak_max_gb), 4),
            'decode_memory_est_peak_maxlen_gb': round(float(self._online_decode_memory_est_peak_maxlen_gb), 4),
            'decode_memory_est_peak_sumlen_gb': round(float(self._online_decode_memory_est_peak_sumlen_gb), 4),
            'decode_memory_aware_cap_enabled': int(self._online_decode_memory_aware_cap_enabled),
            'decode_memory_aware_margin_gb': round(float(self._online_decode_memory_aware_margin_gb), 4),
            'decode_memory_aware_peak_factor': round(float(self._online_decode_memory_aware_peak_factor), 4),
            'prefill_backpressure_events': int(self._online_prefill_backpressure_events),
            'prefill_batch_failed_steps': int(self._online_prefill_batch_failed_steps),
            'prefill_chunk_failed_steps': int(self._online_prefill_chunk_failed_steps),
            'prefill_activate_failed_steps': int(self._online_prefill_activate_failed_steps),
            'prefill_no_progress_steps': int(self._online_prefill_no_progress_steps),
            'prefill_no_progress_peak': int(self._online_prefill_no_progress_peak),
            'prefill_no_progress_fail_count': int(self._online_prefill_no_progress_fail_count),
            'prefill_batch_merge_peak_gb': round(float(self._online_prefill_batch_merge_peak_gb), 4),
            'prefill_batch_input_peak_gb': round(float(self._online_prefill_batch_input_peak_gb), 4),
            'prefill_batch_forward_peak_gb': round(float(self._online_prefill_batch_forward_peak_gb), 4),
            'prefill_batch_slice_peak_gb': round(float(self._online_prefill_batch_slice_peak_gb), 4),
            'prefill_chunk_input_peak_gb': round(float(self._online_prefill_chunk_input_peak_gb), 4),
            'prefill_chunk_forward_peak_gb': round(float(self._online_prefill_chunk_forward_peak_gb), 4),
            'decode_retry_timeout_fail_count': int(self._online_decode_retry_timeout_fail_count),
            'decode_retry_timeout_seq_ids': list(self._online_decode_retry_timeout_seq_ids),
            'decode_no_progress_steps': int(self._online_decode_no_progress_steps),
            'decode_no_progress_streak': int(self._online_decode_no_progress_streak),
            'n_free': int(decode_window_status.get('n_free', 0)),
            'wm_low': int(decode_window_status.get('wm_low', 0)),
            'wm_high': int(decode_window_status.get('wm_high', 0)),
            'p2_low_threshold': int(decode_window_status.get('p2_low_threshold', 0)),
            'p2_target_free_blocks': int(decode_window_status.get('p2_target_free_blocks', 0)),
            'p2_attempts': int(decode_window_status.get('p2_attempts', 0)),
            'p2_successes': int(decode_window_status.get('p2_successes', 0)),
            'p2_fail_streak': int(decode_window_status.get('p2_fail_streak', 0)),
            'p2_last_attempted': int(decode_window_status.get('p2_last_attempted', 0)),
            'p2_last_success': int(decode_window_status.get('p2_last_success', 0)),
            'p2_last_no_candidate': int(decode_window_status.get('p2_last_no_candidate', 0)),
            'p2_last_candidate_count': int(decode_window_status.get('p2_last_candidate_count', 0)),
            'p2_reject_deferred': int(self._online_p2_reject_deferred),
            'p2_reject_no_resident': int(self._online_p2_reject_no_resident),
            'p2_reject_protected': int(self._online_p2_reject_protected),
            'p2_ready_protected_ignored': int(self._online_p2_ready_protected_ignored),
            'p2_reject_active_floor': int(self._online_p2_reject_active_floor),
            'p2_reject_plan_empty': int(self._online_p2_reject_plan_empty),
            'p2_managed_active': int(decode_window_status.get('p2_managed_active', 0)),
            'p2_recover_streak': int(decode_window_status.get('p2_recover_streak', 0)),
            'p2_active_steps': int(decode_window_status.get('p2_active_steps', 0)),
            'p2_candidate_steps': int(decode_window_status.get('p2_candidate_steps', 0)),
            'p2_recovery_fail_windows': int(decode_window_status.get('p2_recovery_fail_windows', 0)),
            'p2_no_candidate_steps': int(decode_window_status.get('p2_no_candidate_steps', 0)),
            'p2_attempted_steps': int(decode_window_status.get('p2_attempted_steps', 0)),
            'p2_success_steps': int(decode_window_status.get('p2_success_steps', 0)),
            'p2_ready_candidate_steps': int(decode_window_status.get('p2_ready_candidate_steps', 0)),
            'p2_decode_candidate_steps': int(decode_window_status.get('p2_decode_candidate_steps', 0)),
            'p2_expected_reclaim_blocks': int(decode_window_status.get('p2_expected_reclaim_blocks', 0)),
            'p2_ready_offload_blocks_total': int(decode_window_status.get('p2_ready_offload_blocks_total', 0)),
            'p2_ready_offload_blocks_last': int(decode_window_status.get('p2_ready_offload_blocks_last', 0)),
            'p2_ready_offload_sequence_steps': int(decode_window_status.get('p2_ready_offload_sequence_steps', 0)),
            'p2_ready_offload_decode_steps': int(decode_window_status.get('p2_ready_offload_decode_steps', 0)),
            'p2_ready_sequences_selected_per_step': int(decode_window_status.get('p2_ready_sequences_selected_per_step', 0)),
            'p2_ready_offload_blocks_per_step': int(decode_window_status.get('p2_ready_offload_blocks_per_step', 0)),
            'p2_ready_target_reclaim_blocks': int(decode_window_status.get('p2_ready_target_reclaim_blocks', 0)),
            'p2_ready_actual_reclaim_blocks': int(decode_window_status.get('p2_ready_actual_reclaim_blocks', 0)),
            'p2_ready_stop_reason': str(decode_window_status.get('p2_ready_stop_reason', '')),
            'p2_ready_stop_target_reached_steps': int(decode_window_status.get('p2_ready_stop_target_reached_steps', 0)),
            'p2_ready_stop_sequence_cap_reached_steps': int(decode_window_status.get('p2_ready_stop_sequence_cap_reached_steps', 0)),
            'p2_ready_stop_block_cap_reached_steps': int(decode_window_status.get('p2_ready_stop_block_cap_reached_steps', 0)),
            'p2_ready_stop_low_benefit_skip_steps': int(decode_window_status.get('p2_ready_stop_low_benefit_skip_steps', 0)),
            'p2_ready_stop_not_needed_steps': int(decode_window_status.get('p2_ready_stop_not_needed_steps', 0)),
            'p2_ready_stop_no_ready_candidate_steps': int(decode_window_status.get('p2_ready_stop_no_ready_candidate_steps', 0)),
            'kv_admission_enabled': int(bool(self.kv_admission_enabled)),
            'kv_admission_blocked_steps': int(self._online_kv_admission_blocked_steps),
            'kv_admission_blocked_requests': int(self._online_kv_admission_blocked_requests),
            'kv_admission_last_free_blocks': int(self._online_kv_admission_last_free_blocks),
            'kv_admission_last_required_blocks': int(self._online_kv_admission_last_required_blocks),
            'kv_admission_last_workload_demand_blocks': int(self._online_kv_admission_last_workload_demand_blocks),
            'kv_admission_last_reserved_blocks': int(self._online_kv_admission_last_reserved_blocks),
            'kv_admission_last_margin_blocks': int(self._online_kv_admission_last_margin_blocks),
            'kv_admission_last_total_required_blocks': int(self._online_kv_admission_last_total_required_blocks),
            'kv_admission_last_prompt_blocks': int(self._online_kv_admission_last_prompt_blocks),
            'kv_admission_last_prompt_resident_blocks': int(self._online_kv_admission_last_prompt_resident_blocks),
            'kv_admission_last_request_blocks': int(self._online_kv_admission_last_request_blocks),
            'kv_admission_last_pending_prompt_blocks': int(self._online_kv_admission_last_pending_prompt_blocks),
            'kv_admission_last_pending_output_blocks': int(self._online_kv_admission_last_pending_output_blocks),
            'kv_admission_last_output_reserve_tokens': int(self._online_kv_admission_last_output_reserve_tokens),
            'kv_admission_last_output_reserve_blocks': int(self._online_kv_admission_last_output_reserve_blocks),
            'kv_admission_last_allowed': int(self._online_kv_admission_last_allowed),
            'kv_admission_include_low_watermark': int(bool(getattr(self, 'kv_admission_include_low_watermark', True))),
            'online_prefill_admission_enabled': int(bool(self.online_prefill_admission_enabled)),
            'online_prefill_admission_blocked_steps': int(self._online_prefill_admission_blocked_steps),
            'online_prefill_admission_blocked_requests': int(self._online_prefill_admission_blocked_requests),
            'online_prefill_admission_last_reason': str(self._online_prefill_admission_last_reason),
            'online_prefill_admission_last_prompt_len': int(self._online_prefill_admission_last_prompt_len),
            'online_prefill_admission_last_bucket': str(self._online_prefill_admission_last_bucket),
            'online_prefill_admission_last_cuda_free_gb': round(float(self._online_prefill_admission_last_cuda_free_gb), 4),
            'online_prefill_admission_last_active_short': int(self._online_prefill_admission_last_active_short),
            'online_prefill_admission_last_active_mid': int(self._online_prefill_admission_last_active_mid),
            'online_prefill_admission_last_active_long': int(self._online_prefill_admission_last_active_long),
            'online_prefill_admission_last_cap': int(self._online_prefill_admission_last_cap),
            'online_prefill_active_token_budget': int(self.online_prefill_active_token_budget),
            'online_prefill_admission_last_active_tokens': int(self._online_prefill_admission_last_active_tokens),
            'online_prefill_admission_last_projected_tokens': int(self._online_prefill_admission_last_projected_tokens),
            'online_prefill_admission_last_token_budget': int(self._online_prefill_admission_last_token_budget),
            'online_prefill_admission_token_budget_blocked_steps': int(self._online_prefill_admission_token_budget_blocked_steps),
            'online_prefill_admission_token_budget_blocked_requests': int(self._online_prefill_admission_token_budget_blocked_requests),
            'online_prefill_admission_last_allowed': int(self._online_prefill_admission_last_allowed),
            'online_prefill_chunk_floor_pause_steps': int(self._online_prefill_chunk_floor_pause_steps),
            'online_prefill_chunk_floor_last_chunk_cap': int(self._online_prefill_chunk_floor_last_chunk_cap),
            'p2_gain_success_steps': int(decode_window_status.get('p2_gain_success_steps', 0)),
            'p2_gain_fail_steps': int(decode_window_status.get('p2_gain_fail_steps', 0)),
            'p2_skipped_low_benefit_steps': int(decode_window_status.get('p2_skipped_low_benefit_steps', 0)),
            'first_p2_step': int(decode_window_status.get('first_p2_step', 0)),
            'cuda_free_post_cleanup_min_gb': round(float(self._online_cuda_free_post_cleanup_gb), 4),
            'cuda_free_post_cleanup_last_gb': round(float(self._online_cuda_free_post_cleanup_last_gb), 4),
            'decode_cuda_free_post_cleanup_recent_min_gb': round(
                float(self.scheduler.decode_cuda_free_post_cleanup_recent_min_gb()),
                4,
            ) if math.isfinite(float(self.scheduler.decode_cuda_free_post_cleanup_recent_min_gb())) else 0.0,
            'p2_cuda_pressure_signal_steps': int(decode_window_status.get('p2_cuda_pressure_signal_steps', 0)),
            'cuda_total_gb': round(float(self._online_cuda_total_bytes) / 1024**3, 4) if self._online_cuda_total_bytes else 0.0,
            'cuda_free_min_gb': round(float(self._online_cuda_free_min_bytes) / 1024**3, 4) if self._online_cuda_total_bytes else 0.0,
            'cuda_alloc_peak_gb': round(float(self._online_cuda_alloc_peak_bytes) / 1024**3, 4),
            'cuda_reserved_peak_gb': round(float(self._online_cuda_reserved_peak_bytes) / 1024**3, 4),
            'offloader_delta_step': dict(offload_step_delta),
            'offloader_delta_cum': dict(offload_cum_delta),
            'decode_paged_flash_enabled': int(bool(self.decode_paged_flash_enabled)),
            'decode_paged_flash_active': int(bool(self.decode_paged_flash_active)),
            'decode_paged_flash_reason': str(self.decode_paged_flash_reason),
            'decode_paged_flash_strict': int(bool(self.decode_paged_flash_strict)),
            'decode_page16_native_strict': int(bool(self.decode_page16_native_strict)),
            'kv_logical_block_size': int(self.scheduler.pool.B),
            'flash_attn_enabled': int(bool(self.flash_attn_enabled)),
            'selected_writeback_enabled': int(decode_window_status.get('selected_writeback_enabled', 0)),
            'prefill_writeback_backend': str(decode_window_status.get('prefill_writeback_backend', '')),
            'gpu_selected_writeback_steps': int(decode_window_status.get('gpu_selected_writeback_steps', 0)),
            'cpu_selected_compaction_steps': int(decode_window_status.get('cpu_selected_compaction_steps', 0)),
            'gpu_writeback_oom_fallbacks': int(decode_window_status.get('gpu_writeback_oom_fallbacks', 0)),
            'writeback_transaction_rollbacks': int(decode_window_status.get('writeback_transaction_rollbacks', 0)),
            'raw_kv_cpu_stash_bytes': int(decode_window_status.get('raw_kv_cpu_stash_bytes', 0)),
            'selected_global_block_count': int(decode_window_status.get('selected_global_block_count', 0)),
            'writeback_est_required_gb': round(float(decode_window_status.get('writeback_est_required_gb_x1000', 0)) / 1000.0, 4),
            'writeback_free_gb': round(float(decode_window_status.get('writeback_free_gb_x1000', 0)) / 1000.0, 4),
            'writeback_block_selection_shared_layers': int(decode_window_status.get('writeback_block_selection_shared_layers', 0)),
            'score_full_attention_materialized': int(decode_window_status.get('score_full_attention_materialized', 0)),
            'retain_ratio': float(getattr(getattr(self.scheduler, 'snapkv', None), 'retain_ratio', 0.0) or 0.0),
            'retain_budget_tokens': int(getattr(getattr(self.scheduler, 'snapkv', None), 'retain_budget_tokens', 0) or 0),
            'compression_mode': str(getattr(getattr(self.scheduler, 'snapkv', None), 'compression_mode', 'ratio')),
            'effective_retained_tokens': int((getattr(getattr(self.scheduler, 'snapkv', None), 'last_debug', {}) or {}).get('effective_retained_tokens', 0) or 0),
            'retained_block_count': int((getattr(getattr(self.scheduler, 'snapkv', None), 'last_debug', {}) or {}).get('retained_block_count', 0) or 0),
            'decode_path_mode': str(self.decode_path_mode),
            'decode_path_selected': str(decode_stats.get('decode_path_selected', self._decode_path_selected_label())),
            'decode_backend': str(decode_stats.get('decode_backend', self._decode_backend_label())),
            'decode_paged_direct_steps': int(
                decode_stats.get('decode_paged_direct_steps', self._online_decode_paged_direct_steps)
            ),
            'decode_page16_native_steps': int(
                decode_stats.get('decode_page16_native_steps', self._online_decode_page16_native_steps)
            ),
            'decode_rebuild_steps': int(decode_stats.get('decode_rebuild_steps', self._online_decode_rebuild_steps)),
            'decode_materialize_kv_bytes': int(
                decode_stats.get('decode_materialize_kv_bytes', self._online_decode_materialize_kv_bytes)
            ),
            'decode_paged_direct_blocked_reason': str(
                decode_stats.get(
                    'decode_paged_direct_blocked_reason',
                    self._online_decode_paged_direct_blocked_reason,
                )
            ),
            'decode_paged_direct_resident_miss_steps': int(
                decode_stats.get(
                    'decode_paged_direct_resident_miss_steps',
                    self._online_decode_paged_direct_resident_miss_steps,
                )
            ),
            'decode_page16_native_blocked_reason': str(
                decode_stats.get(
                    'decode_page16_native_blocked_reason',
                    self._online_decode_page16_native_blocked_reason,
                )
            ),
            'decode_page16_native_resident_miss_steps': int(
                decode_stats.get(
                    'decode_page16_native_resident_miss_steps',
                    self._online_decode_page16_native_resident_miss_steps,
                )
            ),
            'decode_page16_native_kernel_ms': round(
                float(decode_stats.get('decode_page16_native_kernel_ms', self._online_decode_page16_native_kernel_ms)),
                3,
            ),
            'decode_path_fallback_count': int(
                decode_stats.get('decode_path_fallback_count', self._online_decode_path_fallback_count)
            ),
            'decode_path_fallback_reason_topk': decode_stats.get(
                'decode_path_fallback_reason_topk',
                self._decode_path_fallback_reason_topk(),
            ),
            'prefill_pause_steps': int(self._online_prefill_pause_steps),
            'priority_retry_queue': int(len(self._priority_retry)),
            'step_ms': round(step_ms, 3),
        }

    def _generate_with_online(
        self,
        prompts: List[str],
        return_metrics: bool = False,
        return_details: bool = False,
        step_callback: Optional[Any] = None,
    ):
        if self.has_pending_requests():
            raise RuntimeError("engine has pending online requests; drain them before calling generate()")
        if not prompts:
            if return_metrics and return_details:
                return [], {}, {'token_ids': [], 'token_logprobs': []}
            if return_metrics:
                return [], {}
            if return_details:
                return [], {'token_ids': [], 'token_logprobs': []}
            return []

        self._reset_online_runtime(clear_request_counter=True)
        self._return_details_online = bool(return_details)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._online_total_t0 = time.perf_counter()

        rid_order: List[int] = []
        for i, prompt in enumerate(prompts):
            rid = self.submit(prompt, request_id=i, max_new_tokens=self.max_new_tokens)
            rid_order.append(rid)

        while self.has_pending_requests():
            step_stats = self.step()
            if step_callback is not None:
                try:
                    step_callback(dict(step_stats))
                except Exception:
                    pass

        finished_map = {int(item['request_id']): item for item in self.collect_finished()}
        outputs = [finished_map.get(rid, {}).get('output', '') for rid in rid_order]
        failed_request_errors = [str(finished_map.get(rid, {}).get('error', '') or '').strip() for rid in rid_order]
        if not return_metrics and not return_details:
            return outputs

        sched = self.scheduler
        total_ms = (time.perf_counter() - self._online_total_t0) * 1000.0
        decode_ms = sum(self._online_decode_step_lat_ms)
        generated_tokens = int(self._online_generated_tokens)
        stats_after = sched.offloader.get_stats(reset=False)
        stats_delta = self._stats_delta(stats_after, self._online_offload_stats_before)
        thrash_win16 = float(sched.get_thrash_win16())
        tps = generated_tokens / max(1e-6, total_ms / 1000.0)
        avg_mb = (
            float(sum(self._online_decode_microbatch_sizes)) / float(len(self._online_decode_microbatch_sizes))
            if self._online_decode_microbatch_sizes else 0.0
        )

        decode_window_status = self.scheduler.get_decode_window_status()
        metrics = {
            'num_prompts': int(len(prompts)),
            'generated_tokens': int(generated_tokens),
            'prefill_ms': round(self._online_prefill_ms, 3),
            'decode_ms': round(decode_ms, 3),
            'total_ms': round(total_ms, 3),
            'tokens_per_sec': round(tps, 4),
            'failed_request_errors': list(failed_request_errors),
            'decode_steps': int(len(self._online_decode_step_lat_ms)),
            'decode_step_p50_ms': round(self._percentile(self._online_decode_step_lat_ms, 0.50), 3),
            'decode_step_p95_ms': round(self._percentile(self._online_decode_step_lat_ms, 0.95), 3),
            'offloader_delta': stats_delta,
            'thrash_win16': round(thrash_win16, 6),
            'prefill_min_n_free': int(self._online_prefill_min_n_free),
            'global_min_n_free': int(self._online_global_min_n_free),
            'kv_total_blocks': int(self._online_kv_total_blocks),
            'kv_peak_used_blocks': int(self._online_kv_peak_used_blocks),
            'decode_min_n_free': int(self._online_decode_min_n_free),
            'decode_active_cap_boot': int(self._online_decode_active_cap_boot),
            'decode_active_cap_final': int(self._online_decode_active_cap),
            'decode_active_cap_min_seen': int(self._online_decode_active_cap_min_seen),
            'decode_append_fail_count': int(self._online_decode_append_fail_count),
            'decode_backpressure_events': int(self._online_decode_backpressure_events),
            'decode_memory_cap_events': int(self._online_decode_memory_cap_events),
            'decode_memory_cap_min_batch': int(self._online_decode_memory_cap_min_batch),
            'decode_memory_guard_target_batch_last': int(self._online_decode_memory_guard_target_batch_last),
            'decode_memory_guard_source_batch_last': int(self._online_decode_memory_guard_source_batch_last),
            'decode_memory_guard_free_gb_last': round(float(self._online_decode_memory_guard_free_gb_last), 4),
            'decode_memory_guard_budget_gb_last': round(float(self._online_decode_memory_guard_budget_gb_last), 4),
            'decode_memory_guard_reserve_gb_last': round(float(self._online_decode_memory_guard_reserve_gb_last), 4),
            'decode_memory_guard_hard_guard_gb': round(float(self._online_decode_memory_guard_hard_guard_gb), 4),
            'guard_seen_count': int(self._online_guard_seen_count),
            'guard_effective_shrink_count': int(self._online_guard_effective_shrink_count),
            'guard_strong_shrink_count': int(self._online_guard_strong_shrink_count),
            'guard_target_batch_min': int(self._online_guard_target_batch_min),
            'guard_source_batch_max': int(self._online_guard_source_batch_max),
            'decode_length_bucketed_steps': int(self._online_decode_length_bucketed_steps),
            'decode_length_bucket_subbatch_count': int(self._online_decode_length_bucket_subbatch_count),
            'decode_length_bucket_singleton_count': int(self._online_decode_length_bucket_singleton_count),
            'decode_length_bucket_max_trigger_ratio': round(float(self._online_decode_length_bucket_max_trigger_ratio), 4),
            'decode_memory_est_peak_max_gb': round(float(self._online_decode_memory_est_peak_max_gb), 4),
            'decode_memory_est_peak_maxlen_gb': round(float(self._online_decode_memory_est_peak_maxlen_gb), 4),
            'decode_memory_est_peak_sumlen_gb': round(float(self._online_decode_memory_est_peak_sumlen_gb), 4),
            'decode_memory_aware_cap_enabled': int(self._online_decode_memory_aware_cap_enabled),
            'decode_memory_aware_margin_gb': round(float(self._online_decode_memory_aware_margin_gb), 4),
            'decode_memory_aware_peak_factor': round(float(self._online_decode_memory_aware_peak_factor), 4),
            'prefill_backpressure_events': int(self._online_prefill_backpressure_events),
            'prefill_batch_failed_steps': int(self._online_prefill_batch_failed_steps),
            'prefill_chunk_failed_steps': int(self._online_prefill_chunk_failed_steps),
            'prefill_activate_failed_steps': int(self._online_prefill_activate_failed_steps),
            'prefill_no_progress_steps': int(self._online_prefill_no_progress_steps),
            'prefill_no_progress_peak': int(self._online_prefill_no_progress_peak),
            'prefill_no_progress_fail_count': int(self._online_prefill_no_progress_fail_count),
            'decode_retry_timeout_fail_count': int(self._online_decode_retry_timeout_fail_count),
            'decode_retry_timeout_seq_ids': list(self._online_decode_retry_timeout_seq_ids),
            'decode_no_progress_steps': int(self._online_decode_no_progress_steps),
            'decode_no_progress_peak': int(self._online_decode_no_progress_peak),
            'decode_paged_flash_enabled': int(bool(self.decode_paged_flash_enabled)),
            'decode_paged_flash_active': int(bool(self.decode_paged_flash_active)),
            'decode_paged_flash_reason': str(self.decode_paged_flash_reason),
            'decode_paged_flash_strict': int(bool(self.decode_paged_flash_strict)),
            'decode_page16_native_strict': int(bool(self.decode_page16_native_strict)),
            'kv_logical_block_size': int(self.scheduler.pool.B),
            'flash_attn_enabled': int(bool(self.flash_attn_enabled)),
            'selected_writeback_enabled': int(decode_window_status.get('selected_writeback_enabled', 0)),
            'prefill_writeback_backend': str(decode_window_status.get('prefill_writeback_backend', '')),
            'gpu_selected_writeback_steps': int(decode_window_status.get('gpu_selected_writeback_steps', 0)),
            'cpu_selected_compaction_steps': int(decode_window_status.get('cpu_selected_compaction_steps', 0)),
            'gpu_writeback_oom_fallbacks': int(decode_window_status.get('gpu_writeback_oom_fallbacks', 0)),
            'writeback_transaction_rollbacks': int(decode_window_status.get('writeback_transaction_rollbacks', 0)),
            'raw_kv_cpu_stash_bytes': int(decode_window_status.get('raw_kv_cpu_stash_bytes', 0)),
            'selected_global_block_count': int(decode_window_status.get('selected_global_block_count', 0)),
            'writeback_est_required_gb': round(float(decode_window_status.get('writeback_est_required_gb_x1000', 0)) / 1000.0, 4),
            'writeback_free_gb': round(float(decode_window_status.get('writeback_free_gb_x1000', 0)) / 1000.0, 4),
            'writeback_block_selection_shared_layers': int(decode_window_status.get('writeback_block_selection_shared_layers', 0)),
            'score_full_attention_materialized': int(decode_window_status.get('score_full_attention_materialized', 0)),
            'decode_path_mode': str(self.decode_path_mode),
            'decode_path_selected': str(self._decode_path_selected_label()),
            'decode_backend': str(self._decode_backend_label()),
            'decode_paged_direct_steps': int(self._online_decode_paged_direct_steps),
            'decode_page16_native_steps': int(self._online_decode_page16_native_steps),
            'decode_rebuild_steps': int(self._online_decode_rebuild_steps),
            'decode_materialize_kv_bytes': int(self._online_decode_materialize_kv_bytes),
            'decode_paged_direct_blocked_reason': str(self._online_decode_paged_direct_blocked_reason),
            'decode_paged_direct_resident_miss_steps': int(self._online_decode_paged_direct_resident_miss_steps),
            'decode_page16_native_blocked_reason': str(self._online_decode_page16_native_blocked_reason),
            'decode_page16_native_resident_miss_steps': int(self._online_decode_page16_native_resident_miss_steps),
            'decode_page16_native_kernel_ms': round(float(self._online_decode_page16_native_kernel_ms), 3),
            'decode_path_fallback_count': int(self._online_decode_path_fallback_count),
            'decode_path_fallback_reason_topk': self._decode_path_fallback_reason_topk(),
            'avg_decode_microbatch_size': round(avg_mb, 4),
            'cuda_free_post_cleanup_min_gb': round(float(self._online_cuda_free_post_cleanup_gb), 4),
            'cuda_free_post_cleanup_last_gb': round(float(self._online_cuda_free_post_cleanup_last_gb), 4),
            'decode_cuda_free_post_cleanup_recent_min_gb': round(
                float(self._online_decode_cuda_free_post_cleanup_recent_min_gb),
                4,
            ) if math.isfinite(float(self._online_decode_cuda_free_post_cleanup_recent_min_gb)) else 0.0,
            'cuda_total_gb': round(float(self._online_cuda_total_bytes) / 1024**3, 4) if self._online_cuda_total_bytes else 0.0,
            'cuda_free_min_gb': round(float(self._online_cuda_free_min_bytes) / 1024**3, 4) if self._online_cuda_total_bytes else 0.0,
            'cuda_alloc_peak_gb': round(float(self._online_cuda_alloc_peak_bytes) / 1024**3, 4),
            'cuda_reserved_peak_gb': round(float(self._online_cuda_reserved_peak_bytes) / 1024**3, 4),
            'p2_cuda_pressure_signal_steps': int(decode_window_status.get('p2_cuda_pressure_signal_steps', 0)),
            'p2_active_steps': int(decode_window_status.get('p2_active_steps', 0)),
            'p2_candidate_steps': int(decode_window_status.get('p2_candidate_steps', 0)),
            'p2_recovery_fail_windows': int(decode_window_status.get('p2_recovery_fail_windows', 0)),
            'p2_no_candidate_steps': int(decode_window_status.get('p2_no_candidate_steps', 0)),
            'p2_attempted_steps': int(decode_window_status.get('p2_attempted_steps', 0)),
            'p2_success_steps': int(decode_window_status.get('p2_success_steps', 0)),
            'p2_target_free_blocks': int(decode_window_status.get('p2_target_free_blocks', 0)),
            'p2_attempts': int(decode_window_status.get('p2_attempts', 0)),
            'p2_successes': int(decode_window_status.get('p2_successes', 0)),
            'p2_fail_streak': int(decode_window_status.get('p2_fail_streak', 0)),
            'p2_last_attempted': int(decode_window_status.get('p2_last_attempted', 0)),
            'p2_last_success': int(decode_window_status.get('p2_last_success', 0)),
            'p2_last_candidate_count': int(decode_window_status.get('p2_last_candidate_count', 0)),
            'p2_last_no_candidate': int(decode_window_status.get('p2_last_no_candidate', 0)),
            'p2_reject_deferred': int(self._online_p2_reject_deferred),
            'p2_reject_no_resident': int(self._online_p2_reject_no_resident),
            'p2_reject_protected': int(self._online_p2_reject_protected),
            'p2_ready_protected_ignored': int(self._online_p2_ready_protected_ignored),
            'p2_reject_active_floor': int(self._online_p2_reject_active_floor),
            'p2_reject_plan_empty': int(self._online_p2_reject_plan_empty),
            'p2_managed_active': int(decode_window_status.get('p2_managed_active', 0)),
            'p2_recover_streak': int(decode_window_status.get('p2_recover_streak', 0)),
            'p2_ready_candidate_steps': int(decode_window_status.get('p2_ready_candidate_steps', 0)),
            'p2_decode_candidate_steps': int(decode_window_status.get('p2_decode_candidate_steps', 0)),
            'p2_expected_reclaim_blocks': int(decode_window_status.get('p2_expected_reclaim_blocks', 0)),
            'p2_ready_offload_blocks_total': int(decode_window_status.get('p2_ready_offload_blocks_total', 0)),
            'p2_ready_offload_blocks_last': int(decode_window_status.get('p2_ready_offload_blocks_last', 0)),
            'p2_ready_offload_sequence_steps': int(decode_window_status.get('p2_ready_offload_sequence_steps', 0)),
            'p2_ready_offload_decode_steps': int(decode_window_status.get('p2_ready_offload_decode_steps', 0)),
            'p2_ready_sequences_selected_per_step': int(decode_window_status.get('p2_ready_sequences_selected_per_step', 0)),
            'p2_ready_offload_blocks_per_step': int(decode_window_status.get('p2_ready_offload_blocks_per_step', 0)),
            'p2_ready_target_reclaim_blocks': int(decode_window_status.get('p2_ready_target_reclaim_blocks', 0)),
            'p2_ready_actual_reclaim_blocks': int(decode_window_status.get('p2_ready_actual_reclaim_blocks', 0)),
            'p2_ready_stop_reason': str(decode_window_status.get('p2_ready_stop_reason', '')),
            'p2_ready_stop_target_reached_steps': int(decode_window_status.get('p2_ready_stop_target_reached_steps', 0)),
            'p2_ready_stop_sequence_cap_reached_steps': int(decode_window_status.get('p2_ready_stop_sequence_cap_reached_steps', 0)),
            'p2_ready_stop_block_cap_reached_steps': int(decode_window_status.get('p2_ready_stop_block_cap_reached_steps', 0)),
            'p2_ready_stop_low_benefit_skip_steps': int(decode_window_status.get('p2_ready_stop_low_benefit_skip_steps', 0)),
            'p2_ready_stop_not_needed_steps': int(decode_window_status.get('p2_ready_stop_not_needed_steps', 0)),
            'p2_ready_stop_no_ready_candidate_steps': int(decode_window_status.get('p2_ready_stop_no_ready_candidate_steps', 0)),
            'kv_admission_enabled': int(bool(self.kv_admission_enabled)),
            'kv_admission_blocked_steps': int(self._online_kv_admission_blocked_steps),
            'kv_admission_blocked_requests': int(self._online_kv_admission_blocked_requests),
            'kv_admission_last_free_blocks': int(self._online_kv_admission_last_free_blocks),
            'kv_admission_last_required_blocks': int(self._online_kv_admission_last_required_blocks),
            'kv_admission_last_workload_demand_blocks': int(self._online_kv_admission_last_workload_demand_blocks),
            'kv_admission_last_reserved_blocks': int(self._online_kv_admission_last_reserved_blocks),
            'kv_admission_last_margin_blocks': int(self._online_kv_admission_last_margin_blocks),
            'kv_admission_last_total_required_blocks': int(self._online_kv_admission_last_total_required_blocks),
            'kv_admission_last_prompt_blocks': int(self._online_kv_admission_last_prompt_blocks),
            'kv_admission_last_prompt_resident_blocks': int(self._online_kv_admission_last_prompt_resident_blocks),
            'kv_admission_last_request_blocks': int(self._online_kv_admission_last_request_blocks),
            'kv_admission_last_pending_prompt_blocks': int(self._online_kv_admission_last_pending_prompt_blocks),
            'kv_admission_last_pending_output_blocks': int(self._online_kv_admission_last_pending_output_blocks),
            'kv_admission_last_output_reserve_tokens': int(self._online_kv_admission_last_output_reserve_tokens),
            'kv_admission_last_output_reserve_blocks': int(self._online_kv_admission_last_output_reserve_blocks),
            'kv_admission_last_allowed': int(self._online_kv_admission_last_allowed),
            'online_prefill_admission_enabled': int(bool(self.online_prefill_admission_enabled)),
            'online_prefill_admission_blocked_steps': int(self._online_prefill_admission_blocked_steps),
            'online_prefill_admission_blocked_requests': int(self._online_prefill_admission_blocked_requests),
            'online_prefill_admission_last_reason': str(self._online_prefill_admission_last_reason),
            'online_prefill_admission_last_prompt_len': int(self._online_prefill_admission_last_prompt_len),
            'online_prefill_admission_last_bucket': str(self._online_prefill_admission_last_bucket),
            'online_prefill_admission_last_cuda_free_gb': round(float(self._online_prefill_admission_last_cuda_free_gb), 4),
            'online_prefill_admission_last_active_short': int(self._online_prefill_admission_last_active_short),
            'online_prefill_admission_last_active_mid': int(self._online_prefill_admission_last_active_mid),
            'online_prefill_admission_last_active_long': int(self._online_prefill_admission_last_active_long),
            'online_prefill_admission_last_cap': int(self._online_prefill_admission_last_cap),
            'online_prefill_active_token_budget': int(self.online_prefill_active_token_budget),
            'online_prefill_admission_last_active_tokens': int(self._online_prefill_admission_last_active_tokens),
            'online_prefill_admission_last_projected_tokens': int(self._online_prefill_admission_last_projected_tokens),
            'online_prefill_admission_last_token_budget': int(self._online_prefill_admission_last_token_budget),
            'online_prefill_admission_token_budget_blocked_steps': int(self._online_prefill_admission_token_budget_blocked_steps),
            'online_prefill_admission_token_budget_blocked_requests': int(self._online_prefill_admission_token_budget_blocked_requests),
            'online_prefill_admission_last_allowed': int(self._online_prefill_admission_last_allowed),
            'online_prefill_chunk_floor_pause_steps': int(self._online_prefill_chunk_floor_pause_steps),
            'online_prefill_chunk_floor_last_chunk_cap': int(self._online_prefill_chunk_floor_last_chunk_cap),
            'p2_gain_success_steps': int(decode_window_status.get('p2_gain_success_steps', 0)),
            'p2_gain_fail_steps': int(decode_window_status.get('p2_gain_fail_steps', 0)),
            'p2_skipped_low_benefit_steps': int(decode_window_status.get('p2_skipped_low_benefit_steps', 0)),
            'first_p2_step': int(decode_window_status.get('first_p2_step', -1)),
        }
        metrics['wm_low'] = int(decode_window_status.get('wm_low', 0))
        metrics['wm_high'] = int(decode_window_status.get('wm_high', 0))
        metrics['n_free_final'] = int(decode_window_status.get('n_free', 0))
        metrics['p2_low_threshold'] = int(decode_window_status.get('p2_low_threshold', 0))
        if torch.cuda.is_available():
            metrics['peak_cuda_mem_gb'] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
            metrics['peak_gpu_mem_allocated_gb'] = metrics['peak_cuda_mem_gb']

        details = None
        if return_details:
            details = {
                'token_ids': [finished_map.get(rid, {}).get('token_ids', []) for rid in rid_order],
                'token_logprobs': [finished_map.get(rid, {}).get('token_logprobs', []) for rid in rid_order],
            }

        if return_metrics and return_details:
            return outputs, metrics, details
        if return_metrics:
            return outputs, metrics
        return outputs, details

    def generate(
        self,
        prompts: List[str],
        return_metrics: bool = False,
        return_details: bool = False,
        step_callback: Optional[Any] = None,
    ):
        return self._generate_with_online(
            prompts=prompts,
            return_metrics=return_metrics,
            return_details=return_details,
            step_callback=step_callback,
        )

    def _ensure_decode_blocks_on_gpu(self, seq_ids: List[int]) -> None:
        sched = self.scheduler
        largest_missing = 0
        missing_by_seq: Dict[int, List[int]] = {}
        total_missing = 0
        total_materialized = 0
        for sid in seq_ids:
            entry = sched.offloader._get_seq_layer0_entry(int(sid))
            if entry is None:
                continue
            sched.offloader._ensure_entry_maps(entry)
            total_materialized += int(getattr(entry, 'materialized_blocks', 0) or len(entry.gpu_block_map))
            missing = [idx for idx, bid in enumerate(entry.gpu_block_map) if int(bid) < 0]
            if missing:
                missing_by_seq[int(sid)] = missing
                largest_missing = max(largest_missing, len(missing))
                total_missing += len(missing)
        self._online_last_missing_blocks_scheduled = int(total_missing)
        self._online_last_materialized_blocks_scheduled = int(total_materialized)
        prefetch_budget = max(
            int(sched.offloader.prefetch_budget_blocks_base),
            int(largest_missing),
        )
        sched.offloader.set_step_transfer_budgets(prefetch_budget=prefetch_budget)
        for sid, missing in missing_by_seq.items():
            ok = sched.offloader.ensure_sequence_blocks_on_gpu(
                sid,
                missing,
                allow_evict=True,
                protected_seqs=seq_ids,
            )
            if not ok:
                raise MemoryError(f'Failed to prefetch sequence {sid} before decode')

    def _prepare_paged_direct_context(self, seq_ids: List[int]) -> Tuple[Dict[str, Any], List[int], List[int]]:
        ctx = self.scheduler.prepare_paged_direct_context(seq_ids)
        self._online_last_missing_blocks_scheduled = int(ctx.get('resident_missing_blocks', 0) or 0)
        self._online_last_materialized_blocks_scheduled = int(ctx.get('materialized_blocks', 0) or 0)
        return (
            ctx,
            list(ctx.get('seq_lens', [0] * len(seq_ids))),
            list(ctx.get('logical_seq_lens', [0] * len(seq_ids))),
        )

    def _prepare_page16_native_context(self, seq_ids: List[int]) -> Tuple[Dict[str, Any], List[int], List[int]]:
        self._ensure_decode_blocks_on_gpu(seq_ids)
        ctx = self.scheduler.prepare_page16_native_context(seq_ids)
        self._online_last_missing_blocks_scheduled = int(ctx.get('resident_missing_blocks', 0) or 0)
        self._online_last_materialized_blocks_scheduled = int(ctx.get('materialized_blocks', 0) or 0)
        return (
            ctx,
            list(ctx.get('seq_lens', [0] * len(seq_ids))),
            list(ctx.get('logical_seq_lens', [0] * len(seq_ids))),
        )

    def _rebuild_pkv_paged_direct(self, seq_ids: List[int], return_seq_lens: bool = False):
        """
        Paged materialize fast path in Python:
        - Reads per-sequence KV pages directly from pool blocks.
        - Still materializes a DynamicCache, so it is not true direct decode.
        """
        from transformers.cache_utils import DynamicCache

        sched = self.scheduler
        if not seq_ids:
            raise RuntimeError("empty decode batch")

        self._ensure_decode_blocks_on_gpu(seq_ids)

        cache = DynamicCache()
        seq_lens_ref: Optional[List[int]] = None
        logical_seq_lens_ref: Optional[List[int]] = None

        for layer_id in range(sched.num_layers):
            entries = []
            for sid in seq_ids:
                entry = sched.offloader.page_table.get((sid, layer_id))
                if entry is None:
                    raise MemoryError(f"Missing KV entry: seq={sid} layer={layer_id}")
                if entry.state != OffloadState.ON_GPU or not entry.block_ids:
                    raise MemoryError(
                        f"Invalid KV state for seq={sid} layer={layer_id}: "
                        f"state={entry.state}, blocks={len(entry.block_ids)}"
                    )
                entries.append(entry)

            seq_lens = [int(e.seq_len) for e in entries]
            logical_seq_lens = [int(getattr(e, "logical_seq_len", 0) or e.seq_len) for e in entries]
            if seq_lens_ref is None:
                seq_lens_ref = list(seq_lens)
                logical_seq_lens_ref = list(logical_seq_lens)
            elif seq_lens != seq_lens_ref:
                raise RuntimeError("Inconsistent per-layer sequence lengths while rebuilding KV cache")
            elif logical_seq_lens != logical_seq_lens_ref:
                raise RuntimeError("Inconsistent logical sequence lengths while rebuilding KV cache")

            K_list: List[torch.Tensor] = []
            V_list: List[torch.Tensor] = []
            for e in entries:
                K_i, V_i = sched.pool.read_kv_from_blocks(layer_id, e.block_ids, int(e.seq_len))
                if K_i.dim() != 3 or V_i.dim() != 3:
                    raise RuntimeError("Unexpected KV rank in paged_direct path")
                K_list.append(K_i.unsqueeze(0))
                V_list.append(V_i.unsqueeze(0))

            max_len = max(int(x.shape[2]) for x in K_list) if K_list else 0
            if max_len <= 0:
                continue
            K_pad = [torch.nn.functional.pad(k, (0, 0, 0, max_len - int(k.shape[2]))) for k in K_list]
            V_pad = [torch.nn.functional.pad(v, (0, 0, 0, max_len - int(v.shape[2]))) for v in V_list]
            K_layer = torch.cat(K_pad, dim=0)
            V_layer = torch.cat(V_pad, dim=0)
            self._record_decode_materialize_kv_bytes(K_layer, V_layer)
            cache.update(K_layer, V_layer, layer_id)

        if return_seq_lens:
            return cache, (seq_lens_ref if seq_lens_ref is not None else [0] * len(seq_ids)), (logical_seq_lens_ref if logical_seq_lens_ref is not None else [0] * len(seq_ids))
        return cache

    def _rebuild_pkv(self, seq_ids: List[int], return_seq_lens: bool = False):
        """Rebuild DynamicCache from paged KV blocks via block table gather."""
        from transformers.cache_utils import DynamicCache

        sched = self.scheduler

        self._ensure_decode_blocks_on_gpu(seq_ids)

        pkv_list = []
        seq_lens_ref = None
        logical_seq_lens_ref = None
        for layer_id in range(sched.num_layers):
            entries = []
            for sid in seq_ids:
                entry = sched.offloader.page_table.get((sid, layer_id))
                if entry is None:
                    raise MemoryError(f"Missing KV entry: seq={sid} layer={layer_id}")
                if entry.state != OffloadState.ON_GPU or not entry.block_ids:
                    raise MemoryError(
                        f"Invalid KV state for seq={sid} layer={layer_id}: "
                        f"state={entry.state}, blocks={len(entry.block_ids)}"
                    )
                entries.append(entry)

            if not entries:
                pkv_list.append(None)
                continue

            seq_block_ids_list = [e.block_ids for e in entries]
            seq_lens = [int(e.seq_len) for e in entries]
            logical_seq_lens = [int(getattr(e, "logical_seq_len", 0) or e.seq_len) for e in entries]
            if seq_lens_ref is None:
                seq_lens_ref = list(seq_lens)
                logical_seq_lens_ref = list(logical_seq_lens)
            elif seq_lens != seq_lens_ref:
                raise RuntimeError("Inconsistent per-layer sequence lengths while rebuilding KV cache")
            elif logical_seq_lens != logical_seq_lens_ref:
                raise RuntimeError("Inconsistent logical sequence lengths while rebuilding KV cache")

            k_cache_layer, v_cache_layer, block_table = sched.pool.build_block_table(layer_id, seq_block_ids_list)

            batch, max_blocks = block_table.shape
            flat_block_ids = block_table.to(torch.int64).reshape(-1)

            k_pages = k_cache_layer.index_select(0, flat_block_ids).view(
                batch,
                max_blocks,
                sched.pool.B,
                sched.num_kv_heads,
                sched.head_dim,
            )
            v_pages = v_cache_layer.index_select(0, flat_block_ids).view(
                batch,
                max_blocks,
                sched.pool.B,
                sched.num_kv_heads,
                sched.head_dim,
            )

            k_full = k_pages.permute(0, 3, 1, 2, 4).reshape(
                batch,
                sched.num_kv_heads,
                max_blocks * sched.pool.B,
                sched.head_dim,
            )
            v_full = v_pages.permute(0, 3, 1, 2, 4).reshape(
                batch,
                sched.num_kv_heads,
                max_blocks * sched.pool.B,
                sched.head_dim,
            )

            Ks, Vs = [], []
            for i, l in enumerate(seq_lens):
                Ks.append(k_full[i: i + 1, :, :l, :])
                Vs.append(v_full[i: i + 1, :, :l, :])

            max_len = max(k.shape[2] for k in Ks)
            Ks = [torch.nn.functional.pad(k, (0, 0, 0, max_len - k.shape[2])) for k in Ks]
            Vs = [torch.nn.functional.pad(v, (0, 0, 0, max_len - v.shape[2])) for v in Vs]
            K_layer = torch.cat(Ks, dim=0)
            V_layer = torch.cat(Vs, dim=0)
            self._record_decode_materialize_kv_bytes(K_layer, V_layer)
            pkv_list.append((K_layer, V_layer))

        cache = DynamicCache()
        for layer_id, kv in enumerate(pkv_list):
            if kv is None:
                continue
            K_layer, V_layer = kv
            cache.update(K_layer, V_layer, layer_id)

        if return_seq_lens:
            return cache, (seq_lens_ref if seq_lens_ref is not None else [0] * len(seq_ids)), (logical_seq_lens_ref if logical_seq_lens_ref is not None else [0] * len(seq_ids))
        return cache
