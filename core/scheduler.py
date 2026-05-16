import math
import os
from collections import deque
import time
import types
import torch
from typing import Any, Deque, Dict, List, Optional, Tuple
from kv_types import BlockTableEntry, OffloadState, PhaseState
from pool import PagedKVPool
from compress import BlockAlignedSnapKV, repeat_kv
from offload import AsyncOffloadManager

try:
    from page16_native import HAS_TRITON as HAS_PAGE16_NATIVE, page16_decode_attention
except Exception:
    HAS_PAGE16_NATIVE = False
    page16_decode_attention = None

try:
    from flash_attn import flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rotary_pos_emb
except Exception:
    llama_apply_rotary_pos_emb = None


class KVScheduler:
    def __init__(
        self,
        model,
        block_size: int = 16,
        sink_len: int = 16,
        snapkv_observation_len: int = 16,
        retain_ratio: float = 0.20,
        retain_budget_tokens: int = 0,
        gpu_mem_frac: float = 0.75,
        cpu_mem_gb: float = 48.0,
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
        p2_cuda_pressure_min_gb: float = 1.0,
        kv_min_resident_ratio: float = 0.20,
        selected_writeback_enabled: bool = False,
    ):
        self.model = model
        self.phase = PhaseState.IDLE
        cfg = model.config

        self.num_layers = cfg.num_hidden_layers
        self.num_q_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, 'num_key_value_heads', self.num_q_heads)
        self.head_dim = cfg.hidden_size // self.num_q_heads
        self.n_rep = self.num_q_heads // self.num_kv_heads

        self.pool = PagedKVPool(
            block_size,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            torch.float16,
            gpu_mem_frac,
            cpu_mem_gb,
        )
        # Lock hysteresis watermarks to avoid frequent re-trigger. Ratios are
        # overrideable for calibration; defaults preserve the historical policy.
        wm_low_ratio = float(max(0.0, min(1.0, wm_low_ratio)))
        wm_high_ratio = float(max(wm_low_ratio, min(1.0, wm_high_ratio)))
        self.wm_low_ratio = wm_low_ratio
        self.wm_high_ratio = wm_high_ratio
        self.pool.N_wm_low = max(1, int(self.pool.N_total * wm_low_ratio))
        self.pool.N_wm_high = max(self.pool.N_wm_low + 1, int(self.pool.N_total * wm_high_ratio))
        self.snapkv = BlockAlignedSnapKV(block_size, sink_len, snapkv_observation_len, retain_ratio, retain_budget_tokens)
        self.snapkv_anchor_tokens = max(0, int(min(block_size, snapkv_observation_len)))
        self.offloader = AsyncOffloadManager(self.pool, self.num_layers)
        self.offloader.min_resident_steps = 16
        self.offloader.offload_budget_blocks = max(1, int(offload_budget_blocks))
        self.offloader.prefetch_budget_blocks = max(1, int(prefetch_budget_blocks))
        self.p2_enabled = bool(p2_enabled)
        self.p2_min_reclaim_blocks = max(1, int(p2_min_reclaim_blocks))
        self.p2_gain_window_steps = max(1, int(p2_gain_window_steps))
        self.p2_gain_fail_cooldown_steps = max(0, int(p2_gain_fail_cooldown_steps))
        # Runtime is now P2-only. Keep only the pressure path and P2 protection tokens.
        self.p2_sink_tokens = max(0, int(p2_sink_tokens))
        self.p2_recent_tokens = max(0, int(p2_recent_tokens))
        self.decode_window_tiered = bool(decode_window_tiered)
        self.decode_window_thrash_window_steps = max(1, int(decode_window_thrash_window_steps))
        self.decode_window_thrash_low = max(0.0, float(decode_window_thrash_low))
        self.decode_window_thrash_high = max(
            self.decode_window_thrash_low, float(decode_window_thrash_high)
        )
        self.decode_window_pressure_steps = max(1, int(decode_window_pressure_steps))
        self.decode_window_recover_steps = max(1, int(decode_window_recover_steps))
        self.ready_decode_eviction_threshold = max(1, int(ready_decode_eviction_threshold))
        self.p2_target_free_blocks_cfg = max(0, int(p2_target_free_blocks))
        self.p2_cuda_pressure_min_gb = float(max(0.0, p2_cuda_pressure_min_gb))
        self.kv_min_resident_ratio = float(max(0.0, min(1.0, kv_min_resident_ratio)))
        env_selected_writeback = str(os.environ.get("KV_MIDDLEWARE_SELECTED_WRITEBACK", "")).strip().lower() in {"1", "true", "yes", "on"}
        env_disable_selected_writeback = str(os.environ.get("KV_MIDDLEWARE_DISABLE_SELECTED_WRITEBACK", "")).strip().lower() in {"1", "true", "yes", "on"}
        self.selected_writeback_enabled = (bool(selected_writeback_enabled) or bool(env_selected_writeback)) and not bool(env_disable_selected_writeback)
        self.selected_writeback_safety_margin_gb = float(os.environ.get("KV_MIDDLEWARE_SELECTED_WRITEBACK_MARGIN_GB", "0.25") or 0.25)
        self._decode_window_step_thrash: Deque[float] = deque(
            maxlen=self.decode_window_thrash_window_steps
        )
        base_stats = self.offloader.get_stats(reset=False)
        self._decode_window_last_io_total = int(base_stats.get('offload_success', 0)) + int(
            base_stats.get('prefetch_success', 0)
        )
        self._p2_attempts = 0
        self._p2_successes = 0
        self._p2_fail_streak = 0
        self._p2_last_attempted = False
        self._p2_last_success = False
        self._p2_last_no_candidate = False
        self._p2_last_candidate_count = 0
        self._p2_managed_active = False
        self._p2_recover_streak = 0
        self._p2_active_steps = 0
        self._p2_candidate_steps = 0
        self._p2_recovery_fail_windows = 0
        self._p2_no_candidate_steps = 0
        self._p2_no_candidate_pressure_streak = 0
        self._p2_ready_candidate_steps = 0
        self._p2_decode_candidate_steps = 0
        self._p2_expected_reclaim_blocks = 0
        self._p2_ready_offload_blocks_total = 0
        self._p2_ready_offload_blocks_last = 0
        self._p2_ready_offload_sequence_steps = 0
        self._p2_ready_offload_decode_steps = 0
        self._p2_ready_offload_last_step = -1
        self._p2_ready_sequences_selected_per_step = 0
        self._p2_ready_offload_blocks_per_step = 0
        self._p2_ready_target_reclaim_blocks = 0
        self._p2_ready_actual_reclaim_blocks = 0
        self._p2_ready_stop_reason = "none"
        self._p2_ready_stop_reason_counts: Dict[str, int] = {}
        self._p2_gain_success_steps = 0
        self._p2_gain_fail_steps = 0
        self._p2_skipped_low_benefit_steps = 0
        self._p2_gain_fail_cooldown_until = 0
        self._p2_gain_recent: Deque[int] = deque(maxlen=self.p2_gain_window_steps)
        self._first_p2_step = -1
        self._cuda_free_post_cleanup_gb = float("inf")
        self._decode_cuda_free_post_cleanup_recent: Deque[float] = deque(
            maxlen=self.decode_window_pressure_steps
        )
        self._p2_cuda_pressure_signal_steps = 0
        self._p2_cuda_pressure_accounted_step = -1
        self._p2_activity_accounted_step = -1
        self._last_pressure_signal = False
        self.prefill_writeback_backend = "legacy_cpu_full"
        self.gpu_selected_writeback_steps = 0
        self.cpu_selected_compaction_steps = 0
        self.gpu_writeback_oom_fallbacks = 0
        self.writeback_transaction_rollbacks = 0
        self.raw_kv_cpu_stash_bytes = 0
        self.selected_global_block_count = 0
        self.writeback_est_required_gb = 0.0
        self.writeback_free_gb = 0.0
        self.writeback_block_selection_shared_layers = 0
        self.score_full_attention_materialized = 0

        self.raw_kv_cache: Dict[int, Dict[int, Tuple]] = {i: {} for i in range(self.num_layers)}
        self.q_cache: Dict[int, Dict[int, torch.Tensor]] = {i: {} for i in range(self.num_layers)}

        self.active_seqs: List[int] = []
        self.decode_step_count: int = 0
        self._hooks = []
        self._patched_modules = []
        self._paged_direct_context: Optional[Dict[str, Any]] = None
        self._paged_direct_active = False
        self._page16_native_kernel_ms_accum = 0.0
        self._page16_native_kernel_calls = 0

        self._install_hooks()

    def paged_flash_decode_support(self) -> Tuple[bool, str]:
        if not HAS_FLASH_ATTN:
            return False, "flash_attn_not_available"
        # flash_attn_with_kvcache paged-KV path requires page_block_size % 256 == 0.
        if int(self.pool.B) % 256 != 0:
            return False, f"block_size_{self.pool.B}_not_multiple_of_256"
        return True, "ok"

    def page16_native_decode_support(self) -> Tuple[bool, str]:
        if not HAS_PAGE16_NATIVE or page16_decode_attention is None:
            return False, "page16_native_unavailable"
        if int(self.pool.B) != 16:
            return False, f"block_size_{self.pool.B}_not_16"
        if int(self.head_dim) > 256:
            return False, f"head_dim_{self.head_dim}_unsupported"
        if int(self.num_q_heads) % int(self.num_kv_heads) != 0:
            return False, "gqa_ratio_unsupported"
        return True, "ok"

    def _get_layers(self):
        m = self.model
        if hasattr(m, 'model') and hasattr(m.model, 'layers'):
            return m.model.layers
        if hasattr(m, 'transformer') and hasattr(m.transformer, 'h'):
            return m.transformer.h
        raise ValueError(f"Unsupported model layer structure: {type(m)}")

    def _get_attn(self, layer):
        for name in ['self_attn', 'attn', 'attention', 'self_attention']:
            if hasattr(layer, name):
                return getattr(layer, name)
        return None

    @staticmethod
    def _stash_prefill_kv_tensor(x: torch.Tensor, keep_on_gpu: bool = False) -> torch.Tensor:
        # Legacy path stashes raw prefill KV on CPU. Selected-writeback keeps
        # it on GPU so Level 1 can slice selected blocks without round-tripping.
        t = x.detach()
        if t.device.type == 'cuda' and not keep_on_gpu:
            t = t.to('cpu')
        return t.contiguous()

    @staticmethod
    def _normalize_kv_pair(kv_pair, batch_idx: int = 0, seq_len: int = None, keep_on_gpu: bool = False):
        if not isinstance(kv_pair, (tuple, list)) or len(kv_pair) < 2:
            return None
        K, V = kv_pair[0], kv_pair[1]
        if not (torch.is_tensor(K) and torch.is_tensor(V)):
            return None
        if K.numel() == 0 or V.numel() == 0:
            return None

        if K.dim() == 4 and V.dim() == 4:
            if K.shape[0] == 0 or V.shape[0] == 0:
                return None
            if batch_idx >= K.shape[0] or batch_idx >= V.shape[0]:
                return None
            K_sel = K[batch_idx].detach()
            V_sel = V[batch_idx].detach()
            if seq_len is not None and seq_len > 0:
                K_sel = K_sel[:, :seq_len, :]
                V_sel = V_sel[:, :seq_len, :]
            return KVScheduler._stash_prefill_kv_tensor(K_sel, keep_on_gpu=keep_on_gpu), KVScheduler._stash_prefill_kv_tensor(V_sel, keep_on_gpu=keep_on_gpu)

        if K.dim() == 3 and V.dim() == 3:
            K_sel = K.detach()
            V_sel = V.detach()
            if seq_len is not None and seq_len > 0:
                K_sel = K_sel[:, :seq_len, :]
                V_sel = V_sel[:, :seq_len, :]
            return KVScheduler._stash_prefill_kv_tensor(K_sel, keep_on_gpu=keep_on_gpu), KVScheduler._stash_prefill_kv_tensor(V_sel, keep_on_gpu=keep_on_gpu)

        return None

    def _install_hooks(self):
        for lid, layer in enumerate(self._get_layers()):
            attn = self._get_attn(layer)
            if attn is None or not hasattr(attn, 'q_proj'):
                continue
            self._patch_attn(attn, lid)

    def _patch_attn(self, attn_module, layer_id: int):
        scheduler = self
        original_q_proj = attn_module.q_proj
        original_forward = attn_module.forward

        class CachingQProj(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.original = original_q_proj

            def forward(self, x):
                Q_out = self.original(x)
                if scheduler.phase in (PhaseState.PREFILL, PhaseState.COMPRESS):
                    b, s, _ = Q_out.shape
                    Q = Q_out.view(b, s, scheduler.num_q_heads, scheduler.head_dim).transpose(1, 2)
                    for i, sid in enumerate(scheduler.active_seqs):
                        if i < b:
                            q_tail = Q[i].detach()
                            prev = scheduler.q_cache[layer_id].get(sid)
                            if prev is not None:
                                q_tail = torch.cat([prev, q_tail], dim=1)
                            if q_tail.shape[1] > scheduler.snapkv.Lo:
                                q_tail = q_tail[:, -scheduler.snapkv.Lo :, :]
                            scheduler.q_cache[layer_id][sid] = q_tail
                return Q_out

        attn_module.q_proj = CachingQProj()

        def direct_forward(
            module,
            hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            if not (scheduler.phase == PhaseState.DECODE and scheduler._paged_direct_active):
                return original_forward(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )

            ctx = scheduler._paged_direct_context or {}
            layer_views = ctx.get('layer_views') or {}
            view = layer_views.get(layer_id)
            if view is None:
                raise RuntimeError(f"paged_direct_missing_layer_view layer={layer_id}")
            if llama_apply_rotary_pos_emb is None:
                raise RuntimeError("direct_decode_llama_rope_not_available")
            if position_embeddings is None or len(position_embeddings) < 2:
                raise RuntimeError("direct_decode_missing_position_embeddings")

            bsz, q_len, _ = hidden_states.shape
            if q_len != 1:
                raise RuntimeError(f"direct_decode_requires_q_len_1 got={q_len}")

            q = module.q_proj(hidden_states)
            k = module.k_proj(hidden_states)
            v = module.v_proj(hidden_states)
            q = q.view(bsz, q_len, scheduler.num_q_heads, scheduler.head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, scheduler.num_kv_heads, scheduler.head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, scheduler.num_kv_heads, scheduler.head_dim).transpose(1, 2)

            cos, sin = position_embeddings
            q, k = llama_apply_rotary_pos_emb(q, k, cos, sin)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

            softmax_scale = getattr(module, 'scaling', None)
            backend = str(ctx.get('backend', 'paged_direct_flash'))
            if backend == 'page16_native':
                if not HAS_PAGE16_NATIVE or page16_decode_attention is None:
                    raise RuntimeError("page16_native_unavailable")
                old_lens = ctx['cache_seqlens']
                batch_idx = torch.arange(bsz, device=hidden_states.device, dtype=torch.long)
                block_pos = torch.div(old_lens.to(torch.long), scheduler.pool.B, rounding_mode='floor')
                tok_pos = torch.remainder(old_lens.to(torch.long), scheduler.pool.B)
                block_ids = view['block_table'][batch_idx, block_pos]
                view['k_cache'][block_ids.to(torch.long), tok_pos, :, :] = k[:, 0, :, :]
                view['v_cache'][block_ids.to(torch.long), tok_pos, :, :] = v[:, 0, :, :]
                attn_lens = (old_lens + 1).to(dtype=torch.int32)
                t0 = time.perf_counter()
                out = page16_decode_attention(
                    q.squeeze(1),
                    view['k_cache'],
                    view['v_cache'],
                    view['block_table'],
                    attn_lens,
                    softmax_scale=float(softmax_scale if softmax_scale is not None else (scheduler.head_dim ** -0.5)),
                    n_rep=int(scheduler.n_rep),
                ).unsqueeze(1)
                scheduler._page16_native_kernel_ms_accum += (time.perf_counter() - t0) * 1000.0
                scheduler._page16_native_kernel_calls += 1
            else:
                if not HAS_FLASH_ATTN or flash_attn_with_kvcache is None:
                    raise RuntimeError("paged_direct_flash_attn_not_available")
                out = flash_attn_with_kvcache(
                    q=q,
                    k_cache=view['k_cache'],
                    v_cache=view['v_cache'],
                    k=k,
                    v=v,
                    cache_seqlens=ctx['cache_seqlens'],
                    block_table=view['block_table'],
                    softmax_scale=softmax_scale,
                    causal=True,
                )
            out = out.reshape(bsz, q_len, scheduler.num_q_heads * scheduler.head_dim).contiguous()
            out = module.o_proj(out)
            return out, None

        attn_module.forward = types.MethodType(direct_forward, attn_module)
        self._patched_modules.append((attn_module, original_q_proj, original_forward))

        # Legacy compatibility: some implementations expose (K,V) in attention output.
        def post_hook(module, args, output):
            if scheduler.phase != PhaseState.PREFILL:
                return
            if not scheduler.active_seqs:
                return
            if not isinstance(output, tuple):
                return

            for item in output:
                if not isinstance(item, (tuple, list)) or len(item) < 2:
                    continue
                K, V = item[0], item[1]
                if not (torch.is_tensor(K) and torch.is_tensor(V)):
                    continue

                if K.dim() == 4 and V.dim() == 4:
                    for i, sid in enumerate(scheduler.active_seqs):
                        kv_norm = scheduler._normalize_kv_pair(item, batch_idx=i, keep_on_gpu=scheduler.selected_writeback_enabled)
                        if kv_norm is not None:
                            scheduler.raw_kv_cache[layer_id][sid] = kv_norm
                    return

                kv_norm = scheduler._normalize_kv_pair(item, batch_idx=0, keep_on_gpu=scheduler.selected_writeback_enabled)
                if kv_norm is not None:
                    sid = scheduler.active_seqs[0]
                    scheduler.raw_kv_cache[layer_id][sid] = kv_norm
                    return

        h = attn_module.register_forward_hook(post_hook)
        self._hooks.append(h)

    def restore_hooks(self):
        for item in self._patched_modules:
            if len(item) == 3:
                attn_module, orig_q_proj, orig_forward = item
                attn_module.q_proj = orig_q_proj
                attn_module.forward = orig_forward
            else:
                attn_module, orig_q_proj = item
                attn_module.q_proj = orig_q_proj
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._patched_modules.clear()

    def capture_prefill_kv(
        self,
        seq_id: int,
        past_key_values: Any,
        batch_idx: int = 0,
        seq_len: int = None,
    ):
        """Capture per-layer KV from model output cache (new Transformers cache API)."""
        if past_key_values is None:
            return

        for layer_id in range(self.num_layers):
            try:
                kv_pair = past_key_values[layer_id]
            except Exception:
                continue

            kv_norm = self._normalize_kv_pair(kv_pair, batch_idx=batch_idx, seq_len=seq_len, keep_on_gpu=self.selected_writeback_enabled)
            if kv_norm is None:
                continue

            self.raw_kv_cache[layer_id][seq_id] = kv_norm


    def _selected_writeback_cuda_free_gb(self) -> float:
        if not torch.cuda.is_available():
            return float("inf")
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            return float(free_bytes) / 1024**3
        except Exception:
            return 0.0

    @staticmethod
    def _tensor_nbytes(x: Optional[torch.Tensor]) -> int:
        if not torch.is_tensor(x):
            return 0
        return int(x.numel() * x.element_size())

    def _selection_budget_blocks(self, L: int) -> Tuple[int, int, str]:
        B = int(self.snapkv.B)
        budget = int(getattr(self.snapkv, "retain_budget_tokens", 0) or 0)
        if budget > 0 and int(L) <= budget:
            return 0, int(L), "input_le_budget"
        if int(L) < int(self.snapkv.Ls) + int(self.snapkv.Lo) + B:
            return 0, int(L), "too_short"
        mid_tokens = max(0, int(L) - int(self.snapkv.Ls) - int(self.snapkv.Lo))
        mid_blocks = int(math.ceil(mid_tokens / B))
        if budget > 0:
            mid_budget = max(0, budget - int(self.snapkv.Ls) - int(self.snapkv.Lo))
            k_blocks = max(0, min(mid_blocks, int(math.floor(mid_budget / B))))
            effective = int(self.snapkv.Ls) + k_blocks * B + int(self.snapkv.Lo)
            return int(k_blocks), int(effective), "fixed_budget"
        k_blocks = max(1, min(mid_blocks, int(math.floor(float(self.snapkv.retain_ratio) * mid_blocks))))
        effective = int(self.snapkv.Ls) + k_blocks * B + int(self.snapkv.Lo)
        return int(k_blocks), int(effective), "ratio"

    def _score_layer_blocks_for_selection(self, K: torch.Tensor, Q_obs: torch.Tensor) -> torch.Tensor:
        if K.dim() != 3 or Q_obs.dim() != 3:
            raise RuntimeError("selected_writeback_bad_score_shape")
        device = torch.device("cuda")
        K_gpu = K if K.device.type == "cuda" else K.to(device)
        Q_gpu = Q_obs if Q_obs.device.type == "cuda" else Q_obs.to(device)
        L = int(K_gpu.shape[1])
        B = int(self.snapkv.B)
        if L < int(self.snapkv.Ls) + int(self.snapkv.Lo) + B:
            return torch.empty((0,), dtype=torch.float32, device=device)
        K_mid = K_gpu[:, int(self.snapkv.Ls):-int(self.snapkv.Lo), :]
        L_mid = int(K_mid.shape[1])
        aligned = int(math.ceil(max(1, L_mid) / B) * B)
        if aligned > L_mid:
            K_mid = torch.nn.functional.pad(K_mid, (0, 0, 0, aligned - L_mid))
        N = int(aligned // B)
        if N <= 0:
            return torch.empty((0,), dtype=torch.float32, device=device)
        K_mid_exp = repeat_kv(K_mid.unsqueeze(0), int(self.n_rep))
        Qb = Q_gpu.unsqueeze(0)
        # Observation-window score only: [heads, Lo, L_mid], never [L, L].
        self.score_full_attention_materialized = 0
        attn = torch.softmax(torch.matmul(Qb, K_mid_exp.transpose(-2, -1)) * (float(self.head_dim) ** -0.5), dim=-1)
        score = attn.mean(dim=2).max(dim=1).values
        score_block = score.view(1, N, B).sum(dim=-1).squeeze(0).float()
        if score_block.numel() >= 3:
            score_block = torch.nn.functional.max_pool1d(score_block.view(1, 1, -1), kernel_size=3, stride=1, padding=1).view(-1)
        return score_block

    def _compute_per_layer_selected_blocks(self, seq_id: int, raw_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[Dict[int, List[int]], int, int, str]:
        first = next((kv for kv in raw_by_layer.values() if kv is not None), None)
        if first is None:
            raise RuntimeError(f"selected_writeback_missing_raw seq={seq_id}")
        L = int(first[0].shape[1])
        k_blocks, effective_len, mode = self._selection_budget_blocks(L)
        selected_by_layer: Dict[int, List[int]] = {}
        if k_blocks <= 0 or mode in ("input_le_budget", "too_short"):
            selected_by_layer = {int(layer_id): [] for layer_id in range(self.num_layers)}
        else:
            for layer_id in range(self.num_layers):
                raw = raw_by_layer.get(layer_id)
                if raw is None:
                    raise RuntimeError(f"selected_writeback_missing_raw seq={seq_id} layer={layer_id}")
                Q_full = self.q_cache[layer_id].get(seq_id)
                if Q_full is None:
                    raise RuntimeError(f"selected_writeback_missing_q seq={seq_id} layer={layer_id}")
                Q_obs = Q_full[:, -int(self.snapkv.Lo):, :]
                score = self._score_layer_blocks_for_selection(raw[0], Q_obs)
                k = max(0, min(int(k_blocks), int(score.numel())))
                if k > 0:
                    idx = torch.topk(score, k=k, dim=0).indices
                    selected_by_layer[int(layer_id)] = sorted({int(x) for x in idx.detach().cpu().tolist()})
                else:
                    selected_by_layer[int(layer_id)] = []
        total_blocks = int(math.ceil(max(1, int(effective_len)) / int(self.pool.B)))
        self.selected_global_block_count = int(total_blocks)
        self.writeback_block_selection_shared_layers = 0
        retained_idx_debug = [selected_by_layer.get(int(layer_id), []) for layer_id in range(self.num_layers)]
        self.snapkv.last_debug = {
            "block_size": int(self.snapkv.B),
            "sink_len": int(self.snapkv.Ls),
            "obs_len": int(self.snapkv.Lo),
            "retain_ratio": float(self.snapkv.retain_ratio),
            "retain_budget_tokens": int(getattr(self.snapkv, "retain_budget_tokens", 0) or 0),
            "compression_mode": "fixed_budget" if int(getattr(self.snapkv, "retain_budget_tokens", 0) or 0) > 0 else "ratio",
            "logical_seq_len": int(L),
            "retained_block_count": int(k_blocks),
            "selected_global_block_count": int(total_blocks),
            "effective_retained_tokens": int(effective_len),
            "retained_block_idx": retained_idx_debug,
            "shared_block_selection": False,
            "selection_granularity": "per_layer_block_topk",
            "compression_skipped": mode if mode in ("input_le_budget", "too_short") else "",
        }
        return selected_by_layer, int(effective_len), int(L), str(mode)

    def _evict_one_for_prefill_allocation(self, seq_id: int, target_free: int) -> None:
        before = int(self.pool.n_free)
        victim = self.offloader.evict_coldest_sequence(int(seq_id))
        try:
            self.offloader._try_finalize_offload(int(victim), force_sync=True)
        except Exception:
            pass
        after = int(self.pool.n_free)
        if after <= before and after < int(target_free):
            raise MemoryError(
                f"Eviction made no progress for seq={seq_id}: n_free={after}, target_free={int(target_free)}"
            )

    def _allocate_writeback_blocks(self, seq_id: int, n_req: int) -> List[int]:
        if self.pool.n_free < self.pool.N_wm_low:
            target_free = max(int(n_req), int(self.pool.N_wm_high))
            while self.pool.n_free < target_free:
                self._evict_one_for_prefill_allocation(seq_id, target_free)
        while self.pool.n_free < int(n_req):
            self._evict_one_for_prefill_allocation(seq_id, int(n_req))
        block_ids = self.pool.allocate_blocks(int(n_req))
        if not block_ids:
            raise MemoryError(f"Cannot allocate {int(n_req)} blocks for seq={seq_id}")
        return list(block_ids)

    def _write_piece_to_blocks(self, layer_id: int, block_ids: List[int], dst_offset: int, K_piece: torch.Tensor, V_piece: torch.Tensor) -> int:
        if K_piece.shape[1] != V_piece.shape[1]:
            raise RuntimeError("selected_writeback_piece_len_mismatch")
        B = int(self.pool.B)
        total = int(K_piece.shape[1])
        src = 0
        pos = int(dst_offset)
        while src < total:
            block_idx = pos // B
            off = pos % B
            if block_idx >= len(block_ids):
                raise RuntimeError("selected_writeback_block_overflow")
            take = min(total - src, B - off)
            bid = int(block_ids[block_idx])
            k_chunk = K_piece[:, src:src + take, :]
            v_chunk = V_piece[:, src:src + take, :]
            if k_chunk.device.type != "cuda":
                k_chunk = k_chunk.to("cuda", non_blocking=True)
            if v_chunk.device.type != "cuda":
                v_chunk = v_chunk.to("cuda", non_blocking=True)
            self.pool.k_cache[layer_id, bid, off:off + take] = k_chunk.transpose(0, 1).to(dtype=self.pool.dtype)
            self.pool.v_cache[layer_id, bid, off:off + take] = v_chunk.transpose(0, 1).to(dtype=self.pool.dtype)
            src += take
            pos += take
        return int(pos)

    def _pad_piece_to_block(self, K_piece: torch.Tensor, V_piece: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = int(self.pool.B)
        missing = B - int(K_piece.shape[1])
        if missing <= 0:
            return K_piece, V_piece
        shape = (int(K_piece.shape[0]), missing, int(K_piece.shape[2]))
        k_pad = torch.zeros(shape, dtype=K_piece.dtype, device=K_piece.device)
        v_pad = torch.zeros(shape, dtype=V_piece.dtype, device=V_piece.device)
        return torch.cat([K_piece, k_pad], dim=1), torch.cat([V_piece, v_pad], dim=1)

    def _write_block_indexed_layer_to_pool(
        self,
        layer_id: int,
        block_ids: List[int],
        K: torch.Tensor,
        V: torch.Tensor,
        source_block_indices: List[int],
    ) -> bool:
        """Vectorized aligned writeback: one K and one V scatter per layer."""
        B = int(self.pool.B)
        if not source_block_indices or len(source_block_indices) != len(block_ids):
            return False
        if int(K.shape[1]) % B != 0 or int(V.shape[1]) % B != 0:
            return False
        max_src = max(int(x) for x in source_block_indices)
        raw_blocks = int(K.shape[1]) // B
        if max_src < 0 or max_src >= raw_blocks:
            return False
        src_device = K.device
        idx = torch.tensor([int(x) for x in source_block_indices], dtype=torch.long, device=src_device)
        K_blocks = K.contiguous().view(int(K.shape[0]), raw_blocks, B, int(K.shape[2]))
        V_blocks = V.contiguous().view(int(V.shape[0]), raw_blocks, B, int(V.shape[2]))
        K_sel = K_blocks.index_select(1, idx).permute(1, 2, 0, 3).contiguous()
        V_sel = V_blocks.index_select(1, idx).permute(1, 2, 0, 3).contiguous()
        if K_sel.device.type != "cuda":
            K_sel = K_sel.to("cuda", non_blocking=True)
        if V_sel.device.type != "cuda":
            V_sel = V_sel.to("cuda", non_blocking=True)
        dst = torch.tensor([int(x) for x in block_ids], dtype=torch.long, device="cuda")
        self.pool.k_cache[layer_id, dst] = K_sel.to(dtype=self.pool.dtype)
        self.pool.v_cache[layer_id, dst] = V_sel.to(dtype=self.pool.dtype)
        return True

    def _write_selected_layer_to_pool(
        self,
        layer_id: int,
        block_ids: List[int],
        K: torch.Tensor,
        V: torch.Tensor,
        selected_mid_blocks: List[int],
        effective_len: int,
        mode: str,
    ) -> torch.Tensor:
        L = int(K.shape[1])
        anchor_start = max(0, L - int(self.snapkv.Lo))
        anchor_end = min(L, anchor_start + int(self.snapkv_anchor_tokens))
        anchor_k_mean = None
        if anchor_end > anchor_start:
            anchor_k_mean = K[:, anchor_start:anchor_end, :].float().mean(dim=(0, 1)).detach().cpu()
        Ls = int(self.snapkv.Ls)
        Lo = int(self.snapkv.Lo)
        B = int(self.pool.B)
        if mode in ("input_le_budget", "too_short"):
            if int(effective_len) % B == 0 and int(K.shape[1]) >= int(effective_len):
                source_blocks = list(range(int(effective_len) // B))
                if self._write_block_indexed_layer_to_pool(layer_id, block_ids, K[:, :effective_len, :], V[:, :effective_len, :], source_blocks):
                    return anchor_k_mean
            self._write_piece_to_blocks(layer_id, block_ids, 0, K[:, :effective_len, :], V[:, :effective_len, :])
            return anchor_k_mean

        # Fast path: sink, selected mid blocks, and obs are all full 16-token blocks.
        # If L is not block-aligned, obs is the last 16 real tokens and must stay a slice.
        if (
            Ls % B == 0
            and Lo % B == 0
            and L % B == 0
            and int(effective_len) % B == 0
            and len(block_ids) == int(effective_len) // B
        ):
            sink_blocks = list(range(0, Ls // B))
            mid_base = Ls // B
            mid_blocks = [mid_base + int(x) for x in selected_mid_blocks]
            obs_start = L - Lo
            obs_blocks = list(range(obs_start // B, L // B))
            source_blocks = sink_blocks + mid_blocks + obs_blocks
            if len(source_blocks) == len(block_ids):
                if self._write_block_indexed_layer_to_pool(layer_id, block_ids, K, V, source_blocks):
                    return anchor_k_mean

        pos = 0
        pos = self._write_piece_to_blocks(layer_id, block_ids, pos, K[:, :Ls, :], V[:, :Ls, :])
        mid_start = Ls
        mid_end = max(mid_start, L - Lo)
        for mid_idx in selected_mid_blocks:
            start = mid_start + int(mid_idx) * B
            end = min(start + B, mid_end)
            if end > start:
                K_piece, V_piece = K[:, start:end, :], V[:, start:end, :]
            else:
                K_piece = K[:, :0, :]
                V_piece = V[:, :0, :]
            K_piece, V_piece = self._pad_piece_to_block(K_piece, V_piece)
            pos = self._write_piece_to_blocks(layer_id, block_ids, pos, K_piece, V_piece)
        obs_start = max(0, L - Lo)
        pos = self._write_piece_to_blocks(layer_id, block_ids, pos, K[:, obs_start:L, :], V[:, obs_start:L, :])
        if int(pos) != int(effective_len):
            raise RuntimeError(f"selected_writeback_len_mismatch got={pos} expected={effective_len}")
        return anchor_k_mean

    def _estimate_writeback_required_gb(self, effective_len: int) -> float:
        bytes_per_layer = 2 * int(self.num_kv_heads) * int(effective_len) * int(self.head_dim) * 2
        return float(bytes_per_layer) / 1024**3 + float(self.selected_writeback_safety_margin_gb)

    def _selected_writeback_transaction(
        self,
        seq_id: int,
        raw_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        selected_mid_blocks_by_layer,
        effective_len: int,
        logical_len: int,
        mode: str,
        backend: str,
    ) -> None:
        n_req = int(math.ceil(max(1, int(effective_len)) / int(self.pool.B)))
        block_ids = self._allocate_writeback_blocks(seq_id, n_req)
        entries: Dict[Tuple[int, int], BlockTableEntry] = {}
        try:
            if backend == "gpu_selected_writeback" and str(os.environ.get("KV_MIDDLEWARE_FORCE_GPU_WRITEBACK_OOM", "")).strip().lower() in {"1", "true", "yes", "on"}:
                raise torch.cuda.OutOfMemoryError("forced selected writeback OOM")
            anchor_means: Dict[int, Optional[torch.Tensor]] = {}
            for layer_id in range(self.num_layers):
                raw = raw_by_layer.get(layer_id)
                if raw is None:
                    raise RuntimeError(f"selected_writeback_missing_raw seq={seq_id} layer={layer_id}")
                K, V = raw
                if backend == "cpu_selected_compaction":
                    if K.device.type == "cuda" or V.device.type == "cuda":
                        K_cpu = K.detach().to("cpu").contiguous()
                        V_cpu = V.detach().to("cpu").contiguous()
                        self.raw_kv_cpu_stash_bytes += self._tensor_nbytes(K_cpu) + self._tensor_nbytes(V_cpu)
                        self.raw_kv_cache[layer_id][seq_id] = (K_cpu, V_cpu)
                        raw_by_layer[layer_id] = (K_cpu, V_cpu)
                        K, V = K_cpu, V_cpu
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                else:
                    if K.device.type != "cuda":
                        K = K.to("cuda")
                    if V.device.type != "cuda":
                        V = V.to("cuda")
                if isinstance(selected_mid_blocks_by_layer, dict):
                    layer_selected_mid_blocks = selected_mid_blocks_by_layer.get(int(layer_id), [])
                else:
                    layer_selected_mid_blocks = selected_mid_blocks_by_layer
                anchor_means[layer_id] = self._write_selected_layer_to_pool(
                    layer_id, block_ids, K, V, layer_selected_mid_blocks, int(effective_len), mode
                )
            for layer_id in range(self.num_layers):
                anchor_k = anchor_means.get(layer_id)
                if torch.is_tensor(anchor_k):
                    anchor_k = anchor_k.detach().cpu()
                entries[(int(seq_id), int(layer_id))] = BlockTableEntry(
                    seq_id=int(seq_id),
                    layer_id=int(layer_id),
                    state=OffloadState.ON_GPU,
                    block_ids=list(block_ids),
                    seq_len=int(effective_len),
                    logical_seq_len=int(logical_len),
                    materialized_blocks=len(block_ids),
                    prefill_anchor_k_mean=anchor_k,
                    gpu_block_map=list(block_ids),
                    cpu_k_blocks=[None] * len(block_ids),
                    cpu_v_blocks=[None] * len(block_ids),
                )
            with self.offloader.lock:
                for key, entry in entries.items():
                    self.offloader.page_table[key] = entry
            self.offloader._mark_resident(int(seq_id))
        except Exception:
            self.writeback_transaction_rollbacks += 1
            try:
                self.pool.free_blocks_by_ids(block_ids)
            except Exception:
                pass
            raise

    def _post_prefill_compress_selected_writeback(self, seq_ids: List[int], reset_decode_state: bool = True):
        self.phase = PhaseState.COMPRESS
        for seq_id in seq_ids:
            raw_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            for layer_id in range(self.num_layers):
                raw = self.raw_kv_cache[layer_id].get(seq_id)
                if raw is None:
                    raise RuntimeError(f"selected_writeback_missing_raw seq={seq_id} layer={layer_id}")
                raw_by_layer[layer_id] = raw
            selected_mid_blocks_by_layer, effective_len, logical_len, mode = self._compute_per_layer_selected_blocks(int(seq_id), raw_by_layer)
            self.writeback_est_required_gb = float(self._estimate_writeback_required_gb(effective_len))
            self.writeback_free_gb = float(self._selected_writeback_cuda_free_gb())
            force_cpu = str(os.environ.get("KV_MIDDLEWARE_FORCE_CPU_SELECTED_COMPACTION", "")).strip().lower() in {"1", "true", "yes", "on"}
            backend = "cpu_selected_compaction" if force_cpu or self.writeback_free_gb < self.writeback_est_required_gb else "gpu_selected_writeback"
            try:
                self._selected_writeback_transaction(seq_id, raw_by_layer, selected_mid_blocks_by_layer, effective_len, logical_len, mode, backend)
            except Exception as exc:
                cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
                is_oom = (cuda_oom_type is not None and isinstance(exc, cuda_oom_type)) or ("out of memory" in str(exc).lower())
                if backend == "gpu_selected_writeback" and is_oom:
                    self.gpu_writeback_oom_fallbacks += 1
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    backend = "cpu_selected_compaction"
                    self._selected_writeback_transaction(seq_id, raw_by_layer, selected_mid_blocks_by_layer, effective_len, logical_len, mode, backend)
                else:
                    self.prefill_writeback_backend = f"{backend}_failed"
                    raise
            if backend == "gpu_selected_writeback":
                self.gpu_selected_writeback_steps += 1
            elif backend == "cpu_selected_compaction":
                self.cpu_selected_compaction_steps += 1
            self.prefill_writeback_backend = backend
            for layer_id in range(self.num_layers):
                self.raw_kv_cache[layer_id].pop(seq_id, None)
                self.q_cache[layer_id].pop(seq_id, None)
        torch.cuda.empty_cache()
        self.phase = PhaseState.DECODE
        if reset_decode_state:
            self.decode_step_count = 0
            self._decode_window_step_thrash.clear()
            self._decode_cuda_free_post_cleanup_recent.clear()
        base_stats = self.offloader.get_stats(reset=False)
        self._decode_window_last_io_total = int(base_stats.get('offload_success', 0)) + int(base_stats.get('prefetch_success', 0))

    def post_prefill_compress(self, seq_ids: List[int], reset_decode_state: bool = True):
        if self.selected_writeback_enabled:
            return self._post_prefill_compress_selected_writeback(seq_ids, reset_decode_state=reset_decode_state)
        return self._post_prefill_compress_legacy(seq_ids, reset_decode_state=reset_decode_state)

    def _post_prefill_compress_legacy(self, seq_ids: List[int], reset_decode_state: bool = True):
        self.phase = PhaseState.COMPRESS

        for seq_id in seq_ids:
            all_K_c, all_V_c, all_anchor_k, seq_len_c = [], [], [], None
            logical_seq_len = None
            for layer_id in range(self.num_layers):
                raw = self.raw_kv_cache[layer_id].get(seq_id)
                if raw is None:
                    all_K_c.append(None)
                    all_V_c.append(None)
                    all_anchor_k.append(None)
                    continue

                K, V = raw
                if torch.is_tensor(K) and K.device.type != 'cuda':
                    K = K.to('cuda')
                if torch.is_tensor(V) and V.device.type != 'cuda':
                    V = V.to('cuda')
                L = K.shape[1]
                if logical_seq_len is None:
                    logical_seq_len = int(L)
                # prefill_anchors semantics:
                # fixed at prefill end, taken from the first `anchor_tokens` positions
                # inside the observation window [L-Lo, L), not the sink prefix.
                obs_start = max(0, int(L - self.snapkv.Lo))
                anchor_end = min(int(L), int(obs_start + self.snapkv_anchor_tokens))
                anchor_k_mean = None
                if anchor_end > obs_start:
                    anchor_k_mean = K[:, obs_start:anchor_end, :].float().mean(dim=(0, 1)).detach().cpu()
                retain_budget_tokens = int(getattr(self.snapkv, "retain_budget_tokens", 0) or 0)
                if (
                    float(getattr(self.snapkv, "retain_ratio", 1.0)) >= 1.0
                    or (retain_budget_tokens > 0 and L <= retain_budget_tokens)
                    or L < self.snapkv.Ls + self.snapkv.Lo + self.snapkv.B
                ):
                    # Raw mode should preserve the prefill KV as-is. Running the
                    # SnapKV materialization path at retain_ratio=1.0 creates a
                    # full-size temporary concatenate during activation and can
                    # OOM even though no compression is requested.
                    K_c, V_c = K, V
                else:
                    Q_full = self.q_cache[layer_id].get(seq_id)
                    if Q_full is not None:
                        Q_obs = Q_full[:, -self.snapkv.Lo:, :].unsqueeze(0)
                        Kb, Vb = self.snapkv.compress(K.unsqueeze(0), V.unsqueeze(0), Q_obs, self.n_rep)
                        K_c, V_c = Kb.squeeze(0), Vb.squeeze(0)
                    else:
                        K_c, V_c = K, V

                all_K_c.append(K_c)
                all_V_c.append(V_c)
                all_anchor_k.append(anchor_k_mean)
                if seq_len_c is None:
                    seq_len_c = K_c.shape[1]

            if seq_len_c is None:
                continue

            n_req = math.ceil(seq_len_c / self.pool.B)

            def _evict_one_for_prefill_allocation(target_free: int) -> None:
                # Allocation-time eviction cannot be purely async: if n_free does
                # not increase before the next loop iteration, this path can spin
                # forever while repeatedly issuing offloads.  Synchronize the
                # selected victim just for this hard allocation path.
                before = int(self.pool.n_free)
                victim = self.offloader.evict_coldest_sequence(seq_id)
                try:
                    self.offloader._try_finalize_offload(int(victim), force_sync=True)
                except Exception:
                    pass
                after = int(self.pool.n_free)
                if after <= before and after < int(target_free):
                    raise MemoryError(
                        f"Eviction made no progress for seq={seq_id}: "
                        f"n_free={after}, target_free={int(target_free)}"
                    )

            # Hysteresis eviction: only trigger when below low watermark, then
            # evict up to high watermark to avoid immediate re-trigger.
            if self.pool.n_free < self.pool.N_wm_low:
                target_free = max(n_req, self.pool.N_wm_high)
                while self.pool.n_free < target_free:
                    _evict_one_for_prefill_allocation(target_free)

            while self.pool.n_free < n_req:
                _evict_one_for_prefill_allocation(n_req)

            block_ids = self.pool.allocate_blocks(n_req)
            if not block_ids:
                raise MemoryError(f"Cannot allocate {n_req} blocks for seq={seq_id}")

            self.offloader.register_all_layers(
                seq_id,
                block_ids,
                seq_len_c,
                all_K_c,
                all_V_c,
                anchor_k_means=all_anchor_k,
                logical_seq_len=(seq_len_c if logical_seq_len is None else logical_seq_len),
            )
            for layer_id in range(self.num_layers):
                self.raw_kv_cache[layer_id].pop(seq_id, None)
                self.q_cache[layer_id].pop(seq_id, None)

        torch.cuda.empty_cache()
        self.phase = PhaseState.DECODE
        if reset_decode_state:
            self.decode_step_count = 0
            self._decode_window_step_thrash.clear()
            self._decode_cuda_free_post_cleanup_recent.clear()
        base_stats = self.offloader.get_stats(reset=False)
        self._decode_window_last_io_total = int(base_stats.get('offload_success', 0)) + int(
            base_stats.get('prefetch_success', 0)
        )

    def reset_runtime_state(self, cuda_free_gb: float = float("inf")) -> None:
        self.decode_step_count = 0
        self._decode_window_step_thrash.clear()
        self._decode_cuda_free_post_cleanup_recent.clear()
        base_stats = self.offloader.get_stats(reset=False)
        self._decode_window_last_io_total = int(base_stats.get('offload_success', 0)) + int(
            base_stats.get('prefetch_success', 0)
        )
        self._p2_attempts = 0
        self._p2_successes = 0
        self._p2_fail_streak = 0
        self._p2_last_attempted = False
        self._p2_last_success = False
        self._p2_last_no_candidate = False
        self._p2_last_candidate_count = 0
        self._p2_managed_active = False
        self._p2_recover_streak = 0
        self._p2_active_steps = 0
        self._p2_candidate_steps = 0
        self._p2_recovery_fail_windows = 0
        self._p2_no_candidate_steps = 0
        self._p2_no_candidate_pressure_streak = 0
        self._p2_ready_candidate_steps = 0
        self._p2_decode_candidate_steps = 0
        self._p2_expected_reclaim_blocks = 0
        self._p2_gain_success_steps = 0
        self._p2_gain_fail_steps = 0
        self._p2_skipped_low_benefit_steps = 0
        self._p2_gain_fail_cooldown_until = 0
        self._p2_gain_recent.clear()
        self._first_p2_step = -1
        self._p2_cuda_pressure_signal_steps = 0
        self._p2_cuda_pressure_accounted_step = -1
        self._p2_activity_accounted_step = -1
        self._last_pressure_signal = False
        self._page16_native_kernel_ms_accum = 0.0
        self._page16_native_kernel_calls = 0
        self.end_paged_direct_decode()
        self.set_cuda_free_post_cleanup_gb(cuda_free_gb)

    def prepare_page16_native_context(self, seq_ids: List[int]) -> Dict[str, Any]:
        seq_ids = [int(sid) for sid in seq_ids]
        if not seq_ids:
            raise RuntimeError("page16_native_empty_batch")
        ok, reason = self.page16_native_decode_support()
        if not ok:
            raise RuntimeError(f"page16_native_blocked:{reason}")

        seq_lens_ref: Optional[List[int]] = None
        logical_lens_ref: Optional[List[int]] = None
        resident_missing_blocks = 0
        for sid in seq_ids:
            res = self.offloader.reserve_decode_slot(sid)
            if not res.ok:
                raise RuntimeError(f"page16_native_blocked:{res.reason}")

        layer_views: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_id in range(self.num_layers):
            entries = []
            for sid in seq_ids:
                entry = self.offloader.page_table.get((sid, layer_id))
                if entry is None:
                    raise RuntimeError(f"page16_native_missing_entry seq={sid} layer={layer_id}")
                self.offloader._ensure_entry_maps(entry)
                if entry.state != OffloadState.ON_GPU:
                    raise RuntimeError(
                        f"page16_native_not_on_gpu seq={sid} layer={layer_id} state={entry.state}"
                    )
                missing = sum(1 for bid in entry.gpu_block_map if int(bid) < 0)
                resident_missing_blocks += int(missing)
                if missing:
                    raise RuntimeError(
                        f"page16_native_resident_missing seq={sid} layer={layer_id} missing={missing}"
                    )
                if not entry.gpu_block_map:
                    raise RuntimeError(f"page16_native_empty_blocks seq={sid} layer={layer_id}")
                entries.append(entry)

            seq_lens = [int(e.seq_len) for e in entries]
            logical_lens = [int(getattr(e, 'logical_seq_len', 0) or e.seq_len) for e in entries]
            if seq_lens_ref is None:
                seq_lens_ref = list(seq_lens)
                logical_lens_ref = list(logical_lens)
            elif seq_lens != seq_lens_ref:
                raise RuntimeError("page16_native_inconsistent_seq_lens")
            elif logical_lens != logical_lens_ref:
                raise RuntimeError("page16_native_inconsistent_logical_lens")

            seq_block_ids_list = [list(e.gpu_block_map) for e in entries]
            k_cache, v_cache, block_table = self.pool.build_block_table(layer_id, seq_block_ids_list)
            layer_views[layer_id] = {
                'k_cache': k_cache,
                'v_cache': v_cache,
                'block_table': block_table,
            }

        if seq_lens_ref is None or logical_lens_ref is None:
            raise RuntimeError("page16_native_no_layers")
        cache_seqlens = torch.tensor(seq_lens_ref, dtype=torch.int32, device='cuda')
        return {
            'backend': 'page16_native',
            'seq_ids': list(seq_ids),
            'seq_lens': list(seq_lens_ref),
            'logical_seq_lens': list(logical_lens_ref),
            'cache_seqlens': cache_seqlens,
            'layer_views': layer_views,
            'resident_missing_blocks': int(resident_missing_blocks),
            'materialized_blocks': int(sum(len(v['block_table'].reshape(-1)) for v in layer_views.values())),
        }

    def prepare_paged_direct_context(self, seq_ids: List[int]) -> Dict[str, Any]:
        """Build a strict resident paged-KV context for direct decode.

        The returned cache_seqlens are the pre-forward lengths. reserve_decode_slot()
        may append a new physical page for the token that flash-attn will write.
        """
        seq_ids = [int(sid) for sid in seq_ids]
        if not seq_ids:
            raise RuntimeError("paged_direct_empty_batch")
        ok, reason = self.paged_flash_decode_support()
        if not ok:
            raise RuntimeError(f"paged_direct_blocked:{reason}")

        seq_lens_ref: Optional[List[int]] = None
        logical_lens_ref: Optional[List[int]] = None
        resident_missing_blocks = 0

        for sid in seq_ids:
            res = self.offloader.reserve_decode_slot(sid)
            if not res.ok:
                raise RuntimeError(f"paged_direct_blocked:{res.reason}")

        layer_views: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_id in range(self.num_layers):
            entries = []
            for sid in seq_ids:
                entry = self.offloader.page_table.get((sid, layer_id))
                if entry is None:
                    raise RuntimeError(f"paged_direct_missing_entry seq={sid} layer={layer_id}")
                self.offloader._ensure_entry_maps(entry)
                if entry.state != OffloadState.ON_GPU:
                    raise RuntimeError(
                        f"paged_direct_not_on_gpu seq={sid} layer={layer_id} state={entry.state}"
                    )
                missing = sum(1 for bid in entry.gpu_block_map if int(bid) < 0)
                resident_missing_blocks += int(missing)
                if missing:
                    raise RuntimeError(
                        f"paged_direct_resident_missing seq={sid} layer={layer_id} missing={missing}"
                    )
                if not entry.gpu_block_map:
                    raise RuntimeError(f"paged_direct_empty_blocks seq={sid} layer={layer_id}")
                entries.append(entry)

            seq_lens = [int(e.seq_len) for e in entries]
            logical_lens = [int(getattr(e, 'logical_seq_len', 0) or e.seq_len) for e in entries]
            if seq_lens_ref is None:
                seq_lens_ref = list(seq_lens)
                logical_lens_ref = list(logical_lens)
            elif seq_lens != seq_lens_ref:
                raise RuntimeError("paged_direct_inconsistent_seq_lens")
            elif logical_lens != logical_lens_ref:
                raise RuntimeError("paged_direct_inconsistent_logical_lens")

            seq_block_ids_list = [list(e.gpu_block_map) for e in entries]
            k_cache, v_cache, block_table = self.pool.build_block_table(layer_id, seq_block_ids_list)
            layer_views[layer_id] = {
                'k_cache': k_cache,
                'v_cache': v_cache,
                'block_table': block_table,
            }

        if seq_lens_ref is None or logical_lens_ref is None:
            raise RuntimeError("paged_direct_no_layers")

        cache_seqlens = torch.tensor(seq_lens_ref, dtype=torch.int32, device='cuda')
        return {
            'backend': 'paged_direct_flash',
            'seq_ids': list(seq_ids),
            'seq_lens': list(seq_lens_ref),
            'logical_seq_lens': list(logical_lens_ref),
            'cache_seqlens': cache_seqlens,
            'layer_views': layer_views,
            'resident_missing_blocks': int(resident_missing_blocks),
            'materialized_blocks': int(sum(len(v['block_table'].reshape(-1)) for v in layer_views.values())),
        }

    def begin_paged_direct_decode(self, ctx: Dict[str, Any]) -> None:
        self._paged_direct_context = ctx
        self._paged_direct_active = True

    def end_paged_direct_decode(self) -> None:
        self._paged_direct_active = False
        self._paged_direct_context = None

    def commit_paged_direct_decode(self, seq_ids: List[int]) -> None:
        for sid in [int(s) for s in seq_ids]:
            for layer_id in range(self.num_layers):
                entry = self.offloader.page_table.get((sid, layer_id))
                if entry is None:
                    raise RuntimeError(f"paged_direct_commit_missing_entry seq={sid} layer={layer_id}")
                self.offloader._ensure_entry_maps(entry)
                if int(getattr(entry, 'logical_seq_len', 0)) <= 0:
                    entry.logical_seq_len = int(entry.seq_len)
                entry.seq_len += 1
                entry.logical_seq_len += 1
                entry.last_access = int(self.decode_step_count)
                entry.materialized_blocks = max(
                    int(getattr(entry, 'materialized_blocks', 0) or 0),
                    len(entry.gpu_block_map),
                )
                self.offloader._update_entry_state(entry)
            self.offloader._mark_resident(sid)
            self.offloader._inc_stat('decode_append_success')

    def decode_step_schedule(self, layer_id: int, seq_id: int, k_tok: torch.Tensor, v_tok: torch.Tensor):
        entry = self.offloader.page_table.get((seq_id, layer_id))
        if entry:
            entry.last_access = self.decode_step_count

        return self.offloader.append_decode_token(seq_id, layer_id, k_tok, v_tok)

    def p2_low_threshold(self) -> int:
        return int(self.pool.N_wm_low)

    def p2_target_free_threshold(self) -> int:
        cfg_target = int(self.p2_target_free_blocks_cfg or 0)
        if cfg_target > 0:
            return max(int(self.p2_low_threshold()), cfg_target)
        low = int(self.p2_low_threshold())
        high = int(self.pool.N_wm_high)
        hysteresis = max(
            int(self.p2_min_reclaim_blocks),
            int(self.ready_decode_eviction_threshold),
        )
        return max(low, min(high, low + hysteresis))

    def kv_used_ratio(self) -> float:
        total = max(1, int(self.pool.N_total))
        used = max(0, total - int(self.pool.n_free))
        return float(used) / float(total)

    def set_cuda_free_post_cleanup_gb(self, free_gb: float):
        try:
            val = float(free_gb)
        except Exception:
            val = float("inf")
        if not math.isfinite(val) or val < 0.0:
            val = float("inf")
        self._cuda_free_post_cleanup_gb = val

    def cuda_free_post_cleanup_gb(self) -> float:
        return float(self._cuda_free_post_cleanup_gb)

    def cuda_free_post_cleanup_last_gb(self) -> float:
        return float(self._cuda_free_post_cleanup_gb)

    def decode_cuda_free_post_cleanup_recent_min_gb(self) -> float:
        if not self._decode_cuda_free_post_cleanup_recent:
            return float("inf")
        return float(min(self._decode_cuda_free_post_cleanup_recent))

    def p2_cuda_pressure_signal(self, has_decode_work: bool) -> bool:
        if not bool(has_decode_work):
            return False
        return float(self.decode_cuda_free_post_cleanup_recent_min_gb()) <= float(self.p2_cuda_pressure_min_gb)

    def p2_gain_fail_cooldown_active(self) -> bool:
        return int(self.decode_step_count) < int(self._p2_gain_fail_cooldown_until)

    def min_resident_blocks_required(self, logical_blocks: int) -> int:
        logical = max(0, int(logical_blocks))
        if logical <= 0 or self.kv_min_resident_ratio <= 0.0:
            return 0
        return max(1, int(math.ceil(float(logical) * float(self.kv_min_resident_ratio))))

    def _enter_p2_managed(self):
        if not self._p2_managed_active:
            self._p2_managed_active = True
            self._p2_recover_streak = 0

    def _exit_p2_managed(self):
        self._p2_managed_active = False
        self._p2_recover_streak = 0
        self._p2_fail_streak = 0
        self._p2_last_attempted = False
        self._p2_last_success = False
        self._p2_last_no_candidate = False
        self._p2_last_candidate_count = 0

    def record_p2_attempt(
        self,
        attempted: bool,
        success: bool,
        no_candidate: bool = False,
        candidate_count: int = 0,
    ):
        if not bool(self.p2_enabled):
            self._exit_p2_managed()
            self._p2_no_candidate_pressure_streak = 0
            return
        self._p2_last_attempted = bool(attempted)
        self._p2_last_success = bool(success)
        self._p2_last_no_candidate = bool(no_candidate)
        self._p2_last_candidate_count = max(0, int(candidate_count))
        low_pressure = (
            int(self.pool.n_free) <= int(self.p2_low_threshold())
            or bool(self.p2_cuda_pressure_signal(has_decode_work=True))
        )
        should_manage = bool(
            low_pressure and (
                bool(attempted)
                or int(self._p2_last_candidate_count) > 0
                or bool(self._p2_managed_active)
            )
        )
        if should_manage:
            self._enter_p2_managed()
            if int(self._p2_activity_accounted_step) != int(self.decode_step_count):
                self._p2_active_steps += 1
                self._p2_activity_accounted_step = int(self.decode_step_count)
            if bool(attempted) and int(self._first_p2_step) < 0:
                self._first_p2_step = int(self.decode_step_count)
            if self._p2_last_candidate_count > 0:
                self._p2_candidate_steps += 1
            if bool(self._p2_last_no_candidate):
                self._p2_no_candidate_steps += 1
        elif low_pressure and bool(self._p2_last_no_candidate):
            self._p2_no_candidate_steps += 1
        if low_pressure and bool(self._p2_last_no_candidate):
            self._p2_no_candidate_pressure_streak += 1
        elif low_pressure and (bool(attempted) or int(self._p2_last_candidate_count) > 0):
            self._p2_no_candidate_pressure_streak = 0
        elif not low_pressure:
            self._p2_no_candidate_pressure_streak = 0
        if attempted:
            self._p2_attempts += 1
        if attempted and success:
            self._p2_successes += 1
            self._p2_fail_streak = 0
        elif attempted and low_pressure and should_manage:
            self._p2_fail_streak += 1
            self._p2_recovery_fail_windows += 1
        elif low_pressure and (not should_manage):
            self._p2_fail_streak = 0
        elif not low_pressure:
            self._p2_fail_streak = 0

    def record_p2_candidate_mix(
        self,
        ready_candidate_count: int,
        decode_candidate_count: int,
        expected_reclaim_blocks: int,
    ) -> None:
        ready_count = max(0, int(ready_candidate_count))
        decode_count = max(0, int(decode_candidate_count))
        self._p2_expected_reclaim_blocks = max(0, int(expected_reclaim_blocks))
        if ready_count > 0:
            self._p2_ready_candidate_steps += 1
        if decode_count > 0:
            self._p2_decode_candidate_steps += 1

    def reset_p2_ready_step_telemetry(self) -> None:
        self._p2_ready_sequences_selected_per_step = 0
        self._p2_ready_offload_blocks_per_step = 0
        self._p2_ready_target_reclaim_blocks = 0
        self._p2_ready_actual_reclaim_blocks = 0
        self._p2_ready_stop_reason = "none"

    def record_p2_ready_selection(self, selected_sequences: int, planned_blocks: int, target_reclaim_blocks: int, actual_reclaim_blocks: int, stop_reason: str) -> None:
        reason = str(stop_reason or "unknown")
        self._p2_ready_sequences_selected_per_step = max(0, int(selected_sequences))
        self._p2_ready_offload_blocks_per_step = max(0, int(planned_blocks))
        self._p2_ready_target_reclaim_blocks = max(0, int(target_reclaim_blocks))
        self._p2_ready_actual_reclaim_blocks = max(0, int(actual_reclaim_blocks))
        self._p2_ready_stop_reason = reason
        self._p2_ready_stop_reason_counts[reason] = int(self._p2_ready_stop_reason_counts.get(reason, 0)) + 1

    def record_p2_ready_offload_blocks(self, blocks: int, sequence_steps: int = 1) -> None:
        n = max(0, int(blocks))
        if n <= 0:
            return
        self._p2_ready_offload_blocks_last = n
        self._p2_ready_offload_blocks_total += n
        self._p2_ready_offload_sequence_steps += max(1, int(sequence_steps))
        step = int(self.decode_step_count)
        if int(self._p2_ready_offload_last_step) != step:
            self._p2_ready_offload_decode_steps += 1
            self._p2_ready_offload_last_step = step

    def record_p2_low_benefit_skip(self, expected_reclaim_blocks: int) -> None:
        self._p2_expected_reclaim_blocks = max(0, int(expected_reclaim_blocks))
        self._p2_skipped_low_benefit_steps += 1
        self._p2_gain_recent.append(0)
        if (
            len(self._p2_gain_recent) >= int(self.p2_gain_window_steps)
            and sum(self._p2_gain_recent) <= 0
            and int(self.p2_gain_fail_cooldown_steps) > 0
        ):
            self._p2_gain_fail_cooldown_until = int(self.decode_step_count) + int(self.p2_gain_fail_cooldown_steps)

    def record_p2_gain_result(self, gain_success: bool) -> None:
        ok = bool(gain_success)
        self._p2_gain_recent.append(1 if ok else 0)
        if ok:
            self._p2_gain_success_steps += 1
            self._p2_gain_fail_cooldown_until = 0
        else:
            self._p2_gain_fail_steps += 1
            if (
                len(self._p2_gain_recent) >= int(self.p2_gain_window_steps)
                and sum(self._p2_gain_recent) <= 0
                and int(self.p2_gain_fail_cooldown_steps) > 0
            ):
                self._p2_gain_fail_cooldown_until = int(self.decode_step_count) + int(self.p2_gain_fail_cooldown_steps)

    def _short_window_thrash_avg(self) -> float:
        if not self._decode_window_step_thrash:
            return 0.0
        return float(sum(self._decode_window_step_thrash) / len(self._decode_window_step_thrash))

    def get_thrash_win16(self) -> float:
        return float(self._short_window_thrash_avg())

    def _update_short_window_thrash(self, active_seq_count: int):
        stats = self.offloader.get_stats(reset=False)
        io_total = int(stats.get('offload_success', 0)) + int(stats.get('prefetch_success', 0))
        delta = max(0, io_total - self._decode_window_last_io_total)
        self._decode_window_last_io_total = io_total
        denom = max(1, int(active_seq_count))
        self._decode_window_step_thrash.append(float(delta) / float(denom))

    def update_decode_pressure(self, active_seq_count: int = 0, refresh_thrash: bool = True):
        if refresh_thrash:
            self._update_short_window_thrash(active_seq_count)
        if int(active_seq_count) > 0 and math.isfinite(float(self._cuda_free_post_cleanup_gb)):
            self._decode_cuda_free_post_cleanup_recent.append(float(self._cuda_free_post_cleanup_gb))
        thrash_win16 = self._short_window_thrash_avg()
        n_free = int(self.pool.n_free)
        high_threshold = int(self.pool.N_wm_high)
        p2_low_threshold = int(self.p2_low_threshold())
        p2_target_threshold = int(self.p2_target_free_threshold())
        has_decode_work = int(active_seq_count) > 0
        p2_cuda_pressure_signal = self.p2_cuda_pressure_signal(has_decode_work=has_decode_work)
        raw_pressure_signal = n_free <= p2_low_threshold
        recover_signal = (
            n_free >= high_threshold
            and not bool(p2_cuda_pressure_signal)
        )
        if self.decode_window_tiered:
            raw_pressure_signal = raw_pressure_signal or thrash_win16 >= self.decode_window_thrash_low
            recover_signal = recover_signal and thrash_win16 < self.decode_window_thrash_low

        raw_pressure_signal = bool(raw_pressure_signal or p2_cuda_pressure_signal)

        if bool(p2_cuda_pressure_signal) and int(self._p2_cuda_pressure_accounted_step) != int(self.decode_step_count):
            self._p2_cuda_pressure_signal_steps += 1
            self._p2_cuda_pressure_accounted_step = int(self.decode_step_count)

        self._last_pressure_signal = bool(raw_pressure_signal)

        if self._p2_managed_active:
            if n_free >= p2_target_threshold and recover_signal:
                self._p2_recover_streak += 1
            else:
                self._p2_recover_streak = 0
            if self._p2_recover_streak >= self.decode_window_recover_steps:
                self._exit_p2_managed()

    def get_decode_window_status(self) -> Dict[str, int]:
        return {
            'recover_steps': int(self.decode_window_recover_steps),
            'thrash_win16_window_steps': int(self.decode_window_thrash_window_steps),
            'thrash_win16_low_x1000': int(self.decode_window_thrash_low * 1000),
            'thrash_win16_high_x1000': int(self.decode_window_thrash_high * 1000),
            'thrash_win16_x1000': int(self._short_window_thrash_avg() * 1000),
            'wm_low': int(self.pool.N_wm_low),
            'wm_high': int(self.pool.N_wm_high),
            'wm_low_ratio_x1000': int(float(self.wm_low_ratio) * 1000.0),
            'wm_high_ratio_x1000': int(float(self.wm_high_ratio) * 1000.0),
            'n_free': int(self.pool.n_free),
            'p2_low_threshold': int(self.p2_low_threshold()),
            'p2_target_free_blocks': int(self.p2_target_free_blocks_cfg),
            'p2_recover_threshold': int(self.p2_target_free_threshold()),
            'p2_attempts': int(self._p2_attempts),
            'p2_successes': int(self._p2_successes),
            'p2_fail_streak': int(self._p2_fail_streak),
            'p2_last_attempted': int(bool(self._p2_last_attempted)),
            'p2_last_success': int(bool(self._p2_last_success)),
            'p2_last_no_candidate': int(bool(self._p2_last_no_candidate)),
            'p2_last_candidate_count': int(self._p2_last_candidate_count),
            'p2_managed_active': int(bool(self._p2_managed_active)),
            'p2_recover_streak': int(self._p2_recover_streak),
            'p2_active_steps': int(self._p2_active_steps),
            'p2_candidate_steps': int(self._p2_candidate_steps),
            'p2_recovery_fail_windows': int(self._p2_recovery_fail_windows),
            'p2_no_candidate_steps': int(self._p2_no_candidate_steps),
            'p2_attempted_steps': int(self._p2_attempts),
            'p2_success_steps': int(self._p2_successes),
            'p2_min_reclaim_blocks': int(self.p2_min_reclaim_blocks),
            'p2_ready_candidate_steps': int(self._p2_ready_candidate_steps),
            'p2_decode_candidate_steps': int(self._p2_decode_candidate_steps),
            'p2_expected_reclaim_blocks': int(self._p2_expected_reclaim_blocks),
            'p2_ready_offload_blocks_total': int(self._p2_ready_offload_blocks_total),
            'p2_ready_offload_blocks_last': int(self._p2_ready_offload_blocks_last),
            'p2_ready_offload_sequence_steps': int(self._p2_ready_offload_sequence_steps),
            'p2_ready_offload_decode_steps': int(self._p2_ready_offload_decode_steps),
            'p2_ready_sequences_selected_per_step': int(self._p2_ready_sequences_selected_per_step),
            'p2_ready_offload_blocks_per_step': int(self._p2_ready_offload_blocks_per_step),
            'p2_ready_target_reclaim_blocks': int(self._p2_ready_target_reclaim_blocks),
            'p2_ready_actual_reclaim_blocks': int(self._p2_ready_actual_reclaim_blocks),
            'p2_ready_stop_reason': str(self._p2_ready_stop_reason),
            'p2_ready_stop_target_reached_steps': int(self._p2_ready_stop_reason_counts.get('target_reached', 0)),
            'p2_ready_stop_sequence_cap_reached_steps': int(self._p2_ready_stop_reason_counts.get('sequence_cap_reached', 0)),
            'p2_ready_stop_block_cap_reached_steps': int(self._p2_ready_stop_reason_counts.get('block_cap_reached', 0)),
            'p2_ready_stop_low_benefit_skip_steps': int(self._p2_ready_stop_reason_counts.get('low_benefit_skip', 0)),
            'p2_ready_stop_not_needed_steps': int(self._p2_ready_stop_reason_counts.get('not_needed', 0)),
            'p2_ready_stop_no_ready_candidate_steps': int(self._p2_ready_stop_reason_counts.get('no_ready_candidate', 0)),
            'p2_gain_success_steps': int(self._p2_gain_success_steps),
            'p2_gain_fail_steps': int(self._p2_gain_fail_steps),
            'p2_skipped_low_benefit_steps': int(self._p2_skipped_low_benefit_steps),
            'p2_gain_fail_cooldown_until': int(self._p2_gain_fail_cooldown_until),
            'first_p2_step': int(self._first_p2_step),
            'p2_no_candidate_pressure_streak': int(self._p2_no_candidate_pressure_streak),
            'cuda_free_post_cleanup_gb_x1000': int(max(0.0, float(self._cuda_free_post_cleanup_gb)) * 1000.0) if math.isfinite(self._cuda_free_post_cleanup_gb) else -1,
            'decode_cuda_free_post_cleanup_recent_min_gb_x1000': int(max(0.0, float(self.decode_cuda_free_post_cleanup_recent_min_gb())) * 1000.0) if math.isfinite(self.decode_cuda_free_post_cleanup_recent_min_gb()) else -1,
            'p2_cuda_pressure_signal_steps': int(self._p2_cuda_pressure_signal_steps),
            'p2_cuda_pressure_min_gb_x1000': int(float(self.p2_cuda_pressure_min_gb) * 1000.0),
            'kv_min_resident_ratio_x1000': int(float(self.kv_min_resident_ratio) * 1000.0),
            'selected_writeback_enabled': int(bool(self.selected_writeback_enabled)),
            'prefill_writeback_backend': str(self.prefill_writeback_backend),
            'gpu_selected_writeback_steps': int(self.gpu_selected_writeback_steps),
            'cpu_selected_compaction_steps': int(self.cpu_selected_compaction_steps),
            'gpu_writeback_oom_fallbacks': int(self.gpu_writeback_oom_fallbacks),
            'writeback_transaction_rollbacks': int(self.writeback_transaction_rollbacks),
            'raw_kv_cpu_stash_bytes': int(self.raw_kv_cpu_stash_bytes),
            'selected_global_block_count': int(self.selected_global_block_count),
            'writeback_est_required_gb_x1000': int(max(0.0, float(self.writeback_est_required_gb)) * 1000.0),
            'writeback_free_gb_x1000': int(max(0.0, float(self.writeback_free_gb)) * 1000.0),
            'writeback_block_selection_shared_layers': int(self.writeback_block_selection_shared_layers),
            'score_full_attention_materialized': int(self.score_full_attention_materialized),
            'offload_budget_blocks': int(self.offloader.offload_budget_blocks),
            'prefetch_budget_blocks': int(self.offloader.prefetch_budget_blocks),
        }

    def release_sequence(self, seq_id: int):
        self.offloader.release_sequence(seq_id)
