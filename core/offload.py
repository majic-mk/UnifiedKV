import math
import threading
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from kv_types import BlockTableEntry, DecodeAppendResult, OffloadState
from pool import PagedKVPool


class AsyncOffloadManager:
    def __init__(self, pool: PagedKVPool, num_layers: int):
        self.pool = pool
        self.num_layers = num_layers
        self.current_step = 0
        self.min_resident_steps = 16
        self.offload_budget_blocks_base = 64
        self.offload_budget_blocks_max = 256
        self.prefetch_budget_blocks_base = 64
        self.prefetch_budget_blocks_max = 256
        self.p2_partial_offload_chunk_blocks = 128
        self.offload_budget_blocks = self.offload_budget_blocks_base
        self.prefetch_budget_blocks = self.prefetch_budget_blocks_base
        self._budget_step = -1
        self._offload_used_blocks = 0
        self._prefetch_used_blocks = 0
        self.lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.page_table: Dict[Tuple[int, int], BlockTableEntry] = {}
        self.seq_resident_until: Dict[int, int] = {}
        self.transfer_stream = torch.cuda.Stream(priority=-1)
        self.compute_stream = torch.cuda.default_stream()
        self._stats = {
            'evict_calls': 0,
            'evict_success': 0,
            'evict_fail': 0,
            'offload_calls': 0,
            'offload_success': 0,
            'offload_fail': 0,
            'prefetch_calls': 0,
            'prefetch_success': 0,
            'prefetch_fail': 0,
            'prefetch_noop': 0,
            'prefetch_inflight': 0,
            'ensure_calls': 0,
            'ensure_success': 0,
            'ensure_fail': 0,
            'window_prune_calls': 0,
            'window_prune_success': 0,
            'window_prune_fail': 0,
            'window_prune_skip': 0,
            'window_prune_skip_no_candidate_blocks': 0,
            'window_prune_skip_trigger_not_met': 0,
            'window_prune_skip_min_drop_not_met': 0,
            'window_prune_tokens_dropped': 0,
            'window_prune_frozen': 0,
            'window_prune_unfreeze': 0,
            'decode_append_calls': 0,
            'decode_append_success': 0,
            'decode_append_fail': 0,
            'decode_append_retryable': 0,
        }

    @staticmethod
    def _clamp(v: int, lo: int, hi: int) -> int:
        return max(int(lo), min(int(hi), int(v)))

    def configure_transfer_budget_limits(
        self,
        offload_base: Optional[int] = None,
        offload_max: Optional[int] = None,
        prefetch_base: Optional[int] = None,
        prefetch_max: Optional[int] = None,
        partial_chunk: Optional[int] = None,
    ):
        if offload_base is not None:
            self.offload_budget_blocks_base = max(1, int(offload_base))
        if offload_max is not None:
            self.offload_budget_blocks_max = max(self.offload_budget_blocks_base, int(offload_max))
        if prefetch_base is not None:
            self.prefetch_budget_blocks_base = max(1, int(prefetch_base))
        if prefetch_max is not None:
            self.prefetch_budget_blocks_max = max(self.prefetch_budget_blocks_base, int(prefetch_max))
        if partial_chunk is not None:
            self.p2_partial_offload_chunk_blocks = max(1, int(partial_chunk))
        self.offload_budget_blocks = self._clamp(self.offload_budget_blocks, self.offload_budget_blocks_base, self.offload_budget_blocks_max)
        self.prefetch_budget_blocks = self._clamp(self.prefetch_budget_blocks, self.prefetch_budget_blocks_base, self.prefetch_budget_blocks_max)

    def set_step_transfer_budgets(
        self,
        offload_budget: Optional[int] = None,
        prefetch_budget: Optional[int] = None,
    ):
        if offload_budget is not None:
            self.offload_budget_blocks = self._clamp(int(offload_budget), self.offload_budget_blocks_base, self.offload_budget_blocks_max)
        if prefetch_budget is not None:
            self.prefetch_budget_blocks = self._clamp(int(prefetch_budget), self.prefetch_budget_blocks_base, self.prefetch_budget_blocks_max)

    def _materialized_block_count(self, entry: BlockTableEntry) -> int:
        materialized = int(getattr(entry, 'materialized_blocks', 0) or 0)
        if materialized > 0:
            return materialized
        gpu_map = list(getattr(entry, 'gpu_block_map', []) or [])
        if gpu_map:
            return len(gpu_map)
        block_ids = list(getattr(entry, 'block_ids', []) or [])
        if block_ids:
            return len(block_ids)
        if self.pool.B <= 0:
            return 0
        seq_len = int(getattr(entry, 'seq_len', 0) or 0)
        return max(0, int(math.ceil(float(seq_len) / float(self.pool.B))))

    def _ensure_entry_maps(self, entry: BlockTableEntry):
        materialized_blocks = self._materialized_block_count(entry)
        entry.materialized_blocks = int(materialized_blocks)
        if materialized_blocks <= 0:
            entry.gpu_block_map = []
            entry.cpu_k_blocks = []
            entry.cpu_v_blocks = []
            return
        if not getattr(entry, 'gpu_block_map', None):
            entry.gpu_block_map = list(getattr(entry, 'block_ids', [])[:materialized_blocks])
        if len(entry.gpu_block_map) < materialized_blocks:
            entry.gpu_block_map = list(entry.gpu_block_map) + [-1] * (materialized_blocks - len(entry.gpu_block_map))
        else:
            entry.gpu_block_map = list(entry.gpu_block_map[:materialized_blocks])
        if not getattr(entry, 'cpu_k_blocks', None):
            entry.cpu_k_blocks = [None] * materialized_blocks
        if not getattr(entry, 'cpu_v_blocks', None):
            entry.cpu_v_blocks = [None] * materialized_blocks
        if len(entry.cpu_k_blocks) < materialized_blocks:
            entry.cpu_k_blocks = list(entry.cpu_k_blocks) + [None] * (materialized_blocks - len(entry.cpu_k_blocks))
        else:
            entry.cpu_k_blocks = list(entry.cpu_k_blocks[:materialized_blocks])
        if len(entry.cpu_v_blocks) < materialized_blocks:
            entry.cpu_v_blocks = list(entry.cpu_v_blocks) + [None] * (materialized_blocks - len(entry.cpu_v_blocks))
        else:
            entry.cpu_v_blocks = list(entry.cpu_v_blocks[:materialized_blocks])
        entry.block_ids = [int(bid) for bid in entry.gpu_block_map if int(bid) >= 0]

    def _resident_block_count(self, entry: BlockTableEntry) -> int:
        self._ensure_entry_maps(entry)
        return int(sum(1 for bid in entry.gpu_block_map if int(bid) >= 0))

    def _update_entry_state(self, entry: BlockTableEntry):
        self._ensure_entry_maps(entry)
        resident = self._resident_block_count(entry)
        logical = int(getattr(entry, 'materialized_blocks', 0) or len(entry.gpu_block_map))
        if resident <= 0:
            entry.state = OffloadState.ON_CPU
        elif resident >= logical:
            entry.state = OffloadState.ON_GPU
        else:
            entry.state = OffloadState.MIXED
        entry.block_ids = [int(bid) for bid in entry.gpu_block_map if int(bid) >= 0]

    def _block_token_range(self, logical_block_idx: int, seq_len: int) -> Tuple[int, int]:
        start = int(logical_block_idx) * int(self.pool.B)
        end = min(int(seq_len), start + int(self.pool.B))
        return start, end

    def _read_single_block_from_gpu(self, layer_id: int, block_id: int, logical_block_idx: int, seq_len: int):
        start, end = self._block_token_range(logical_block_idx, seq_len)
        chunk = max(0, end - start)
        if chunk <= 0:
            empty = torch.empty(self.pool.num_kv_heads, 0, self.pool.head_dim, dtype=self.pool.dtype, device='cuda')
            return empty, empty.clone()
        K = self.pool.k_cache[layer_id, int(block_id), :chunk].transpose(0, 1).contiguous()
        V = self.pool.v_cache[layer_id, int(block_id), :chunk].transpose(0, 1).contiguous()
        return K, V

    def _write_single_block_to_gpu(self, layer_id: int, block_id: int, logical_block_idx: int, seq_len: int, K: torch.Tensor, V: torch.Tensor):
        start, end = self._block_token_range(logical_block_idx, seq_len)
        chunk = max(0, end - start)
        if chunk <= 0:
            return
        self.pool.k_cache[layer_id, int(block_id), :chunk] = K[:, :chunk].transpose(0, 1)
        self.pool.v_cache[layer_id, int(block_id), :chunk] = V[:, :chunk].transpose(0, 1)

    def _bytes_for_block_indices(self, entry: BlockTableEntry, block_indices: List[int]) -> int:
        total = 0
        for idx in block_indices:
            start, end = self._block_token_range(idx, entry.seq_len)
            chunk = max(0, end - start)
            total += 2 * self.pool.num_kv_heads * chunk * self.pool.head_dim * 2
        return int(total)

    def _inc_stat(self, key: str, inc: int = 1):
        with self.stats_lock:
            self._stats[key] = self._stats.get(key, 0) + inc

    def _mark_window_prune_skip(self, reason: Optional[str] = None):
        self._inc_stat('window_prune_skip')
        if reason == 'no_candidate_blocks':
            self._inc_stat('window_prune_skip_no_candidate_blocks')
        elif reason == 'trigger_not_met':
            self._inc_stat('window_prune_skip_trigger_not_met')
        elif reason == 'min_drop_not_met':
            self._inc_stat('window_prune_skip_min_drop_not_met')

    def record_window_prune_skip_reason(self, reason: str):
        self._mark_window_prune_skip(reason=reason)

    def get_stats(self, reset: bool = False) -> Dict[str, int]:
        with self.stats_lock:
            out = dict(self._stats)
            if reset:
                for k in self._stats:
                    self._stats[k] = 0
        return out

    def set_decode_step(self, step: int):
        step_i = int(step)
        self.current_step = step_i
        if step_i != self._budget_step:
            self._budget_step = step_i
            self._offload_used_blocks = 0
            self._prefetch_used_blocks = 0

    def _try_consume_budget(self, kind: str, num_blocks: int) -> bool:
        n = max(0, int(num_blocks))
        # Decode step budget starts after step counter is initialized.
        if self.current_step <= 0:
            return True
        if kind == 'offload':
            if self._offload_used_blocks + n > self.offload_budget_blocks:
                return False
            self._offload_used_blocks += n
            return True
        if kind == 'prefetch':
            if self._prefetch_used_blocks + n > self.prefetch_budget_blocks:
                return False
            self._prefetch_used_blocks += n
            return True
        return True

    def _mark_resident(self, seq_id: int):
        with self.lock:
            self.seq_resident_until[seq_id] = self.current_step + self.min_resident_steps

    def _is_residency_protected(self, seq_id: int) -> bool:
        with self.lock:
            until = self.seq_resident_until.get(seq_id, 0)
        return self.current_step < until

    @staticmethod
    def _normalize_protected(protected_seqs: Optional[List[int]]) -> Set[int]:
        if not protected_seqs:
            return set()
        return set(int(s) for s in protected_seqs)

    def _get_seq_layer0_entry(self, seq_id: int) -> Optional[BlockTableEntry]:
        with self.lock:
            return self.page_table.get((seq_id, 0))

    def _evict_until_free(
        self,
        target_free: int,
        exclude_seq: int,
        protected_seqs: Optional[List[int]] = None,
    ) -> bool:
        target = max(0, int(target_free))
        while self.pool.n_free < target:
            before = int(self.pool.n_free)
            try:
                try:
                    victim = self.evict_coldest_sequence(
                        exclude_seq,
                        protected_seqs=protected_seqs,
                    )
                except TypeError:
                    # Backward compatibility for tests that monkeypatch the old
                    # single-argument evict signature.
                    victim = self.evict_coldest_sequence(exclude_seq)
            except MemoryError:
                return False

            try:
                self._try_finalize_offload(int(victim), force_sync=True)
            except Exception:
                return False

            after = int(self.pool.n_free)
            if after <= before and after < target:
                self._inc_stat('evict_fail')
                return False
        return True

    def _sequence_is_on_gpu(self, seq_id: int) -> bool:
        with self.lock:
            entries = [e for (s, _), e in self.page_table.items() if s == seq_id]
        if not entries:
            return False
        for entry in entries:
            if entry.state != OffloadState.ON_GPU or not entry.block_ids:
                return False
        return True

    def _sequence_has_state(self, seq_id: int, state: OffloadState) -> bool:
        with self.lock:
            for (s, _), entry in self.page_table.items():
                if s == seq_id and entry.state == state:
                    return True
        return False

    def get_sequence_residency(self, seq_id: int) -> Dict[str, int]:
        with self.lock:
            entries = [e for (s, _), e in self.page_table.items() if s == seq_id]
        if not entries:
            return {
                'exists': 0,
                'on_gpu': 0,
                'resident_blocks': 0,
                'logical_blocks': 0,
                'materialized_blocks': 0,
                'resident_ratio_x1000': 0,
                'has_cpu_blocks': 0,
                'state_mixed': 0,
                'last_access': -1,
                'residency_protected': 0,
            }
        resident_blocks = 0
        logical_blocks = 0
        materialized_blocks = 0
        on_gpu = 0
        has_cpu_blocks = 0
        state_mixed = 0
        last_access = -1
        for entry in entries:
            self._ensure_entry_maps(entry)
            resident_blocks = max(resident_blocks, self._resident_block_count(entry))
            materialized_blocks = max(materialized_blocks, int(getattr(entry, 'materialized_blocks', 0) or len(entry.gpu_block_map)))
            if self.pool.B > 0:
                logical_blocks = max(
                    logical_blocks,
                    int(math.ceil(float(int(getattr(entry, 'logical_seq_len', 0) or getattr(entry, 'seq_len', 0) or 0)) / float(self.pool.B))),
                )
            if entry.state == OffloadState.ON_GPU:
                on_gpu = 1
            if entry.state == OffloadState.MIXED:
                state_mixed = 1
            if any(x is not None for x in getattr(entry, 'cpu_k_blocks', []) or []):
                has_cpu_blocks = 1
            if getattr(entry, 'cpu_k', None) is not None or getattr(entry, 'cpu_v', None) is not None:
                has_cpu_blocks = 1
            last_access = max(last_access, int(getattr(entry, 'last_access', -1)))
        resident_ratio_x1000 = int((1000 * resident_blocks) / max(1, materialized_blocks)) if materialized_blocks > 0 else 0
        return {
            'exists': 1,
            'on_gpu': int(on_gpu),
            'resident_blocks': int(resident_blocks),
            'logical_blocks': int(logical_blocks),
            'materialized_blocks': int(materialized_blocks),
            'resident_ratio_x1000': int(resident_ratio_x1000),
            'has_cpu_blocks': int(has_cpu_blocks),
            'state_mixed': int(state_mixed),
            'last_access': int(last_access),
            'residency_protected': int(self._is_residency_protected(seq_id)),
        }

    def _try_finalize_offload(self, seq_id: int, force_sync: bool = False) -> bool:
        with self.lock:
            inflight = [
                e for (s, _), e in self.page_table.items()
                if s == seq_id and e.state == OffloadState.OFFLOAD_INFLIGHT
            ]
        if not inflight:
            return True

        event = inflight[0].transfer_event
        if event is not None:
            if force_sync:
                event.synchronize()
            else:
                try:
                    ready = bool(event.query())
                except Exception:
                    ready = False
                if not ready:
                    return False

        freed_ids = set()
        with self.lock:
            for entry in inflight:
                self._ensure_entry_maps(entry)
                old_map = list(entry.gpu_block_map)
                new_map = list(entry.pending_gpu_block_map or old_map)
                for old_bid, new_bid in zip(old_map, new_map):
                    if int(old_bid) >= 0 and int(new_bid) < 0:
                        freed_ids.add(int(old_bid))
                entry.gpu_block_map = list(new_map)
                entry.pending_gpu_block_map = None
                entry.pending_block_indices = None
                entry.pending_block_ids = None
                entry.transfer_event = None
                entry.last_access = int(self.current_step)
                self._update_entry_state(entry)
        if freed_ids:
            self.pool.free_blocks_by_ids(sorted(freed_ids))
        torch.cuda.empty_cache()
        self._inc_stat('offload_success')
        return True

    def _try_finalize_prefetch(self, seq_id: int, force_sync: bool = False) -> bool:
        with self.lock:
            inflight = [
                e for (s, _), e in self.page_table.items()
                if s == seq_id and e.state == OffloadState.PREFETCH_INFLIGHT
            ]
        if not inflight:
            return self._sequence_is_on_gpu(seq_id)

        event = inflight[0].transfer_event
        if event is not None:
            if force_sync:
                event.synchronize()
            else:
                try:
                    ready = bool(event.query())
                except Exception:
                    ready = False
                if not ready:
                    return False

        bytes_freed = 0
        with self.lock:
            for entry in inflight:
                self._ensure_entry_maps(entry)
                pending = list(entry.pending_gpu_block_map or entry.gpu_block_map)
                pending_indices = list(entry.pending_block_indices or [])
                for idx in pending_indices:
                    if 0 <= int(idx) < len(entry.cpu_k_blocks):
                        bytes_freed += self._bytes_for_block_indices(entry, [int(idx)])
                        entry.cpu_k_blocks[int(idx)] = None
                        entry.cpu_v_blocks[int(idx)] = None
                entry.gpu_block_map = pending
                entry.pending_gpu_block_map = None
                entry.pending_block_indices = None
                entry.pending_block_ids = None
                entry.transfer_event = None
                entry.cpu_k = None
                entry.cpu_v = None
                entry.last_access = int(self.current_step)
                self._update_entry_state(entry)
        self.pool.cpu_used_bytes = max(0, self.pool.cpu_used_bytes - bytes_freed)
        self._mark_resident(seq_id)
        self._inc_stat('prefetch_success')
        return True

    def register(self, seq_id: int, layer_id: int, K: torch.Tensor, V: torch.Tensor) -> List[int]:
        seq_len = K.shape[1]
        n_blocks = math.ceil(seq_len / self.pool.B)
        block_ids = self.pool.allocate_blocks(n_blocks)
        if not block_ids:
            raise MemoryError(f"GPU blocks exhausted: seq={seq_id} layer={layer_id}")
        self.pool.write_kv_to_blocks(layer_id, block_ids, K, V)
        entry = BlockTableEntry(
            seq_id=seq_id,
            layer_id=layer_id,
            state=OffloadState.ON_GPU,
            block_ids=block_ids,
            seq_len=seq_len,
            logical_seq_len=seq_len,
            materialized_blocks=len(block_ids),
            gpu_block_map=list(block_ids),
            cpu_k_blocks=[None] * n_blocks,
            cpu_v_blocks=[None] * n_blocks,
        )
        with self.lock:
            self.page_table[(seq_id, layer_id)] = entry
        self._mark_resident(seq_id)
        return block_ids

    def register_all_layers(
        self,
        seq_id: int,
        block_ids: List[int],
        seq_len: int,
        K_list: list,
        V_list: list,
        anchor_k_means: Optional[list] = None,
        logical_seq_len: Optional[int] = None,
    ):
        logical_len = int(seq_len if logical_seq_len is None else logical_seq_len)
        for layer_id, (K, V) in enumerate(zip(K_list, V_list)):
            if K is None:
                continue
            anchor_k = None
            if anchor_k_means is not None and layer_id < len(anchor_k_means):
                anchor_k = anchor_k_means[layer_id]
                if torch.is_tensor(anchor_k):
                    anchor_k = anchor_k.detach().cpu()
            self.pool.write_kv_to_blocks(layer_id, block_ids, K, V)
            entry = BlockTableEntry(
                seq_id=seq_id,
                layer_id=layer_id,
                state=OffloadState.ON_GPU,
                block_ids=list(block_ids),
                seq_len=seq_len,
                logical_seq_len=logical_len,
                materialized_blocks=len(block_ids),
                prefill_anchor_k_mean=anchor_k,
                gpu_block_map=list(block_ids),
                cpu_k_blocks=[None] * len(block_ids),
                cpu_v_blocks=[None] * len(block_ids),
            )
            with self.lock:
                self.page_table[(seq_id, layer_id)] = entry
        self._mark_resident(seq_id)

    def evict_coldest_sequence(self, exclude_seq: int, protected_seqs: Optional[List[int]] = None) -> int:
        self._inc_stat('evict_calls')
        protected = self._normalize_protected(protected_seqs)
        protected.add(exclude_seq)
        seq_max_access = {}
        with self.lock:
            for (s, _), e in self.page_table.items():
                if s in protected or e.state != OffloadState.ON_GPU:
                    continue
                seq_max_access[s] = max(seq_max_access.get(s, 0), e.last_access)

        if not seq_max_access:
            self._inc_stat('evict_fail')
            raise MemoryError("No evictable sequence on GPU")

        ordered = sorted(seq_max_access, key=seq_max_access.get)

        for target in ordered:
            if self._is_residency_protected(target):
                continue
            if self.offload_sequence(target):
                self._inc_stat('evict_success')
                return target

        # Fallback under hard memory pressure: relax residency guard only.
        for target in ordered:
            if target in protected:
                continue
            if self.offload_sequence(target):
                self._inc_stat('evict_success')
                return target

        self._inc_stat('evict_fail')
        raise MemoryError("Eviction attempted but no sequence could be offloaded")

    def offload_sequence_blocks(self, seq_id: int, logical_block_indices: List[int]) -> bool:
        self._inc_stat('offload_calls')
        self._try_finalize_offload(seq_id, force_sync=False)
        self._try_finalize_prefetch(seq_id, force_sync=False)
        with self.lock:
            entries = {
                l: e
                for (s, l), e in self.page_table.items()
                if s == seq_id and e.state in (OffloadState.ON_GPU, OffloadState.MIXED)
            }
            for e in entries.values():
                self._ensure_entry_maps(e)
        if not entries:
            self._inc_stat('offload_fail')
            return False

        first = next(iter(entries.values()))
        selected = sorted({int(i) for i in logical_block_indices if 0 <= int(i) < len(first.gpu_block_map) and int(first.gpu_block_map[int(i)]) >= 0})
        if not selected:
            self._inc_stat('offload_fail')
            return False
        if not self._try_consume_budget('offload', len(selected)):
            self._inc_stat('offload_fail')
            return False

        bytes_needed = self._bytes_for_block_indices(first, selected) * len(entries)
        if self.pool.cpu_used_bytes + bytes_needed > self.pool.cpu_budget:
            self._inc_stat('offload_fail')
            return False

        event = torch.cuda.Event()
        with torch.cuda.stream(self.transfer_stream):
            for layer_id, entry in sorted(entries.items()):
                pending_map = list(entry.gpu_block_map)
                for idx in selected:
                    block_id = int(entry.gpu_block_map[idx])
                    if block_id < 0:
                        continue
                    K_g, V_g = self._read_single_block_from_gpu(layer_id, block_id, idx, entry.seq_len)
                    K_c = torch.empty_like(K_g, device='cpu', pin_memory=True)
                    V_c = torch.empty_like(V_g, device='cpu', pin_memory=True)
                    K_c.copy_(K_g, non_blocking=True)
                    V_c.copy_(V_g, non_blocking=True)
                    entry.cpu_k_blocks[idx] = K_c
                    entry.cpu_v_blocks[idx] = V_c
                    pending_map[idx] = -1
                entry.pending_gpu_block_map = pending_map
                entry.pending_block_indices = list(selected)
                entry.transfer_event = event
                entry.state = OffloadState.OFFLOAD_INFLIGHT
            event.record(self.transfer_stream)
        self.pool.cpu_used_bytes += bytes_needed
        return True

    def offload_sequence(self, seq_id: int) -> bool:
        with self.lock:
            entries = [e for (s, _), e in self.page_table.items() if s == seq_id]
            for entry in entries:
                self._ensure_entry_maps(entry)
        if not entries:
            self._inc_stat('offload_fail')
            return False
        logical_blocks = max((len(e.gpu_block_map) for e in entries), default=0)
        if logical_blocks <= 0:
            self._inc_stat('offload_fail')
            return False
        ok = self.offload_sequence_blocks(seq_id, list(range(logical_blocks)))
        if ok:
            return True
        if self._sequence_has_state(seq_id, OffloadState.OFFLOAD_INFLIGHT):
            return True
        return bool(self._try_finalize_offload(seq_id, force_sync=False))

    def prefetch_sequence_blocks(self, seq_id: int, logical_block_indices: List[int]) -> bool:
        self._inc_stat('prefetch_calls')
        if self._try_finalize_prefetch(seq_id, force_sync=False):
            return True
        if self._sequence_has_state(seq_id, OffloadState.PREFETCH_INFLIGHT):
            self._inc_stat('prefetch_inflight')
            return False

        with self.lock:
            entries = {
                l: e
                for (s, l), e in self.page_table.items()
                if s == seq_id and e.state in (OffloadState.ON_CPU, OffloadState.MIXED)
            }
            for e in entries.values():
                self._ensure_entry_maps(e)
        if not entries:
            self._inc_stat('prefetch_noop')
            self._inc_stat('prefetch_success')
            return True

        first = next(iter(entries.values()))
        selected = sorted({
            int(i) for i in logical_block_indices
            if 0 <= int(i) < len(first.gpu_block_map)
            and int(first.gpu_block_map[int(i)]) < 0
            and getattr(first, 'cpu_k_blocks', [None])[int(i)] is not None
        })
        if not selected:
            self._inc_stat('prefetch_noop')
            self._inc_stat('prefetch_success')
            return True
        if not self._try_consume_budget('prefetch', len(selected)):
            self._inc_stat('prefetch_fail')
            return False
        block_ids = self.pool.allocate_blocks(len(selected))
        if len(block_ids) != len(selected):
            if block_ids:
                self.pool.free_blocks_by_ids(block_ids)
            self._inc_stat('prefetch_fail')
            return False

        event = torch.cuda.Event()
        try:
            with torch.cuda.stream(self.transfer_stream):
                for layer_id, entry in sorted(entries.items()):
                    pending_map = list(entry.gpu_block_map)
                    for bid, idx in zip(block_ids, selected):
                        K_c = entry.cpu_k_blocks[idx]
                        V_c = entry.cpu_v_blocks[idx]
                        if K_c is None or V_c is None:
                            continue
                        K_g = K_c.to('cuda', non_blocking=True)
                        V_g = V_c.to('cuda', non_blocking=True)
                        self._write_single_block_to_gpu(layer_id, int(bid), idx, entry.seq_len, K_g, V_g)
                        pending_map[idx] = int(bid)
                    entry.pending_gpu_block_map = pending_map
                    entry.pending_block_indices = list(selected)
                    entry.transfer_event = event
                    entry.state = OffloadState.PREFETCH_INFLIGHT
                event.record(self.transfer_stream)
        except Exception:
            self.pool.free_blocks_by_ids(block_ids)
            with self.lock:
                for entry in entries.values():
                    self._update_entry_state(entry)
                    entry.pending_gpu_block_map = None
                    entry.pending_block_indices = None
                    entry.transfer_event = None
            self._inc_stat('prefetch_fail')
            return False
        return False

    def prefetch_sequence(self, seq_id: int) -> bool:
        with self.lock:
            entries = [e for (s, _), e in self.page_table.items() if s == seq_id]
            for entry in entries:
                self._ensure_entry_maps(entry)
        if not entries:
            self._inc_stat('prefetch_noop')
            self._inc_stat('prefetch_success')
            return True
        logical_blocks = max((len(e.gpu_block_map) for e in entries), default=0)
        ok = self.prefetch_sequence_blocks(seq_id, list(range(logical_blocks)))
        if not ok:
            ok = self._try_finalize_prefetch(seq_id, force_sync=True)
        return bool(ok)

    def ensure_sequence_blocks_on_gpu(
        self,
        seq_id: int,
        missing_block_indices: List[int],
        allow_evict: bool = True,
        protected_seqs: Optional[List[int]] = None,
    ) -> bool:
        self._inc_stat('ensure_calls')
        self._try_finalize_offload(seq_id, force_sync=False)
        if self._try_finalize_prefetch(seq_id, force_sync=False):
            with self.lock:
                entries = [e for (s, _), e in self.page_table.items() if s == seq_id]
                for entry in entries:
                    self._ensure_entry_maps(entry)
                if entries:
                    need = [idx for idx in missing_block_indices if idx < len(entries[0].gpu_block_map) and int(entries[0].gpu_block_map[idx]) < 0]
                    if not need:
                        self._inc_stat('ensure_success')
                        return True

        with self.lock:
            entries = [e for (s, _), e in self.page_table.items() if s == seq_id]
            for entry in entries:
                self._ensure_entry_maps(entry)
        if not entries:
            self._inc_stat('ensure_fail')
            return False
        first = entries[0]
        needed = sorted({int(i) for i in missing_block_indices if 0 <= int(i) < len(first.gpu_block_map) and int(first.gpu_block_map[int(i)]) < 0})
        if not needed:
            self._inc_stat('ensure_success')
            return True

        if allow_evict:
            # Prefetch only needs enough blocks for the immediate missing span.
            # Restoring the pool to wm_high here can recursively evict many
            # ready sequences and stall decode under P2 pressure.
            target_free = max(1, len(needed))
            ok_evict = self._evict_until_free(target_free=target_free, exclude_seq=seq_id, protected_seqs=protected_seqs)
            if not ok_evict:
                self._inc_stat('ensure_fail')
                return False

        self.prefetch_sequence_blocks(seq_id, needed)
        final_ok = self._try_finalize_prefetch(seq_id, force_sync=True)
        if final_ok:
            self._inc_stat('ensure_success')
        else:
            self._inc_stat('ensure_fail')
        return bool(final_ok)

    def ensure_sequence_on_gpu(
        self,
        seq_id: int,
        allow_evict: bool = True,
        protected_seqs: Optional[List[int]] = None,
    ) -> bool:
        with self.lock:
            entries = [e for (s, _), e in self.page_table.items() if s == seq_id]
            for entry in entries:
                self._ensure_entry_maps(entry)
        if not entries:
            self._inc_stat('ensure_fail')
            return False
        logical_blocks = max((len(e.gpu_block_map) for e in entries), default=0)
        return self.ensure_sequence_blocks_on_gpu(
            seq_id,
            list(range(logical_blocks)),
            allow_evict=allow_evict,
            protected_seqs=protected_seqs,
        )

    def reserve_decode_slot(self, seq_id: int) -> DecodeAppendResult:
        """Reserve resident GPU storage for one direct-decode token.

        This prepares the paged KV cache for flash_attn_with_kvcache, which writes
        the new token in-place during the model forward. It intentionally does not
        prefetch or fall back to materialization; Stage-1 paged-direct validation is
        strict and only accepts fully resident sequences.
        """
        entries = [self.page_table.get((seq_id, lid)) for lid in range(self.num_layers)]
        if any(e is None for e in entries):
            self._inc_stat('decode_append_fail')
            return DecodeAppendResult(
                ok=False,
                retryable=False,
                reason=f"missing_page_table_entry seq={seq_id}",
            )

        for entry in entries:
            self._ensure_entry_maps(entry)
            if entry.state != OffloadState.ON_GPU:
                self._inc_stat('decode_append_fail')
                self._inc_stat('decode_append_retryable')
                return DecodeAppendResult(
                    ok=False,
                    retryable=True,
                    reason=f"direct_decode_not_on_gpu seq={seq_id} state={entry.state}",
                )
            if any(int(bid) < 0 for bid in entry.gpu_block_map):
                self._inc_stat('decode_append_fail')
                self._inc_stat('decode_append_retryable')
                return DecodeAppendResult(
                    ok=False,
                    retryable=True,
                    reason=f"direct_decode_resident_missing seq={seq_id}",
                )
            if not entry.block_ids:
                self._inc_stat('decode_append_fail')
                return DecodeAppendResult(
                    ok=False,
                    retryable=True,
                    reason=f"direct_decode_no_gpu_blocks seq={seq_id}",
                )

        seq_len = int(entries[0].seq_len)
        if any(int(e.seq_len) != seq_len for e in entries):
            self._inc_stat('decode_append_fail')
            return DecodeAppendResult(
                ok=False,
                retryable=False,
                reason=f"direct_decode_inconsistent_seq_len seq={seq_id}",
            )

        if seq_len % self.pool.B == 0:
            new_ids = self.pool.allocate_blocks(1)
            if not new_ids:
                ok_evict = self._evict_until_free(
                    target_free=max(self.pool.N_wm_high, 1),
                    exclude_seq=seq_id,
                    protected_seqs=[seq_id],
                )
                if not ok_evict:
                    self._inc_stat('decode_append_fail')
                    self._inc_stat('decode_append_retryable')
                    return DecodeAppendResult(
                        ok=False,
                        retryable=True,
                        reason="direct_decode_expand_evict_failed",
                    )
                new_ids = self.pool.allocate_blocks(1)
                if not new_ids:
                    self._inc_stat('decode_append_fail')
                    self._inc_stat('decode_append_retryable')
                    return DecodeAppendResult(
                        ok=False,
                        retryable=True,
                        reason="direct_decode_expand_alloc_failed",
                    )
            for entry in entries:
                self._ensure_entry_maps(entry)
                entry.block_ids.extend(new_ids)
                entry.gpu_block_map.append(int(new_ids[0]))
                entry.cpu_k_blocks.append(None)
                entry.cpu_v_blocks.append(None)
                entry.materialized_blocks = len(entry.gpu_block_map)
                self._update_entry_state(entry)

        return DecodeAppendResult(ok=True)

    def append_decode_token(self, seq_id: int, layer_id: int, k_tok: torch.Tensor, v_tok: torch.Tensor) -> DecodeAppendResult:
        self._inc_stat('decode_append_calls')
        key = (seq_id, layer_id)
        entry = self.page_table.get(key)
        if entry is None:
            self._inc_stat('decode_append_fail')
            return DecodeAppendResult(
                ok=False,
                retryable=False,
                reason=f"missing_page_table_entry seq={seq_id} layer={layer_id}",
            )

        self._ensure_entry_maps(entry)
        if entry.state != OffloadState.ON_GPU or any(int(bid) < 0 for bid in entry.gpu_block_map):
            ok = self.ensure_sequence_on_gpu(seq_id, allow_evict=True, protected_seqs=[seq_id])
            if not ok:
                self._inc_stat('decode_append_fail')
                self._inc_stat('decode_append_retryable')
                reason = f"prefetch_before_decode_append_failed seq={seq_id}"
                if self._sequence_has_state(seq_id, OffloadState.PREFETCH_INFLIGHT):
                    reason = f"prefetch_inflight seq={seq_id}"
                elif self._sequence_has_state(seq_id, OffloadState.OFFLOAD_INFLIGHT):
                    reason = f"offload_inflight seq={seq_id}"
                return DecodeAppendResult(
                    ok=False,
                    retryable=True,
                    reason=reason,
                )
            entry = self.page_table.get(key)
            if entry is None:
                self._inc_stat('decode_append_fail')
                return DecodeAppendResult(
                    ok=False,
                    retryable=False,
                    reason=f"missing_page_table_after_prefetch seq={seq_id} layer={layer_id}",
                )

        self._ensure_entry_maps(entry)
        if not entry.block_ids:
            self._inc_stat('decode_append_fail')
            return DecodeAppendResult(
                ok=False,
                retryable=True,
                reason=f"no_gpu_blocks_for_decode_append seq={seq_id} layer={layer_id}",
            )

        last_used = entry.seq_len % self.pool.B
        if last_used == 0 and layer_id == 0:
            new_ids = self.pool.allocate_blocks(1)
            if not new_ids:
                ok_evict = self._evict_until_free(
                    target_free=max(self.pool.N_wm_high, 1),
                    exclude_seq=seq_id,
                    protected_seqs=[seq_id],
                )
                if not ok_evict:
                    self._inc_stat('decode_append_fail')
                    self._inc_stat('decode_append_retryable')
                    return DecodeAppendResult(
                        ok=False,
                        retryable=True,
                        reason="decode_expand_evict_failed",
                    )
                new_ids = self.pool.allocate_blocks(1)
                if not new_ids:
                    self._inc_stat('decode_append_fail')
                    self._inc_stat('decode_append_retryable')
                    return DecodeAppendResult(
                        ok=False,
                        retryable=True,
                        reason="decode_expand_alloc_failed",
                    )
            for lid in range(self.num_layers):
                e = self.page_table.get((seq_id, lid))
                if e:
                    self._ensure_entry_maps(e)
                    e.block_ids.extend(new_ids)
                    e.gpu_block_map.append(int(new_ids[0]))
                    e.cpu_k_blocks.append(None)
                    e.cpu_v_blocks.append(None)
                    e.materialized_blocks = len(e.gpu_block_map)
                    self._update_entry_state(e)

        self._ensure_entry_maps(entry)
        last_block_id = entry.block_ids[-1]
        self.pool.write_one_token(layer_id, last_block_id, last_used, k_tok, v_tok)
        if int(getattr(entry, 'logical_seq_len', 0)) <= 0:
            entry.logical_seq_len = int(entry.seq_len)
        entry.seq_len += 1
        entry.logical_seq_len += 1
        entry.materialized_blocks = max(int(getattr(entry, 'materialized_blocks', 0) or 0), len(entry.gpu_block_map))
        self._inc_stat('decode_append_success')
        return DecodeAppendResult(ok=True)

    def get_sequence_prune_state(self, seq_id: int) -> Dict[str, object]:
        e = self._get_seq_layer0_entry(seq_id)
        if e is None:
            return {
                'mid_base_blocks': 0,
                'mid_deleted_cum': 0,
                'prune_frozen': 0,
                'freeze_reason': '',
            }
        return {
            'mid_base_blocks': int(e.mid_base_blocks),
            'mid_deleted_cum': int(e.mid_deleted_cum),
            'prune_frozen': int(bool(e.prune_frozen)),
            'freeze_reason': str(e.freeze_reason),
        }

    def count_prune_frozen_sequences(self, seq_ids: Optional[List[int]] = None) -> int:
        with self.lock:
            target = None if seq_ids is None else set(int(s) for s in seq_ids)
            frozen: Set[int] = set()
            for (sid, lid), entry in self.page_table.items():
                if lid != 0:
                    continue
                if target is not None and sid not in target:
                    continue
                if bool(entry.prune_frozen):
                    frozen.add(int(sid))
        return int(len(frozen))

    def unfreeze_sequence_prune(self, seq_id: int, reset_budget: bool = True) -> bool:
        with self.lock:
            entries = [e for (s, _), e in self.page_table.items() if s == seq_id]
            if not entries:
                return False
            changed = False
            for e in entries:
                if e.prune_frozen:
                    changed = True
                e.prune_frozen = False
                e.freeze_reason = ""
                if reset_budget:
                    # Re-arm per-sequence prune budget from current state.
                    e.mid_base_blocks = 0
                    e.mid_deleted_cum = 0
        if changed:
            self._inc_stat('window_prune_unfreeze')
        return changed

    def _block_token_span(self, block_idx: int, total_len: int) -> Tuple[int, int]:
        start = int(block_idx) * self.pool.B
        end = min(int(total_len), (int(block_idx) + 1) * self.pool.B)
        return start, end

    def _blocks_overlapping_range(self, total_blocks: int, total_len: int, start: int, end: int) -> Set[int]:
        if end <= start:
            return set()
        out: Set[int] = set()
        for bi in range(total_blocks):
            bs, be = self._block_token_span(bi, total_len)
            if be > start and bs < end:
                out.add(bi)
        return out

    def _score_candidate_blocks(
        self,
        K_ref: torch.Tensor,
        candidate_blocks: List[int],
        anchor_vec: Optional[torch.Tensor],
        recent_range: Tuple[int, int],
        recent_weight: float,
    ) -> Dict[int, float]:
        if not candidate_blocks:
            return {}
        # Collapse [kv_heads, tokens, head_dim] -> [head_dim]
        def _vec_for_range(lo: int, hi: int) -> Optional[torch.Tensor]:
            if hi <= lo:
                return None
            seg = K_ref[:, lo:hi, :]
            if seg.numel() == 0:
                return None
            return seg.float().mean(dim=(0, 1))

        if anchor_vec is not None and torch.is_tensor(anchor_vec):
            anchor_vec = anchor_vec.float().to(K_ref.device)
        recent_vec = _vec_for_range(recent_range[0], recent_range[1])
        if anchor_vec is None and recent_vec is None:
            return {int(b): 0.0 for b in candidate_blocks}

        rw = float(max(0.0, min(1.0, recent_weight)))
        aw = 1.0 - rw
        scores: Dict[int, float] = {}
        for bi in candidate_blocks:
            bs, be = self._block_token_span(bi, K_ref.shape[1])
            cand_vec = _vec_for_range(bs, be)
            if cand_vec is None:
                scores[int(bi)] = 0.0
                continue
            s_recent = 0.0
            s_anchor = 0.0
            if recent_vec is not None:
                s_recent = float(F.cosine_similarity(cand_vec, recent_vec, dim=0).item())
            if anchor_vec is not None:
                s_anchor = float(F.cosine_similarity(cand_vec, anchor_vec, dim=0).item())
            if recent_vec is None:
                score = s_anchor
            elif anchor_vec is None:
                score = s_recent
            else:
                score = rw * s_recent + aw * s_anchor
            scores[int(bi)] = score
        return scores

    def prune_sequence_window(
        self,
        seq_id: int,
        sink_len: int,
        recent_len: int,
        min_drop_tokens: int = 1,
        anchor_tokens: int = 16,
        recent_score_weight: float = 0.60,
        target_keep_ratio: float = 0.50,
        delete_cap_ratio: float = 0.50,
        min_keep_ratio: float = 0.50,
        min_resident_ratio_total: float = 0.0,
        protected_seqs: Optional[List[int]] = None,
    ) -> int:
        """Keep protected blocks and prune middle blocks with budget/floor guards."""
        self._inc_stat('window_prune_calls')

        with self.lock:
            entries = {
                l: e
                for (s, l), e in self.page_table.items()
                if s == seq_id
            }

        if not entries:
            self._mark_window_prune_skip()
            return 0

        if any(e.state != OffloadState.ON_GPU for e in entries.values()):
            ok = self.ensure_sequence_on_gpu(
                seq_id=seq_id,
                allow_evict=True,
                protected_seqs=protected_seqs,
            )
            if not ok:
                self._inc_stat('window_prune_fail')
                return 0
            with self.lock:
                entries = {
                    l: e
                    for (s, l), e in self.page_table.items()
                    if s == seq_id and e.state == OffloadState.ON_GPU
                }
            if not entries:
                self._inc_stat('window_prune_fail')
                return 0

        first = next(iter(entries.values()))
        old_len = int(first.seq_len)
        old_logical_len = int(getattr(first, 'logical_seq_len', 0) or old_len)
        old_block_ids = list(first.block_ids)
        if old_len <= 0 or not old_block_ids:
            self._mark_window_prune_skip(reason='no_candidate_blocks')
            return 0

        total_blocks = len(old_block_ids)
        sink = max(0, min(int(sink_len), old_len))
        recent_cap = max(0, old_len - sink)
        recent = max(0, min(int(recent_len), recent_cap))
        _ = int(anchor_tokens)

        sink_blocks = self._blocks_overlapping_range(total_blocks, old_len, 0, sink)
        recent_blocks = self._blocks_overlapping_range(total_blocks, old_len, old_len - recent, old_len)
        protected_blocks = sink_blocks | recent_blocks
        candidate_blocks = [bi for bi in range(total_blocks) if bi not in protected_blocks]
        if not candidate_blocks:
            self._mark_window_prune_skip(reason='no_candidate_blocks')
            return 0

        with self.lock:
            base_entry = self.page_table.get((seq_id, 0))
            if base_entry is None:
                self._mark_window_prune_skip()
                return 0
            if base_entry.prune_frozen:
                self._inc_stat('window_prune_frozen')
                self._mark_window_prune_skip()
                return 0
            if base_entry.mid_base_blocks <= 0:
                base_entry.mid_base_blocks = int(len(candidate_blocks))
            mid_base = int(base_entry.mid_base_blocks)
            mid_deleted_cum = int(base_entry.mid_deleted_cum)
            prefill_anchor_vec = base_entry.prefill_anchor_k_mean

        mid_alive = int(len(candidate_blocks))
        if mid_base <= 0 or mid_alive <= 0:
            self._mark_window_prune_skip(reason='no_candidate_blocks')
            return 0

        keep_ratio = float(max(0.0, min(1.0, target_keep_ratio)))
        cap_ratio = float(max(0.0, min(1.0, delete_cap_ratio)))
        floor_ratio = float(max(0.0, min(1.0, min_keep_ratio)))

        target_keep_blocks = max(1, int(math.ceil(mid_alive * keep_ratio)))
        target_delete = max(0, mid_alive - target_keep_blocks)

        cap_limit = int(math.floor(mid_base * cap_ratio))
        cap_left = max(0, cap_limit - mid_deleted_cum)
        floor_abs = int(math.ceil(mid_base * floor_ratio))
        floor_left = max(0, mid_alive - floor_abs)
        allow_delete = min(target_delete, cap_left, floor_left)
        if allow_delete <= 0:
            with self.lock:
                for _, e in entries.items():
                    e.prune_frozen = True
                    e.freeze_reason = 'budget_or_floor'
            self._inc_stat('window_prune_frozen')
            self._mark_window_prune_skip()
            return 0

        keep_candidate_blocks = max(1, mid_alive - allow_delete)

        # Use layer-0 K as block salience proxy; apply the selected block set to all layers.
        K_ref, _ = self.pool.read_kv_from_blocks(0, old_block_ids, old_len)
        scores = self._score_candidate_blocks(
            K_ref=K_ref,
            candidate_blocks=candidate_blocks,
            anchor_vec=prefill_anchor_vec,
            recent_range=(old_len - recent, old_len),
            recent_weight=recent_score_weight,
        )
        sorted_candidates = sorted(candidate_blocks, key=lambda bi: scores.get(int(bi), 0.0), reverse=True)
        kept_candidates = set(sorted_candidates[:keep_candidate_blocks])
        kept_blocks = sorted(set(protected_blocks) | kept_candidates)
        if not kept_blocks:
            self._mark_window_prune_skip(reason='no_candidate_blocks')
            return 0

        keep_len = 0
        for bi in kept_blocks:
            bs, be = self._block_token_span(bi, old_len)
            keep_len += max(0, be - bs)
        dropped = old_len - keep_len
        if dropped < max(1, int(min_drop_tokens)):
            self._mark_window_prune_skip(reason='min_drop_not_met')
            return 0

        n_new_blocks = math.ceil(max(1, keep_len) / self.pool.B)
        # Guard decode-time pruning against collapsing the *current resident cache*,
        # not the original logical prompt length. Prefill compression can reduce
        # resident blocks by an order of magnitude; using logical length here can
        # make the floor impossible to satisfy and effectively disable P3.
        materialized_total_blocks = max(0, int(total_blocks))
        min_total_blocks = 0
        if float(min_resident_ratio_total) > 0.0 and materialized_total_blocks > 0:
            min_total_blocks = max(
                1,
                int(math.ceil(float(materialized_total_blocks) * float(min_resident_ratio_total))),
            )
        if min_total_blocks > 0 and int(n_new_blocks) < int(min_total_blocks):
            self._mark_window_prune_skip(reason='min_drop_not_met')
            return 0
        if self.pool.n_free < n_new_blocks:
            ok_evict = self._evict_until_free(
                target_free=max(n_new_blocks, self.pool.N_wm_high),
                exclude_seq=seq_id,
                protected_seqs=protected_seqs,
            )
            if not ok_evict:
                self._inc_stat('window_prune_fail')
                return 0

        new_block_ids = self.pool.allocate_blocks(n_new_blocks)
        if not new_block_ids:
            self._inc_stat('window_prune_fail')
            return 0

        try:
            for layer_id in sorted(entries.keys()):
                K_full, V_full = self.pool.read_kv_from_blocks(layer_id, old_block_ids, old_len)
                K_parts = []
                V_parts = []
                for bi in kept_blocks:
                    bs, be = self._block_token_span(bi, old_len)
                    if be <= bs:
                        continue
                    K_parts.append(K_full[:, bs:be, :])
                    V_parts.append(V_full[:, bs:be, :])
                if not K_parts:
                    raise RuntimeError("No kept blocks while pruning sequence")
                K_keep = torch.cat(K_parts, dim=1)
                V_keep = torch.cat(V_parts, dim=1)
                self.pool.write_kv_to_blocks(layer_id, new_block_ids, K_keep, V_keep)
        except Exception:
            self.pool.free_blocks_by_ids(new_block_ids)
            self._inc_stat('window_prune_fail')
            return 0

        self.pool.free_blocks_by_ids(old_block_ids)

        with self.lock:
            for _, entry in entries.items():
                entry.block_ids = list(new_block_ids)
                entry.seq_len = keep_len
                entry.logical_seq_len = old_logical_len
                entry.materialized_blocks = len(new_block_ids)
                entry.state = OffloadState.ON_GPU
                entry.last_access = self.current_step
                entry.mid_deleted_cum = int(entry.mid_deleted_cum + max(0, int(allow_delete)))
                entry.mid_base_blocks = int(mid_base)
                entry.gpu_block_map = list(new_block_ids)
                entry.cpu_k_blocks = [None] * len(new_block_ids)
                entry.cpu_v_blocks = [None] * len(new_block_ids)
                entry.pending_gpu_block_map = None
                entry.pending_block_indices = None

        self._inc_stat('window_prune_success')
        self._inc_stat('window_prune_tokens_dropped', dropped)
        self._mark_resident(seq_id)
        return dropped

    def release_sequence(self, seq_id: int):
        keys_to_del = []
        freed_gpu_ids: Set[int] = set()
        cpu_freed = 0
        with self.lock:
            for key, entry in list(self.page_table.items()):
                if key[0] != seq_id:
                    continue
                self._ensure_entry_maps(entry)
                gpu_ids = [int(bid) for bid in entry.gpu_block_map if int(bid) >= 0]
                if entry.pending_gpu_block_map:
                    gpu_ids.extend(int(bid) for bid in entry.pending_gpu_block_map if int(bid) >= 0)
                for bid in gpu_ids:
                    freed_gpu_ids.add(int(bid))
                for idx, cpu_k in enumerate(getattr(entry, 'cpu_k_blocks', []) or []):
                    if cpu_k is not None:
                        cpu_freed += self._bytes_for_block_indices(entry, [idx])
                if getattr(entry, 'cpu_k', None) is not None or getattr(entry, 'cpu_v', None) is not None:
                    cpu_freed += 2 * self.pool.num_kv_heads * entry.seq_len * self.pool.head_dim * 2
                keys_to_del.append(key)
            for k in keys_to_del:
                del self.page_table[k]
            self.seq_resident_until.pop(seq_id, None)
        if freed_gpu_ids:
            self.pool.free_blocks_by_ids(sorted(freed_gpu_ids))
        if cpu_freed > 0:
            self.pool.cpu_used_bytes = max(0, self.pool.cpu_used_bytes - int(cpu_freed))
        torch.cuda.empty_cache()
