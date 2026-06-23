import math
import threading
import torch
from typing import List, Tuple, Dict, Optional


class PagedKVPool:
    """
    flash_attn物理分页KV存储池。
    
    内存布局：
      k_cache: [num_layers, N_total, B, h_kv, d]
      v_cache: 同上
    
    关键设计：
      - 动态计算可用显存：total - model_used - reserve(10%)
      - block_table用0作padding（flash_attn约定）
    """
    
    def __init__(
        self,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        gpu_mem_frac: float = 0.75,
        cpu_mem_gb: float = 48.0,
    ):
        self.B = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        
        total_gpu = torch.cuda.get_device_properties(0).total_memory
        model_used = torch.cuda.memory_allocated()
        reserve = int(0.10 * total_gpu)
        available = total_gpu - model_used - reserve
        
        # gpu_mem_frac 是相对于"可用显存"的比例
        max_kv_mem = int(available * gpu_mem_frac)
        
        bytes_per_block = 2 * num_layers * block_size * num_kv_heads * head_dim * 2
        self.N_total = max(1, int(max_kv_mem / bytes_per_block))
        
        self._k_cache: Optional[torch.Tensor] = None
        self._v_cache: Optional[torch.Tensor] = None
        
        self.lock = threading.Lock()
        self.free_blocks: List[int] = list(range(self.N_total))
        self.used_blocks: Dict[int, int] = {}
        
        self.cpu_budget = int(cpu_mem_gb * 1024**3)
        self.cpu_used_bytes = 0
        
        self.N_wm_low = max(1, int(0.15 * self.N_total))
        self.N_wm_high = max(2, int(0.40 * self.N_total))

    @property
    def is_allocated(self) -> bool:
        return self._k_cache is not None and self._v_cache is not None

    def ensure_allocated(self) -> None:
        if self.is_allocated:
            return
        self._k_cache = torch.zeros(
            self.num_layers, self.N_total, self.B, self.num_kv_heads, self.head_dim,
            dtype=self.dtype, device='cuda'
        )
        self._v_cache = torch.zeros_like(self._k_cache)

    @property
    def k_cache(self) -> torch.Tensor:
        self.ensure_allocated()
        assert self._k_cache is not None
        return self._k_cache

    @property
    def v_cache(self) -> torch.Tensor:
        self.ensure_allocated()
        assert self._v_cache is not None
        return self._v_cache

    def release_if_empty(self) -> bool:
        with self.lock:
            empty = len(self.used_blocks) == 0
        if not empty or not self.is_allocated:
            return False
        self._k_cache = None
        self._v_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    
    @property
    def n_free(self) -> int:
        with self.lock:
            return len(self.free_blocks)
    
    def allocate_blocks(self, n: int) -> List[int]:
        with self.lock:
            if len(self.free_blocks) < n:
                return []
            ids = self.free_blocks[:n]
            self.free_blocks = self.free_blocks[n:]
            for bid in ids:
                self.used_blocks[bid] = 1
            return ids
    
    def free_blocks_by_ids(self, block_ids: List[int]):
        with self.lock:
            for bid in block_ids:
                if bid in self.used_blocks:
                    del self.used_blocks[bid]
                self.free_blocks.append(bid)
    
    def check_watermark(self, n_required: int) -> str:
        with self.lock:
            n = len(self.free_blocks)
        if n < n_required:
            return "VRAM_FULL"
        if n < self.N_wm_high:
            return "WARNING"
        return "OK"
    
    def write_kv_to_blocks(
        self,
        layer_id: int,
        block_ids: List[int],
        K: torch.Tensor,
        V: torch.Tensor,
    ):
        seq_len = K.shape[1]
        for i, bid in enumerate(block_ids):
            start = i * self.B
            end = min(start + self.B, seq_len)
            chunk = end - start
            if chunk > 0:
                self.k_cache[layer_id, bid, :chunk] = K[:, start:end].transpose(0, 1)
                self.v_cache[layer_id, bid, :chunk] = V[:, start:end].transpose(0, 1)
    
    def write_one_token(
        self,
        layer_id: int,
        block_id: int,
        pos_in_block: int,
        k_tok: torch.Tensor,
        v_tok: torch.Tensor,
    ):
        self.k_cache[layer_id, block_id, pos_in_block] = k_tok.squeeze(1)
        self.v_cache[layer_id, block_id, pos_in_block] = v_tok.squeeze(1)
    
    def read_kv_from_blocks(
        self,
        layer_id: int,
        block_ids: List[int],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        K_chunks, V_chunks = [], []
        for i, bid in enumerate(block_ids):
            start = i * self.B
            end = min(start + self.B, seq_len)
            chunk = end - start
            if chunk > 0:
                K_chunks.append(self.k_cache[layer_id, bid, :chunk].transpose(0, 1))
                V_chunks.append(self.v_cache[layer_id, bid, :chunk].transpose(0, 1))
        K = torch.cat(K_chunks, dim=1) if K_chunks else torch.empty(
            self.num_kv_heads, 0, self.head_dim, dtype=self.dtype, device='cuda'
        )
        V = torch.cat(V_chunks, dim=1) if V_chunks else torch.empty_like(K)
        return K, V
    
    def build_block_table(
        self,
        layer_id: int,
        seq_block_ids_list: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(seq_block_ids_list)
        max_blocks = max(len(ids) for ids in seq_block_ids_list)
        block_table = torch.zeros(
            batch, max_blocks, dtype=torch.int32, device='cuda'
        )
        for i, ids in enumerate(seq_block_ids_list):
            if ids:
                block_table[i, :len(ids)] = torch.tensor(
                    ids, dtype=torch.int32, device='cuda'
                )
        return self.k_cache[layer_id], self.v_cache[layer_id], block_table
