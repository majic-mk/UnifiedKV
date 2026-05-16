import math
import torch
import torch.nn.functional as F
from typing import Any, Dict, Tuple


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    GQA支持：将KV head从h_kv扩展到h_q。
    Qwen2-7B: h_kv=8, h_q=32, n_rep=4
    x shape: [batch, h_kv, seq, head_dim]
    """
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, h, n_rep, s, d)
        .reshape(b, h * n_rep, s, d)
    )


class BlockAlignedSnapKV:
    """
    三段式块对齐KV压缩。

    与原始SnapKV的三个关键区别：
      1. Q来源：使用真实q_proj投影得到的Q，不能用K_obs代替
      2. 压缩粒度：以物理块为单位而非Token，消除内部碎片
      3. 池化：使用mean不是sum（window大小不同时sum数值不可比）

    W = L_obs 设计：
      打分窗口大小W与尾端保护区大小L_obs相等
      语义对齐：用L_obs个Q打分，保护L_obs个tail KV
      不再单独设window_size参数

    压缩率估算（默认参数）：
      L=16500, sink=64, obs=64, r=0.20, B=16
      N = (16500-64-64)÷16 = 1023块
      k_blocks = floor(0.2×1023) = 204块
      L_final = 64 + 204×16 + 64 = 3392 tokens
      压缩率 = 20.6%，显存节省约79.4%
    """

    def __init__(
        self,
        block_size: int = 16,
        sink_len: int = 64,
        obs_len: int = 64,
        retain_ratio: float = 0.20,
        retain_budget_tokens: int = 0,
    ):
        self.B = block_size
        self.Ls = sink_len
        self.Lo = obs_len
        self.retain_ratio = retain_ratio
        self.retain_budget_tokens = max(0, int(retain_budget_tokens))
        self.last_debug: Dict[str, Any] = {}

    @property
    def compression_mode(self) -> str:
        return "fixed_budget" if int(self.retain_budget_tokens) > 0 else "ratio"

    def compress(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        Q_obs: torch.Tensor,
        n_rep: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入：prefill完整KV
        输出：压缩后KV，L_final = L_sink + k_blocks×B + L_obs
        """
        b, h_kv, L, d = K.shape
        budget = int(self.retain_budget_tokens)
        if budget > 0 and L <= budget:
            self.last_debug = {
                "block_size": int(self.B),
                "sink_len": int(self.Ls),
                "obs_len": int(self.Lo),
                "retain_ratio": float(self.retain_ratio),
                "retain_budget_tokens": int(budget),
                "compression_mode": "fixed_budget",
                "logical_seq_len": int(L),
                "mid_tokens_before_align": 0,
                "mid_tokens_aligned": 0,
                "mid_block_count": 0,
                "retained_block_count": 0,
                "effective_retained_tokens": int(L),
                "retained_block_idx": [],
                "compression_skipped": "input_le_budget",
            }
            return K, V
        assert L >= self.Ls + self.Lo + self.B, (
            f"序列太短({L})，需要至少{self.Ls + self.Lo + self.B}个token"
        )

        # ── 步骤1：三段切分 ────────────────────────────────
        K_sink = K[:, :, :self.Ls, :]
        V_sink = V[:, :, :self.Ls, :]
        K_obs = K[:, :, -self.Lo:, :]
        V_obs = V[:, :, -self.Lo:, :]
        K_mid = K[:, :, self.Ls:-self.Lo, :]
        V_mid = V[:, :, self.Ls:-self.Lo, :]

        # 中段块对齐：向上取整至B的整数倍
        L_premid = K_mid.shape[2]
        L_mid_aligned = math.ceil(L_premid / self.B) * self.B
        if L_mid_aligned > L_premid:
            pad = L_mid_aligned - L_premid
            K_mid = F.pad(K_mid, (0, 0, 0, pad))
            V_mid = F.pad(V_mid, (0, 0, 0, pad))
        N = L_mid_aligned // self.B

        # ── 步骤2：观测窗口注意力（W = Lo，全部使用）────────
        K_mid_exp = repeat_kv(K_mid, n_rep)
        scale = d ** -0.5
        A = torch.softmax(
            torch.matmul(Q_obs, K_mid_exp.transpose(-2, -1)) * scale,
            dim=-1
        )

        # ── 步骤3：三重池化 ─────────────────────────────────
        S = A.mean(dim=2)
        S = S.max(dim=1).values
        S_block = S.view(b, N, self.B).sum(dim=-1)
        S_smooth = F.max_pool1d(
            S_block.unsqueeze(1), kernel_size=3, stride=1, padding=1
        ).squeeze(1)

        # ── 步骤4：块对齐TopK ────────────────────────────────
        if budget > 0:
            mid_budget_tokens = max(0, int(budget) - int(self.Ls) - int(self.Lo))
            k_blocks = max(0, min(int(N), int(math.floor(mid_budget_tokens / self.B))))
            compression_mode = "fixed_budget"
        else:
            k_blocks = max(1, int(math.floor(self.retain_ratio * N)))
            compression_mode = "ratio"
        if k_blocks > 0:
            _, topk_idx = S_smooth.topk(k_blocks, dim=-1)
            topk_idx = topk_idx.sort(dim=-1).values
        else:
            topk_idx = torch.empty((b, 0), dtype=torch.long, device=S_smooth.device)
        effective_retained_tokens = int(self.Ls) + int(k_blocks) * int(self.B) + int(self.Lo)
        self.last_debug = {
            "block_size": int(self.B),
            "sink_len": int(self.Ls),
            "obs_len": int(self.Lo),
            "retain_ratio": float(self.retain_ratio),
            "retain_budget_tokens": int(budget),
            "compression_mode": compression_mode,
            "logical_seq_len": int(L),
            "mid_tokens_before_align": int(L_premid),
            "mid_tokens_aligned": int(L_mid_aligned),
            "mid_block_count": int(N),
            "retained_block_count": int(k_blocks),
            "effective_retained_tokens": int(effective_retained_tokens),
            "retained_block_idx": topk_idx.detach().cpu().tolist(),
        }

        K_mid_b = K_mid.view(b, h_kv, N, self.B, d)
        V_mid_b = V_mid.view(b, h_kv, N, self.B, d)
        if k_blocks > 0:
            idx = topk_idx[:, None, :, None, None].expand(b, h_kv, k_blocks, self.B, d)
            K_mid_c = K_mid_b.gather(2, idx).reshape(b, h_kv, k_blocks * self.B, d)
            V_mid_c = V_mid_b.gather(2, idx).reshape(b, h_kv, k_blocks * self.B, d)
        else:
            K_mid_c = K_mid[:, :, :0, :]
            V_mid_c = V_mid[:, :, :0, :]

        # ── 步骤5：拼接 ──────────────────────────────────────
        K_final = torch.cat([K_sink, K_mid_c, K_obs], dim=2)
        V_final = torch.cat([V_sink, V_mid_c, V_obs], dim=2)
        return K_final, V_final
