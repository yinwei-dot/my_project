from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    VICReg 投影头：将 HetGAT 输出的节点嵌入 z 扩展到更高维度，
    专用于 VICReg loss 计算，训练结束后丢弃，不参与最终评分。

    结构
    ----
    Linear(in_dim → proj_dim) → BatchNorm1d → ReLU → Linear(proj_dim → proj_dim)

    使用 BatchNorm1d 而非 LayerNorm 的原因
    ---------------------------------------
    VICReg 的方差项（Variance term）需要在 batch 维度上计算标准差。
    BatchNorm 在训练时依赖 batch 统计量（均值/方差），天然与 VICReg 对齐。
    因此调用 forward() 时，N 应为跨图 pool 后的总节点数（约 81），
    而非单图节点数（约 9）——否则 BatchNorm 统计量极不稳定。

    Parameters
    ----------
    in_dim   : HetGAT 最后一层输出维度（hidden_dims[-1]，默认 32）
    proj_dim : 投影空间维度（默认 128，来自 HetGATConfig.proj_dim）
    """

    def __init__(self, in_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : [N, in_dim]  跨图 pool 后的节点嵌入矩阵

        Returns
        -------
        y : [N, proj_dim]  投影空间嵌入，用于 VICReg loss
        """
        return self.net(z)


# ── 独立测试入口 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from config import HetGATConfig

    cfg = HetGATConfig()
    in_dim   = cfg.hidden_dims[-1]   # 32（最后一层 mean-pooled 输出）
    proj_dim = cfg.proj_dim           # 128

    head = ProjectionHead(in_dim, proj_dim)
    print(f"ProjectionHead: {in_dim}d → {proj_dim}d")
    print(head)

    # 模拟 12 图 × 9 节点/图 = 108 个节点的跨图 pool
    dummy = torch.randn(108, in_dim)
    head.train()                      # BatchNorm 需要 train 模式才能用 batch 统计量
    out = head(dummy)
    assert out.shape == (108, proj_dim), f"期望 (108, {proj_dim})，实际 {out.shape}"
    print(f"\n输入 {dummy.shape} → 输出 {out.shape}  ✓")

    # 验证单图（N=7）时 BatchNorm 警告（不应在此场景单独调用）
    single = torch.randn(7, in_dim)
    out_single = head(single)
    print(f"单图输入 {single.shape} → 输出 {out_single.shape}  (仅测试形状，统计量不稳定)")
