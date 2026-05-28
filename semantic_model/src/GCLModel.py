from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from config import HetGATConfig, VICRegConfig
from HetGAT import HetGAT
from ProjectionHead import ProjectionHead
from Augmentor import Augmentor

logger = logging.getLogger(__name__)


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """
    提取方阵 x 的所有非对角线元素（VICReg 协方差项专用）。

    实现原理（以 3×3 为例）：
      flatten()[:-1] 得到前 n²-1 个元素，view(n-1, n+1) 后
      每行末尾恰好是下一行对角线元素的前驱，[:, 1:] 跳过它，
      等价于把对角线元素全部剔除。

    Parameters
    ----------
    x : [n, n] 方阵

    Returns
    -------
    Tensor [n*(n-1)]  所有非对角线元素
    """
    n, m = x.shape
    assert n == m, "输入必须是方阵"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class GCLModel(nn.Module):
    """
    图对比学习模型（VICReg 双视角）。

    训练时
    ------
    HetGAT（共享参数）→ ProjectionHead（共享参数）→ VICReg Loss
    Loss = L_robust + β × L_causal

    L_robust = VICReg(Y_v1a, Y_v1b)   — 语义鲁棒性：同一节点在两次独立扰动下表示一致
    L_causal = VICReg(Y_v1b, Y_v2)    — 因果导向性：v1b 复用，减少前向次数

    推理时
    ------
    只用 HetGAT（ProjectionHead 丢弃），在原始图上前向传播得到节点嵌入 z，
    z 的 L2 范数和节点间余弦距离用于语义评分。

    VICReg 三个约束项的作用
    ----------------------
    不变性（Invariance）：同一节点在两视图下的嵌入 MSE 最小化
    方差（Variance）    ：每维标准差 ≥ γ，防止所有节点坍塌到同一点
    协方差（Covariance）：不同维度去相关，让每维编码不同信息

    正是因为方差项，不同节点的嵌入模长自然产生差异——
    因果中心节点在 V2 里有稳定邻域 → 两视图一致 → 不变性 loss 低 → 嵌入模长大；
    游离时序节点在 V2 里无邻居 → 两视图不一致 → 不变性 loss 高 → 嵌入模长小。
    这与语义评分公式（s_rich = ‖z‖₂）直接对齐。

    Parameters
    ----------
    in_dim     : 节点输入特征维度（TextEncoder 输出，768）
    hetgat_cfg : HetGAT 结构配置
    vicreg_cfg : VICReg 损失超参数
    """

    def __init__(
        self,
        in_dim: int,
        hetgat_cfg: HetGATConfig | None = None,
        vicreg_cfg: VICRegConfig | None = None,
    ) -> None:
        super().__init__()
        hetgat_cfg = hetgat_cfg or HetGATConfig()
        self.vicreg_cfg = vicreg_cfg or VICRegConfig()

        self.encoder   = HetGAT(in_dim, hetgat_cfg)
        # ProjectionHead 输入维度 = HetGAT 最后一层 out_dim（concat=False 时 = hidden_dims[-1]）
        self.projector = ProjectionHead(hetgat_cfg.hidden_dims[-1], hetgat_cfg.proj_dim)

    # ── VICReg Loss ───────────────────────────────────────────────────────────

    def _vicreg_loss(
        self,
        y_a: torch.Tensor,   # [N, D]
        y_b: torch.Tensor,   # [N, D]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        计算两个投影矩阵之间的 VICReg loss。

        N = pool 后总节点数（约 81），D = proj_dim（128）。

        Returns
        -------
        loss    : scalar tensor（可直接加入总 loss 做 backward）
        metrics : {"invariance", "variance", "covariance"} 的 float 值
        """
        cfg = self.vicreg_cfg
        N, D = y_a.shape

        # ── 1. 不变性（Invariance）────────────────────────────────────────────
        # 同一节点在两视图下嵌入的 MSE：拉近正样本对
        inv_loss = F.mse_loss(y_a, y_b)

        # ── 2. 方差（Variance）────────────────────────────────────────────────
        # 每维 std 不低于 gamma；+ eps 避免 sqrt(0) 的梯度问题
        eps = 1e-4
        std_a = (y_a.var(dim=0) + eps).sqrt()   # [D]
        std_b = (y_b.var(dim=0) + eps).sqrt()
        var_loss = (
            F.relu(cfg.gamma - std_a).mean()
            + F.relu(cfg.gamma - std_b).mean()
        ) / 2.0

        # ── 3. 协方差（Covariance）────────────────────────────────────────────
        # 中心化后计算协方差矩阵，惩罚非对角元素（促进不同维度去相关）
        y_a_c = y_a - y_a.mean(dim=0)    # [N, D] 中心化
        y_b_c = y_b - y_b.mean(dim=0)
        cov_a = (y_a_c.T @ y_a_c) / (N - 1)    # [D, D]
        cov_b = (y_b_c.T @ y_b_c) / (N - 1)
        cov_loss = (
            _off_diagonal(cov_a).pow(2).sum() / D
            + _off_diagonal(cov_b).pow(2).sum() / D
        ) / 2.0

        loss = cfg.lambda_ * inv_loss + cfg.mu * var_loss + cfg.nu * cov_loss

        return loss, {
            "invariance": inv_loss.item(),
            "variance":   var_loss.item(),
            "covariance": cov_loss.item(),
        }

    # ── 训练前向 ──────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        graphs: List[Data],
        augmentor: Augmentor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        跨图 pool 所有训练节点，计算 VICReg 双视角 loss。

        流程
        ----
        1. 对每张图产出 (v1a, v1b, v2) 三个增强视图
        2. 分别经 HetGAT 得到 z（节点嵌入，未归一化）
        3. 跨图 cat 后经 ProjectionHead 得到 y（BatchNorm 需要全批统计量）
        4. L_robust = VICReg(Y_v1a, Y_v1b)
           L_causal = VICReg(Y_v1b, Y_v2)    （v1b 复用，减少一次前向）
        5. total = L_robust + β × L_causal

        Parameters
        ----------
        graphs    : List[Data]，训练集图列表
        augmentor : Augmentor 实例

        Returns
        -------
        total_loss : scalar tensor
        metrics    : 各分量值 dict，用于日志记录
        """
        device = next(self.parameters()).device

        Z_v1a: list[torch.Tensor] = []
        Z_v1b: list[torch.Tensor] = []
        Z_v2:  list[torch.Tensor] = []

        for data in graphs:
            data_v1a, data_v1b, data_v2 = augmentor.augment(data)

            z_v1a = self.encoder(
                data_v1a.x.to(device),
                data_v1a.edge_index.to(device),
                data_v1a.edge_type.to(device),
            )
            z_v1b = self.encoder(
                data_v1b.x.to(device),
                data_v1b.edge_index.to(device),
                data_v1b.edge_type.to(device),
            )
            z_v2 = self.encoder(
                data_v2.x.to(device),
                data_v2.edge_index.to(device),
                data_v2.edge_type.to(device),
            )

            Z_v1a.append(z_v1a)
            Z_v1b.append(z_v1b)
            Z_v2.append(z_v2)

        # 跨图 pool → [N_total, hidden_dims[-1]]，再经 ProjectionHead
        # cat 后一次性过 ProjectionHead，让 BatchNorm 看到完整的 batch 统计量
        Y_v1a = self.projector(torch.cat(Z_v1a, dim=0))   # [N_total, proj_dim]
        Y_v1b = self.projector(torch.cat(Z_v1b, dim=0))
        Y_v2  = self.projector(torch.cat(Z_v2,  dim=0))

        loss_robust, m_r = self._vicreg_loss(Y_v1a, Y_v1b)
        loss_causal, m_c = self._vicreg_loss(Y_v1b, Y_v2)

        total = loss_robust + self.vicreg_cfg.beta * loss_causal

        return total, {
            "loss_robust":            loss_robust.item(),
            "loss_robust/invariance": m_r["invariance"],
            "loss_robust/variance":   m_r["variance"],
            "loss_robust/covariance": m_r["covariance"],
            "loss_causal":            loss_causal.item(),
            "loss_causal/invariance": m_c["invariance"],
            "loss_causal/variance":   m_c["variance"],
            "loss_causal/covariance": m_c["covariance"],
            "loss_total":             total.item(),
        }

    # ── 推理 ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, data: Data) -> torch.Tensor:
        """
        推理模式：在原始图（无增强）上前向传播。
        返回 z [N, hidden_dims[-1]]，不经过 ProjectionHead。
        """
        self.eval()
        device = next(self.parameters()).device
        return self.encoder(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_type.to(device),
        )


# ── 独立测试入口 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

    from config import EncoderConfig, HetGATConfig, VICRegConfig, AugmentorConfig
    from TextEncoder import TextEncoder
    from GraphDataset import GraphDataset

    _ROOT = Path(__file__).parent.parent
    graph_dir = _ROOT / "data" / "output" / "doc_results"

    print("[1] 加载 embedding 和图数据...")
    encoder = TextEncoder(EncoderConfig())
    embeddings = encoder.encode_all(graph_dir)
    dataset = GraphDataset(graph_dir, embeddings)

    print("[2] 构建 GCLModel...")
    model = GCLModel(
        in_dim=768,
        hetgat_cfg=HetGATConfig(),
        vicreg_cfg=VICRegConfig(),
    )
    aug = Augmentor(AugmentorConfig())

    print("[3] 跑一次 compute_loss（前 3 张图）...")
    graphs = list(dataset)[:3]
    model.train()
    loss, metrics = model.compute_loss(graphs, aug)

    print(f"\n  total loss : {loss.item():.4f}")
    for k, v in metrics.items():
        print(f"  {k:<35}: {v:.4f}")

    print("\n[4] 测试 encode（推理模式）...")
    z = model.encode(graphs[0])
    print(f"  z.shape = {z.shape}  (期望 [{graphs[0].num_nodes}, {HetGATConfig().hidden_dims[-1]}])")

    print("\n✓ GCLModel 全部测试通过")
