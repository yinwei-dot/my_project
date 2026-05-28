"""
GCL.py — 多视角图对比学习器（多图版本，支持跨文档负样本）

视角1 (鲁棒性): 每张图做两次独立增强，拼接所有图节点做全局 NT-Xent
视角2 (因果导向): 完整图编码，causal 边作为正样本对进行节点级对比
"""
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data

from HetGAT import HetGAT


class GraphContrastiveLearner(nn.Module):
    """
    共享 HetGAT 编码器 + MLP 投影头，支持多图批次训练。

    Parameters
    ----------
    in_dim : int            节点特征维度（RoBERTa hidden_size = 768）
    hetgat_hidden : int     HetGAT 中间层维度（需整除 num_heads）
    hetgat_out : int        HetGAT 输出维度（节点嵌入维度）
    proj_hidden : int       投影头隐层维度
    proj_out : int          投影头输出维度（InfoNCE 空间）
    num_layers : int        HetGAT 层数
    num_heads : int         多头数量
    num_relations : int     边类型数（默认 2: temporal=0, causal=1）
    temperature : float     InfoNCE 温度
    perspective_weights     {'robustness': float, 'causal': float} 两视角损失权重
    feat_mask_rate : float  视角1 特征掩码比率
    edge_drop_rate : float  视角1 边丢弃比率
    use_semantic_att : bool 是否启用 HetGAT 语义注意力
    """

    def __init__(
        self,
        in_dim: int,
        hetgat_hidden: int,
        hetgat_out: int,
        proj_hidden: int,
        proj_out: int,
        num_layers: int,
        num_heads: int,
        num_relations: int = 2,
        temperature: float = 0.5,
        perspective_weights: Dict[str, float] = None,
        feat_mask_rate: float = 0.2,
        edge_drop_rate: float = 0.1,
        use_semantic_att: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.feat_mask_rate = feat_mask_rate
        self.edge_drop_rate = edge_drop_rate

        if perspective_weights is None:
            perspective_weights = {"robustness": 1.0, "causal": 0.6}
        self.perspective_weights = perspective_weights

        # 共享编码器
        self.encoder = HetGAT(
            in_dim, hetgat_hidden, hetgat_out,
            num_layers, num_heads, num_relations,
            dropout=0.6,
            use_semantic_att=use_semantic_att,
        )

        # 投影头（仅训练期使用，推理用 encoder 输出）
        self.projector = nn.Sequential(
            Linear(hetgat_out, proj_hidden),
            nn.ReLU(),
            Linear(proj_hidden, proj_out),
        )

    # ------------------------------------------------------------------
    def _augment(self, x, edge_index, edge_type):
        """特征掩码 + 随机丢边（保持 edge_type 一致）。"""
        device = x.device
        x_aug = x.clone()
        mask = torch.rand(x.shape, device=device) < self.feat_mask_rate
        x_aug[mask] = 0.0

        if edge_index.size(1) == 0:
            return x_aug, edge_index, edge_type

        keep = torch.rand(edge_index.size(1), device=device) >= self.edge_drop_rate
        return x_aug, edge_index[:, keep], edge_type[keep]

    # ------------------------------------------------------------------
    def compute_loss(self, graphs: List[Data]) -> torch.Tensor:
        """
        计算多视角对比损失。

        Parameters
        ----------
        graphs : list of PyG Data
            每个 Data 有 x [N,768], edge_index [2,E], edge_type [E]
        """
        device = next(self.parameters()).device

        # ---- 视角1: 鲁棒性（跨图全局 NT-Xent） ----
        z1_list, z2_list = [], []
        for g in graphs:
            x = g.x.to(device)
            ei = g.edge_index.to(device)
            et = g.edge_type.to(device)

            x1, ei1, et1 = self._augment(x, ei, et)
            x2, ei2, et2 = self._augment(x, ei, et)

            z1_list.append(self.projector(self.encoder(x1, ei1, et1)))
            z2_list.append(self.projector(self.encoder(x2, ei2, et2)))

        z1 = F.normalize(torch.cat(z1_list, dim=0), dim=-1)   # [total_N, proj_out]
        z2 = F.normalize(torch.cat(z2_list, dim=0), dim=-1)
        total_N = z1.shape[0]

        sim12 = torch.mm(z1, z2.t()) / self.temperature        # [total_N, total_N]
        sim21 = sim12.t()
        labels = torch.arange(total_N, device=device)
        loss_robust = (
            F.cross_entropy(sim12, labels) + F.cross_entropy(sim21, labels)
        ) / 2 * self.perspective_weights["robustness"]

        # ---- 视角2: 因果导向（各图内部，causal 边正样本） ----
        loss_causal_acc = torch.tensor(0.0, device=device)
        n_causal = 0
        for g in graphs:
            x = g.x.to(device)
            ei = g.edge_index.to(device)
            et = g.edge_type.to(device)

            causal_mask = et == 1
            if not causal_mask.any():
                continue

            causal_ei = ei[:, causal_mask]
            z = F.normalize(self.projector(self.encoder(x, ei, et)), dim=-1)
            N = z.shape[0]

            sim = torch.mm(z, z.t()) / self.temperature
            sim.masked_fill_(torch.eye(N, device=device).bool(), float("-inf"))

            src, dst = causal_ei
            pos_sim = sim[src, dst]
            logsumexp_row = torch.logsumexp(sim, dim=1)
            loss_g = (-pos_sim + logsumexp_row[src]).mean()
            loss_causal_acc = loss_causal_acc + loss_g
            n_causal += 1

        if n_causal > 0:
            loss_causal = (loss_causal_acc / n_causal) * self.perspective_weights["causal"]
        else:
            loss_causal = torch.tensor(0.0, device=device)

        return loss_robust + loss_causal

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """推理：返回单张图的节点嵌入 [N, hetgat_out]，不经过投影头。"""
        device = next(self.parameters()).device
        x = data.x.to(device)
        ei = data.edge_index.to(device)
        et = data.edge_type.to(device)
        return self.encoder(x, ei, et)
