from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, scatter, degree

from config import ModelConfig


# ── 1. LocalFeatureEncoder ────────────────────────────────────────────────────

class LocalFeatureEncoder(nn.Module):
    """
    节点特征 → GAT 输入。与 NIRM 对齐时：hidden_dims=[]、output_dim=1、out_activation='none'
    即单层 Linear(feat_dim, 1)，无 ReLU（对应 NIRM 的 lin1）。

    可选深层：feat_dim → 64 → 32 → output_dim；隐藏层 ReLU + Dropout。
    输出层支持 none / ReLU / LeakyReLU / Sigmoid。
    """

    def __init__(
        self,
        input_dim: int = 15,   # 通过 TopoModel 调用时由 cfg.feat_dim(=8) 覆盖；直接使用请显式传参
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        out_activation: str = "relu",
        out_leaky_slope: float = 0.1,
        output_dim: int = 16,  # MLP 输出维度，直接作为 GAT 输入维度
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            linear = nn.Linear(prev_dim, h)
            nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            nn.init.constant_(linear.bias, 0.0)
            layers.extend([linear, nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h

        self.hidden_net = nn.Sequential(*layers) if layers else nn.Identity()

        # 输出层：初始化与输出激活对应
        out_layer = nn.Linear(prev_dim, output_dim)
        if out_activation == "none":
            # 与 NIRM lin1 一致：默认 Linear 初始化，无后接激活
            out_act: nn.Module = nn.Identity()
        elif out_activation == "relu":
            nn.init.kaiming_uniform_(out_layer.weight, nonlinearity="relu")
            out_act = nn.ReLU()
        elif out_activation == "leaky_relu":
            nn.init.kaiming_uniform_(
                out_layer.weight,
                a=out_leaky_slope,
                nonlinearity="leaky_relu",
            )
            out_act = nn.LeakyReLU(out_leaky_slope)
        elif out_activation == "sigmoid":
            nn.init.xavier_uniform_(out_layer.weight)
            out_act = nn.Sigmoid()
        else:
            raise ValueError(
                f"不支持的 LocalFeatureEncoder 输出激活: {out_activation}"
            )
        nn.init.constant_(out_layer.bias, 0.0)
        self.out_layer = out_layer
        self.out_activation = out_act
        self.out_activation_name = out_activation
        self.out_leaky_slope = out_leaky_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, input_dim] 节点特征矩阵
        Returns:
            [N, output_dim] 节点表示向量（output_dim=1 时退化为 [N]）
        """
        hidden = self.hidden_net(x)
        pre = self.out_layer(hidden)
        post = self.out_activation(pre)

        pre_detached = pre.detach()
        post_detached = post.detach()
        if self.output_dim == 1:
            # 单维输出：保留标量统计
            self._last_pre_min = float(pre_detached.min().item())
            self._last_pre_max = float(pre_detached.max().item())
            self._last_pre_mean = float(pre_detached.mean().item())
            self._last_pre_std = float(pre_detached.std(unbiased=False).item())
            self._last_post_min = float(post_detached.min().item())
            self._last_post_max = float(post_detached.max().item())
            self._last_post_mean = float(post_detached.mean().item())
            self._last_post_std = float(post_detached.std(unbiased=False).item())
            self._last_zero_ratio = float((post_detached == 0).float().mean().item())
            self._last_lt_005_ratio = float((post_detached < 0.05).float().mean().item())
            self._last_gt_095_ratio = float((post_detached > 0.95).float().mean().item())
            return post.squeeze(-1)  # [N, 1] → [N]
        else:
            # 多维输出：记录范数统计
            pre_norm = pre_detached.norm(dim=-1)
            post_norm = post_detached.norm(dim=-1)
            self._last_pre_min = float(pre_norm.min().item())
            self._last_pre_max = float(pre_norm.max().item())
            self._last_pre_mean = float(pre_norm.mean().item())
            self._last_pre_std = float(pre_norm.std(unbiased=False).item())
            self._last_post_min = float(post_norm.min().item())
            self._last_post_max = float(post_norm.max().item())
            self._last_post_mean = float(post_norm.mean().item())
            self._last_post_std = float(post_norm.std(unbiased=False).item())
            self._last_zero_ratio = float((post_detached.norm(dim=-1) < 1e-6).float().mean().item())
            self._last_lt_005_ratio = 0.0
            self._last_gt_095_ratio = 0.0
            return post  # [N, output_dim]


# ── 2. DirectedGATLayer ───────────────────────────────────────────────────────

class DirectedGATLayer(nn.Module):
    """
    单层有向图注意力网络。

    消息传递方向：destination 节点聚合其 in-neighbors（指向它的节点）的信息。
    与 PyTorch Geometric GATConv 默认约定一致：消息从 source(j) 流向 target(i)。
    注意力计算：[h_src || h_dst] 拼接后与可学习注意力向量做点积，
                按 destination 节点分组 softmax 归一化（对所有入边归一化）。
    多头输出：拼接（concat=True）。
    """

    def __init__(
        self,
        in_features: int,
        out_features_per_head: int,
        num_heads: int,
        dropout: float = 0.2,
        alpha: float = 0.2,      # LeakyReLU 负斜率
    ) -> None:
        super().__init__()
        self.out_features_per_head = out_features_per_head
        self.num_heads = num_heads

        # 线性变换：in_features → num_heads * out_per_head
        self.linear = nn.Linear(in_features, num_heads * out_features_per_head, bias=False)
        # 注意力参数：每个头独立的 [2 * out_per_head] 向量
        self.attention = nn.Parameter(torch.empty(num_heads, 2 * out_features_per_head))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.linear.weight)
        nn.init.kaiming_uniform_(self.attention)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x          : [N, in_features] 节点特征
            edge_index : [2, E] 有向边，edge_index[0]=src, edge_index[1]=dst
        Returns:
            [N, num_heads * out_features_per_head]
        """
        num_nodes = x.size(0)

        # 无边时直接返回零张量（避免后续索引越界）
        if edge_index.numel() == 0:
            return x.new_zeros((num_nodes, self.num_heads * self.out_features_per_head))

        row, col = edge_index[0], edge_index[1]  # src, dst

        # 线性变换 → [N, num_heads, out_per_head]
        Wh = self.linear(x).view(num_nodes, self.num_heads, self.out_features_per_head)

        # 取边两端节点的变换特征
        Wh_src = Wh[row]  # [E, H, F]  source 特征
        Wh_dst = Wh[col]  # [E, H, F]  destination 特征

        # 拼接 [h_src || h_dst] → [E, H, 2F]，计算注意力原始分 → [E, H]
        e = torch.cat([Wh_src, Wh_dst], dim=-1)
        e = self.leakyrelu((e * self.attention.unsqueeze(0)).sum(dim=-1))

        # 按 destination 分组 softmax：每个 destination 的所有入边注意力归一化
        alpha = softmax(e, col, num_nodes=num_nodes)  # [E, H]
        alpha = self.dropout(alpha)

        # 聚合：加权 src 特征累加到 dst → [N, H, F]
        messages = alpha.unsqueeze(-1) * Wh_src       # [E, H, F]
        out = scatter(messages, col, dim=0, dim_size=num_nodes, reduce="sum")  # [N, H, F]

        out = F.leaky_relu(out)
        out = self.dropout(out)

        # 拼接多头输出 → [N, H*F]
        return out.view(num_nodes, self.num_heads * self.out_features_per_head)


# ── 3. DirectedGATEncoder ─────────────────────────────────────────────────────

class DirectedGATEncoder(nn.Module):
    """
    多层有向 GAT，逐层扩展节点感受野。

    第 l 层输入维度 = 第 l-1 层的 hidden_dims[l-1] * num_heads[l-1]（首层为 in_dim）。
    最终输出维度 = hidden_dims[-1] * num_heads[-1]，暴露为 self.final_dim。
    """

    def __init__(
        self,
        num_layers: int = 3,
        in_dim: int = 1,
        hidden_dims: list[int] | None = None,
        num_heads: list[int] | None = None,
        dropout: float = 0.2,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16, 8]
        if num_heads is None:
            num_heads = [8, 4, 2]

        assert len(hidden_dims) == num_layers, "hidden_dims 长度须等于 num_layers"
        assert len(num_heads) == num_layers,   "num_heads 长度须等于 num_layers"

        self.convs = nn.ModuleList()
        current_dim = in_dim
        for l in range(num_layers):
            self.convs.append(
                DirectedGATLayer(
                    in_features=current_dim,
                    out_features_per_head=hidden_dims[l],
                    num_heads=num_heads[l],
                    dropout=dropout,
                    alpha=alpha,
                )
            )
            current_dim = hidden_dims[l] * num_heads[l]

        self.final_dim = current_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x          : [N, in_dim]
            edge_index : [2, E]
        Returns:
            [N, final_dim]
        """
        for conv in self.convs:
            x = conv(x, edge_index)
        return x


# ── 4. NodeScorer ─────────────────────────────────────────────────────────────

class NodeScorer(nn.Module):
    """
    双尺度影响力评分头。

    局部分 s_local：
        节点与 in-邻居的语义相似度均值 + 归一化总度（in_degree + out_degree）。
        衡量节点在局部子图中的结构中心性与信息聚合能力。

    全局分 s_global：
        (h_i + Σh_j) / sqrt(deg+1) 与可学习投影向量 p 的内积。
        衡量节点对整体网络连通性的潜在扰动影响。

    融合：s = s_local + s_global。

    度使用 in_degree + out_degree，对齐 CreateGroundTruth 中基于有向可达性
    的标签语义（路径瓶颈节点需在两个方向都有连接）。

    参数：
        in_dim             : GAT 输出的节点表示维度
        deg_norm           : 度归一化方式，'max'|'log'|'sqrt'|'none'
        s_dis_norm         : 融合分后处理（仅 sigmoid 模式），None|'zscore'|'minmax'
        scorer_output_norm : 最终输出归一化：'sigmoid' | 'minmax' | 'none'
        return_1d          : True 则返回 [N]，否则返回 [N,1]
    """

    def __init__(
        self,
        in_dim: int,
        deg_norm: str = "max",
        s_dis_norm: str | None = None,
        scorer_output_norm: str = "sigmoid",
        return_1d: bool = True,
    ) -> None:
        super().__init__()
        self.global_proj = nn.Linear(in_dim, 1, bias=True)
        self.deg_norm = deg_norm
        self.s_dis_norm = s_dis_norm
        self.scorer_output_norm = scorer_output_norm
        self.return_1d = return_1d

    def _deg_normalize(self, deg: torch.Tensor) -> torch.Tensor:
        """将度归一化为 [0,1]（或全零）。"""
        eps = 1e-6
        if self.deg_norm == "max":
            return (deg / (deg.max() + eps)).unsqueeze(-1)
        elif self.deg_norm == "log":
            return (torch.log1p(deg) / (torch.log1p(deg.max()) + eps)).unsqueeze(-1)
        elif self.deg_norm == "sqrt":
            return (torch.sqrt(deg) / (torch.sqrt(deg.max()) + eps)).unsqueeze(-1)
        else:
            return torch.zeros(deg.size(0), 1, dtype=deg.dtype, device=deg.device)

    def _normalize_output(self, s: torch.Tensor) -> torch.Tensor:
        """可选的输出归一化。"""
        eps = 1e-6
        if self.s_dis_norm == "zscore":
            return (s - s.mean()) / (s.std() + eps)
        elif self.s_dis_norm == "minmax":
            return (s - s.min()) / (s.max() - s.min() + eps)
        return s

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h          : [N, in_dim] GAT 输出的节点表示
            edge_index : [2, E] 有向边
        Returns:
            s_local  : 局部分 [N] 或 [N,1]
            s_global : 全局分 [N] 或 [N,1]
            s        : 融合分 [N] 或 [N,1]
        """
        num_nodes = h.size(0)

        # ── 度：in_degree + out_degree ──────────────────────────────────────
        if edge_index.numel() > 0:
            row, col = edge_index[0], edge_index[1]
            out_deg = degree(row, num_nodes=num_nodes, dtype=h.dtype)
            in_deg  = degree(col, num_nodes=num_nodes, dtype=h.dtype)
        else:
            out_deg = h.new_zeros(num_nodes)
            in_deg  = h.new_zeros(num_nodes)

        deg     = (out_deg + in_deg).clamp(min=1.0)  # 避免除零

        # ── 局部分 ───────────────────────────────────────────────────────────
        if edge_index.numel() > 0:
            # 聚合 in-邻居（src）特征到 dst，与 GAT 聚合方向一致
            neigh_sum = scatter(h[row], col, dim=0, dim_size=num_nodes, reduce="sum")  # [N, d]
        else:
            neigh_sum = h.new_zeros(num_nodes, h.size(1))

        neigh_mean = (neigh_sum + h) / (in_deg + 1.0).unsqueeze(-1)   # [N, d]  含自环：(邻居+自身)/(in_deg+1)
        local_sim  = (h * neigh_mean).sum(dim=-1, keepdim=True)      # [N, 1]
        deg_norm   = self._deg_normalize(deg)                         # [N, 1]
        s_local    = local_sim + deg_norm                             # [N, 1]

        # ── 全局分 ───────────────────────────────────────────────────────────
        # 融合表示：(h_i + Σh_j) / sqrt(deg_i + 1)，缓解高度节点主导
        h_hat    = (h + neigh_sum) / (deg + 1.0).unsqueeze(-1)  # [N, d]，与论文一致：算术均值归一化
        s_global = self.global_proj(h_hat)                                  # [N, 1]

        # ── 融合 ─────────────────────────────────────────────────────────────
        raw = s_local + s_global                                      # [N, 1]
        if self.scorer_output_norm == "none":
            s = raw                                                   # [N, 1]，无后处理，与论文 NIRM 一致
        elif self.scorer_output_norm == "minmax":
            eps = 1e-6
            s = (raw - raw.min()) / (raw.max() - raw.min() + eps)    # [N, 1]，per-doc [0,1]
        else:
            raw = self._normalize_output(raw)                        # 可选 zscore/minmax（仅 sigmoid 模式）
            s = torch.sigmoid(raw)                                   # [N, 1]，约束到 (0,1)

        raw_detached = raw.detach()
        pred_detached = s.detach()
        self._last_raw_min = float(raw_detached.min().item())
        self._last_raw_max = float(raw_detached.max().item())
        self._last_raw_mean = float(raw_detached.mean().item())
        self._last_raw_std = float(raw_detached.std(unbiased=False).item())
        self._last_pred_min = float(pred_detached.min().item())
        self._last_pred_max = float(pred_detached.max().item())
        self._last_pred_mean = float(pred_detached.mean().item())
        self._last_pred_std = float(pred_detached.std(unbiased=False).item())
        self._last_pred_range = self._last_pred_max - self._last_pred_min
        self._last_s_local_mean = float(s_local.detach().mean().item())
        self._last_s_local_std = float(s_local.detach().std(unbiased=False).item())
        self._last_s_global_mean = float(s_global.detach().mean().item())
        self._last_s_global_std = float(s_global.detach().std(unbiased=False).item())
        self._last_global_proj_weight_norm = float(
            self.global_proj.weight.detach().norm().item()
        )
        self._last_global_proj_bias = float(self.global_proj.bias.detach().item())

        if self.return_1d:
            return s_local.squeeze(-1), s_global.squeeze(-1), s.squeeze(-1)
        return s_local, s_global, s



# ── 5. TopoModel ─────────────────────────────────────────────────────────────

class TopoModel(nn.Module):
    """
    拓扑评分完整模型（LocalFeatureEncoder + DirectedGATEncoder + NodeScorer）。

    流程：
        features [N, feat_dim]
        → LocalFeatureEncoder   → [N]     初始重要性分
        → unsqueeze             → [N, 1]  作为 GAT 输入
        → DirectedGATEncoder    → [N, final_dim]  高阶有向结构表示
        → NodeScorer            → [N]     最终重要性分（与 rank_labels 算 MSE）

    参数：
        cfg : ModelConfig 实例，控制 MLP / GAT / NodeScorer 所有超参数。
              传 None 时使用 ModelConfig() 默认值。
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        _feat_out_dim = cfg.mlp_hidden_dims[-1] if cfg.mlp_hidden_dims else 1
        self.feature_encoder = LocalFeatureEncoder(
            input_dim=cfg.feat_dim,
            hidden_dims=cfg.mlp_hidden_dims[:-1] if cfg.mlp_hidden_dims else [],
            dropout=cfg.mlp_dropout,
            out_activation=cfg.feature_out_activation,
            out_leaky_slope=cfg.feature_out_leaky_slope,
            output_dim=_feat_out_dim,
        )
        self.gat_encoder = DirectedGATEncoder(
            num_layers=cfg.gat_num_layers,
            in_dim=_feat_out_dim,
            hidden_dims=cfg.gat_hidden_dims,
            num_heads=cfg.gat_num_heads,
            dropout=cfg.gat_dropout,
            alpha=cfg.gat_alpha,
        )
        final_dim = cfg.gat_hidden_dims[-1] * cfg.gat_num_heads[-1]
        self.scorer = NodeScorer(
            in_dim=final_dim,
            deg_norm=cfg.deg_norm,
            s_dis_norm=cfg.s_dis_norm,
            scorer_output_norm=cfg.scorer_output_norm,
            return_1d=True,
        )

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features   : [N, feat_dim] 节点结构特征
            edge_index : [2, E] 有向边索引
        Returns:
            [N] 节点重要性预测分
        """
        initial = self.feature_encoder(features)                   # [N, feat_out_dim] 或 [N] 当 feat_out_dim=1
        if initial.dim() == 1:
            initial = initial.unsqueeze(-1)                        # [N] → [N, 1] 作为 GAT 输入
        h       = self.gat_encoder(initial, edge_index)          # [N, feat_out_dim] → [N, final_dim]
        s_local, s_global, s = self.scorer(h, edge_index)        # [N]
        # 缓存贡献度诊断数据（每次前向覆盖，供 SwanLab 读取最后一批统计）
        self._last_s_local_abs_mean  = float(s_local.detach().abs().mean().item())
        self._last_s_global_abs_mean = float(s_global.detach().abs().mean().item())
        self._last_feature_pre_min = self.feature_encoder._last_pre_min
        self._last_feature_pre_max = self.feature_encoder._last_pre_max
        self._last_feature_pre_mean = self.feature_encoder._last_pre_mean
        self._last_feature_pre_std = self.feature_encoder._last_pre_std
        self._last_feature_post_min = self.feature_encoder._last_post_min
        self._last_feature_post_max = self.feature_encoder._last_post_max
        self._last_feature_post_mean = self.feature_encoder._last_post_mean
        self._last_feature_post_std = self.feature_encoder._last_post_std
        self._last_feature_zero_ratio = self.feature_encoder._last_zero_ratio
        self._last_feature_lt_005_ratio = self.feature_encoder._last_lt_005_ratio
        self._last_feature_gt_095_ratio = self.feature_encoder._last_gt_095_ratio
        self._last_raw_min = self.scorer._last_raw_min
        self._last_raw_max = self.scorer._last_raw_max
        self._last_raw_mean = self.scorer._last_raw_mean
        self._last_raw_std = self.scorer._last_raw_std
        self._last_pred_min = self.scorer._last_pred_min
        self._last_pred_max = self.scorer._last_pred_max
        self._last_pred_mean = self.scorer._last_pred_mean
        self._last_pred_std = self.scorer._last_pred_std
        self._last_pred_range = self.scorer._last_pred_range
        self._last_s_local_mean = self.scorer._last_s_local_mean
        self._last_s_local_std = self.scorer._last_s_local_std
        self._last_s_global_mean = self.scorer._last_s_global_mean
        self._last_s_global_std = self.scorer._last_s_global_std
        self._last_global_proj_weight_norm = self.scorer._last_global_proj_weight_norm
        self._last_global_proj_bias = self.scorer._last_global_proj_bias
        return s


# ── 验证入口 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 把 src 目录加入 path，以便读取 Dataset
    sys.path.insert(0, str(Path(__file__).parent))
    from Dataset import TopoDataset, _setup_logging
    from config import TopoConfig

    cfg  = TopoConfig(num_workers=1)
    base = Path(__file__).resolve().parents[1]
    logger = _setup_logging(base / "log", "model_verify")

    dataset = TopoDataset.build(
        base / "output" / "features",
        base / "output" / "labels",
        base / "data"   / "output" / "doc_results",
        cfg, logger,
    )

    model = TopoModel(cfg=ModelConfig())
    model.eval()

    sample     = dataset[0]
    features   = sample["features"]    # [N, 8]
    edge_index = sample["edge_index"]  # [2, E]
    labels     = sample["labels"]      # [N]

    with torch.no_grad():
        preds = model(features, edge_index)

    print(f"doc_id     : {sample['doc_id']}")
    print(f"num_nodes  : {sample['num_nodes']}")
    print(f"preds shape: {preds.shape}")
    print(f"preds      : {preds}")
    print(f"labels     : {labels}")
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
