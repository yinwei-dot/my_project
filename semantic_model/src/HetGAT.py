from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter

from config import HetGATConfig

# 四通道关系编号（与 GraphDataset 保持一致）
# 0=temporal_forward  1=temporal_reverse  2=causal_forward  3=causal_reverse
NUM_RELATIONS = 4


class HetGATConv(torch.nn.Module):
    """
    单层异构图注意力（Heterogeneous Graph Attention Convolution）。

    每种关系拥有独立的投影矩阵 W_r 和注意力参数 a_r，
    让模型对"时序正向/反向"和"因果正向/反向"四种消息通道
    分别学习不同的权重，避免把方向信息混在一起。

    消息传递方向：edge_index[0]（src）→ edge_index[1]（dst），
    即 dst 节点聚合来自 src 节点的信息（标准 GAT 方向）。

    Parameters
    ----------
    in_dim : int
        输入节点特征维度。
    out_dim : int
        每个注意力头的输出维度。
    num_heads : int
        注意力头数。
    num_relations : int
        关系类型数量（本项目固定为 4）。
    dropout : float
        注意力权重的 dropout 概率。
    concat : bool
        True  → 多头输出 concat，输出维度 = num_heads × out_dim
        False → 多头输出 mean，  输出维度 = out_dim（最后一层使用）
    use_semantic_att : bool
        是否在多通道融合时学习关系级语义注意力权重。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        num_relations: int,
        dropout: float = 0.3,
        concat: bool = True,
        use_semantic_att: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim          # 每个头的输出维度
        self.head_dim = num_heads * out_dim  # concat 后的总维度
        self.concat = concat
        self.dropout = dropout
        self.use_semantic_att = use_semantic_att

        # 保存最近一次 forward 计算的语义注意力权重，供外部读取
        # 形状 [num_relations]，仅 use_semantic_att=True 时有值
        self.last_alpha: torch.Tensor | None = None

        # ── 关系特定投影矩阵 ─────────────────────────────────────────────────
        # W_r[r] 把节点特征从 in_dim 投影到 head_dim（concat 维度）
        # 每种关系独立，使得同一节点在不同关系语境下呈现不同侧面
        self.W_r = Parameter(torch.empty(num_relations, in_dim, self.head_dim))

        # ── 关系特定注意力参数 ───────────────────────────────────────────────
        # a_r[r] 是长度 2*out_dim 的向量，与拼接后的边表示做点积
        # 得到该边在这种关系下"两端节点相互贡献程度"的原始分数
        self.a_r = Parameter(torch.empty(num_relations, 2 * out_dim))

        # ── 语义注意力（可选）───────────────────────────────────────────────
        # 在把 4 个通道的消息相加之前，先学一套软权重
        # semantic_proj 把 head_dim 压到 128 以减少噪声
        # semantic_q 是可学习的"查询向量"，与各通道的压缩表示做点积得到权重
        if use_semantic_att:
            self.semantic_proj = Linear(self.head_dim, 128)
            self.semantic_q = Parameter(torch.empty(1, 128))

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)

        # 初始化
        torch.nn.init.xavier_uniform_(self.W_r)
        torch.nn.init.xavier_uniform_(self.a_r)
        if use_semantic_att:
            torch.nn.init.xavier_uniform_(self.semantic_proj.weight)
            torch.nn.init.xavier_uniform_(self.semantic_q)

    def forward(
        self,
        x: torch.Tensor,           # [N, in_dim] 节点特征
        edge_index: torch.Tensor,  # [2, E] 边索引
        edge_type: torch.Tensor,   # [E]   边类型（0/1/2/3）
    ) -> torch.Tensor:
        N = x.shape[0]
        num_relations = self.W_r.shape[0]

        # ── Step1：投影 ─────────────────────────────────────────────────────
        # h_r[n, r] = x[n] @ W_r[r]，形状 [N, num_relations, head_dim]
        # einsum 同时对所有节点和所有关系做批量矩阵乘法
        h_r = torch.einsum("ni,rio->nro", x, self.W_r)

        # 拆成多头格式，方便后续按头计算注意力
        # 形状 [N, num_relations, num_heads, out_dim]
        h_r_heads = h_r.view(N, num_relations, self.num_heads, self.out_dim)

        # 存放每个关系通道的聚合结果
        # 形状 [N, num_relations, num_heads, out_dim]
        out_per_rel = torch.zeros(N, num_relations, self.num_heads, self.out_dim, device=x.device)

        # ── Step2：按关系分组，计算注意力 + 消息传递 ──────────────────────────
        for r in range(num_relations):
            # 筛选属于关系 r 的边
            mask_r = edge_type == r
            if not mask_r.any():
                # 当前图中没有这种类型的边，直接跳过
                continue

            edge_r = edge_index[:, mask_r]  # [2, E_r]
            E_r = edge_r.size(1)

            # 取出边两端节点在关系 r 下的投影表示
            # h_src/h_dst 形状 [E_r, num_heads, out_dim]
            h_src = h_r_heads[edge_r[0], r]
            h_dst = h_r_heads[edge_r[1], r]

            # ── 注意力分数 ────────────────────────────────────────────────
            # 拼接 src 和 dst 的投影表示：[E_r, num_heads, 2*out_dim]
            concat_feat = torch.cat([h_src, h_dst], dim=-1)
            # 展平到 [E_r*num_heads, 2*out_dim]，与 a_r[r] 做点积
            att_input = concat_feat.view(-1, 2 * self.out_dim) @ self.a_r[r]
            # LeakyReLU 后 reshape 回 [E_r, num_heads, 1]
            att = self.leaky_relu(att_input).view(E_r, self.num_heads, 1)

            # ── softmax 归一化（按目标节点分组）──────────────────────────
            # 对每个目标节点（dst），把所有指向它的同类型边做 softmax
            exp_att = torch.exp(att.clamp(max=10)).squeeze(-1)  # [E_r, num_heads]
            # 计算每个目标节点的 exp 之和
            sum_exp = torch.zeros(N, self.num_heads, device=x.device)
            sum_exp.scatter_add_(
                0,
                edge_r[1].unsqueeze(-1).expand(E_r, self.num_heads),
                exp_att,
            )
            sum_exp = sum_exp + 1e-12  # 防止除零
            # 归一化得到注意力权重 [E_r, num_heads]
            alpha = exp_att / sum_exp[edge_r[1]]
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            # ── 消息 × 权重，scatter_add 到目标节点 ─────────────────────
            # message 形状 [E_r, num_heads, out_dim]
            message = alpha.unsqueeze(-1) * h_src
            out_rel = torch.zeros(N, self.num_heads, self.out_dim, device=x.device)
            expand_idx = (
                edge_r[1]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(E_r, self.num_heads, self.out_dim)
            )
            out_rel.scatter_add_(0, expand_idx, message)
            out_per_rel[:, r] = out_rel  # 存入对应通道

        # ── Step3：多通道融合 ───────────────────────────────────────────────
        if self.use_semantic_att:
            # 语义注意力：为每个节点学习 4 个通道的软权重
            # out_per_rel + h_r 残差，形状 [N, num_relations, head_dim]
            rel_contrib = out_per_rel.view(N, num_relations, self.head_dim) + h_r

            # 压缩到 128 维：[N*num_relations, 128]
            rel_proj = F.elu(
                self.semantic_proj(rel_contrib.contiguous().view(-1, self.head_dim))
            ).view(N, num_relations, 128)

            # 与查询向量点积得 [N, num_relations, 1]，softmax 归一化
            semantic_att = rel_proj @ self.semantic_q.t()
            semantic_alpha = F.softmax(semantic_att, dim=1)  # 4 个通道权重之和=1

            # 记录所有节点的平均权重，供外部观察哪种关系更重要
            self.last_alpha = semantic_alpha.mean(dim=0).squeeze(-1)  # [num_relations]

            # 加权求和各通道：[N, head_dim]
            out = (rel_contrib * semantic_alpha).sum(dim=1)

        else:
            # 简单融合：4 个通道直接相加 + 残差
            out_heads = out_per_rel.sum(dim=1)              # [N, num_heads, out_dim]
            residual = h_r_heads.mean(dim=1)               # [N, num_heads, out_dim]
            out_heads = out_heads + residual
            out = out_heads.view(N, self.head_dim)         # [N, head_dim]

        # ── Step4：输出格式 ────────────────────────────────────────────────
        if self.concat:
            # 中间层：保留 concat 维度 [N, head_dim]，供下一层使用
            return out
        else:
            # 最后一层：多头 mean，输出 [N, out_dim]
            return out.view(N, self.num_heads, self.out_dim).mean(dim=1)


class HetGAT(torch.nn.Module):
    """
    多层异构图注意力网络（Heterogeneous Graph Attention Network）。

    层结构由 HetGATConfig 的两个等长列表决定：
        hidden_dims[i]：第 i 层每个头的输出维度
        num_heads[i]  ：第 i 层的头数

    维度变化示意（默认 hidden_dims=[64,32], num_heads=[4,4]）：
        输入      768d
        Layer 0   64×4 = 256d  (concat=True)
        ELU
        Layer 1   32d          (concat=False, mean)  ← 最终嵌入

    Parameters
    ----------
    in_dim : int
        输入特征维度，等于 TextEncoder 输出维度（768）。
    cfg : HetGATConfig
        超参配置，包含 hidden_dims / num_heads / dropout / use_semantic_att。
    """

    def __init__(self, in_dim: int, cfg: HetGATConfig | None = None):
        super().__init__()
        cfg = cfg or HetGATConfig()

        # 校验：两个列表必须等长且至少有 1 层
        assert len(cfg.hidden_dims) == len(cfg.num_heads), (
            f"hidden_dims 和 num_heads 长度不一致："
            f"{len(cfg.hidden_dims)} vs {len(cfg.num_heads)}"
        )
        assert len(cfg.hidden_dims) >= 1, "至少需要 1 层"

        num_layers = len(cfg.hidden_dims)
        self.layers = torch.nn.ModuleList()

        for i in range(num_layers):
            is_last = i == num_layers - 1

            # 计算当前层的输入维度
            if i == 0:
                cur_in = in_dim
            else:
                # 上一层是 concat，输出维度 = hidden_dims[i-1] × num_heads[i-1]
                cur_in = cfg.hidden_dims[i - 1] * cfg.num_heads[i - 1]

            self.layers.append(
                HetGATConv(
                    in_dim=cur_in,
                    out_dim=cfg.hidden_dims[i],
                    num_heads=cfg.num_heads[i],
                    num_relations=NUM_RELATIONS,
                    dropout=cfg.dropout,
                    concat=not is_last,          # 中间层 concat，最后层 mean
                    use_semantic_att=cfg.use_semantic_att,
                )
            )

    def forward(
        self,
        x: torch.Tensor,           # [N, in_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_type: torch.Tensor,   # [E]
    ) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            形状 [N, hidden_dims[-1]]，即最终节点嵌入。
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type)
            # 中间层加 ELU 激活；最后一层不加（分数计算用原始值）
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return x


# ── 独立测试入口 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    from config import EncoderConfig, HetGATConfig
    from TextEncoder import TextEncoder
    from GraphDataset import GraphDataset

    _ROOT = Path(__file__).parent.parent
    graph_dir = _ROOT / "data" / "output" / "doc_results"

    # 加载 embedding 和图结构
    encoder = TextEncoder(EncoderConfig())
    embeddings = encoder.encode_all(graph_dir)
    dataset = GraphDataset(graph_dir, embeddings)

    # 构建模型（默认配置：2层，4头）
    cfg = HetGATConfig()
    model = HetGAT(in_dim=768, cfg=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    print(f"\n=== HetGAT 前向验证 ===")
    print(f"配置：hidden_dims={cfg.hidden_dims}, num_heads={cfg.num_heads}")
    print(f"期望输出维度：{cfg.hidden_dims[-1]}d\n")

    with torch.no_grad():
        for data in dataset:
            x = data.x.to(device)
            ei = data.edge_index.to(device)
            et = data.edge_type.to(device)

            z = model(x, ei, et)

            # 检查语义注意力权重（最后一层）
            last_layer = model.layers[-1]
            alpha_str = ""
            if cfg.use_semantic_att and last_layer.last_alpha is not None:
                w = last_layer.last_alpha.cpu().numpy()
                alpha_str = (
                    f"  semantic_alpha: "
                    f"T_fwd={w[0]:.3f} T_rev={w[1]:.3f} "
                    f"C_fwd={w[2]:.3f} C_rev={w[3]:.3f}"
                )

            print(
                f"  {data.doc_id:8s}  "
                f"输入[{x.shape[0]}, 768] → 输出{list(z.shape)}"
                f"{alpha_str}"
            )
