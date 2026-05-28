from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

from config import AugmentorConfig

logger = logging.getLogger(__name__)

# 四通道关系类型（与 GraphDataset 保持一致）
TEMPORAL_FORWARD = 0
TEMPORAL_REVERSE = 1
CAUSAL_FORWARD   = 2
CAUSAL_REVERSE   = 3


class Augmentor:
    """
    图增强器：产生 View 1（语义鲁棒性）和 View 2（因果偏置）的增强视图。

    View 1
    ------
    特征层：对节点特征 x 随机置零（feat_mask_rate_v1 比例的维度）。
    结构层：对四种边类型各自独立以 temporal_drop_v1 / causal_drop_v1 概率删边。
    调用两次 augment_v1() 即得到独立的 V1a 和 V1b。

    View 2（因果偏置视图）
    ----------------------
    特征层：不掩码，保留原始语义。
    结构层：时序边大比例删除（temporal_drop_v2），因果边小比例删除（causal_drop_v2）。
    使得因果邻域丰富的节点在 V2 里仍有稳定表示，时序尾链节点则趋向孤立。

    边 dropout 按 edge_type 分组独立采样，正向和反向边分别处理，
    避免单向边残留产生虚假的方向偏置信号。
    """

    def __init__(self, cfg: AugmentorConfig | None = None) -> None:
        self.cfg = cfg or AugmentorConfig()

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _drop_by_type(
        edge_index: torch.Tensor,          # [2, E]
        edge_type:  torch.Tensor,          # [E]
        drop_rates: dict[int, float],      # {rel_type: 删除概率}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        按 edge_type 分组，对每组独立 Bernoulli 采样后合并剩余边。

        对每种关系类型，在该类型的所有边上独立抽样：
        torch.rand(E_r) >= drop_rate → True 表示保留。
        未出现在 drop_rates 里的边类型一律保留。
        """
        if edge_type.numel() == 0:
            return edge_index, edge_type

        keep = torch.ones(edge_type.size(0), dtype=torch.bool, device=edge_type.device)
        for rel, rate in drop_rates.items():
            type_mask = edge_type == rel
            if not type_mask.any():
                continue
            idx = type_mask.nonzero(as_tuple=False).squeeze(1)  # 该类型边的下标
            keep_rel = torch.rand(idx.size(0), device=edge_type.device) >= rate
            keep[idx] = keep_rel

        return edge_index[:, keep], edge_type[keep]

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def augment_v1(self, data: Data) -> Data:
        """
        View 1 单次增强：特征掩码 + 等比例边 dropout。

        每次调用独立随机 → 连续调用两次即得到 V1a 和 V1b 两个不同视图。
        返回的 Data 只含 x / edge_index / edge_type / num_nodes，
        不含 doc_id / node_ids（增强视图不需要这些元信息）。
        """
        cfg = self.cfg

        # 特征掩码：随机将 feat_mask_rate_v1 比例的维度置零
        x = data.x.clone()
        mask = torch.rand_like(x) < cfg.feat_mask_rate_v1
        x[mask] = 0.0

        # 边 dropout（四通道等比例）
        ei, et = self._drop_by_type(
            data.edge_index, data.edge_type,
            {
                TEMPORAL_FORWARD: cfg.temporal_drop_v1,
                TEMPORAL_REVERSE: cfg.temporal_drop_v1,
                CAUSAL_FORWARD:   cfg.causal_drop_v1,
                CAUSAL_REVERSE:   cfg.causal_drop_v1,
            },
        )
        return Data(x=x, edge_index=ei, edge_type=et, num_nodes=data.num_nodes)

    def augment_v2(self, data: Data) -> Data:
        """
        View 2（因果偏置）增强：保留原始特征，时序边大比例删除，因果边轻度保留。

        让因果边丰富的节点在 V2 里仍有稳定的因果邻域，
        仅有时序边的节点（如游离事件）在 V2 里趋向孤立。
        """
        cfg = self.cfg

        # 特征不掩码
        x = data.x.clone()

        # 边 dropout（时序边大比例，因果边小比例）
        ei, et = self._drop_by_type(
            data.edge_index, data.edge_type,
            {
                TEMPORAL_FORWARD: cfg.temporal_drop_v2,
                TEMPORAL_REVERSE: cfg.temporal_drop_v2,
                CAUSAL_FORWARD:   cfg.causal_drop_v2,
                CAUSAL_REVERSE:   cfg.causal_drop_v2,
            },
        )
        return Data(x=x, edge_index=ei, edge_type=et, num_nodes=data.num_nodes)

    def augment(self, data: Data) -> tuple[Data, Data, Data]:
        """
        返回三个增强视图 (v1a, v1b, v2)。

        v1a 和 v1b 是两次独立的 augment_v1()，随机种子不同。
        v1b 同时用于 L_robust（与 v1a 配对）和 L_causal（与 v2 配对），减少前向次数。
        """
        return self.augment_v1(data), self.augment_v1(data), self.augment_v2(data)


# ── 独立测试入口 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

    from config import EncoderConfig, AugmentorConfig
    from TextEncoder import TextEncoder
    from GraphDataset import GraphDataset

    _ROOT = Path(__file__).parent.parent
    graph_dir = _ROOT / "data" / "output" / "doc_results"

    encoder = TextEncoder(EncoderConfig())
    embeddings = encoder.encode_all(graph_dir)
    dataset = GraphDataset(graph_dir, embeddings)

    aug = Augmentor(AugmentorConfig())

    import unicodedata

    def _cnt(et: torch.Tensor, t: int) -> int:
        return int((et == t).sum().item())

    def _rw(s: str, width: int) -> str:
        """右对齐到 `width` 显示列（CJK 字符按 2 计）。"""
        disp = sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)
        return " " * max(0, width - disp) + s

    header = (
        f"\n{_rw('doc_id',8)}  {_rw('N',2)}  "
        f"{_rw('原始',8)}  {_rw('V1a',8)}  "
        f"{_rw('V1b',8)}  {_rw('V2',8)}  "
        f"{_rw('V1a掩码率',9)}"
    )
    print(header)
    print("-" * 72)
    for data in dataset:
        v1a, v1b, v2 = aug.augment(data)

        orig = f"t{_cnt(data.edge_type,0)}+{_cnt(data.edge_type,2)}c"
        va   = f"t{_cnt(v1a.edge_type,0)}+{_cnt(v1a.edge_type,2)}c"
        vb   = f"t{_cnt(v1b.edge_type,0)}+{_cnt(v1b.edge_type,2)}c"
        v2s  = f"t{_cnt(v2.edge_type,0)}+{_cnt(v2.edge_type,2)}c"
        zero_rate = (v1a.x == 0).float().mean().item()

        print(f"{data.doc_id:>8}  {data.num_nodes:>2}  {orig:>8}  {va:>8}  {vb:>8}  {v2s:>8}  {zero_rate:>9.1%}")
