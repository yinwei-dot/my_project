from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List

import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# 四通道关系类型常量
TEMPORAL_FORWARD = 0
TEMPORAL_REVERSE = 1
CAUSAL_FORWARD = 2
CAUSAL_REVERSE = 3

# semantic_model/ 目录
_PROJECT_ROOT = Path(__file__).parent.parent


def _build_one(graph_path: Path, embeddings: Dict[str, torch.Tensor]) -> Data:
    """
    读取单个 graph.json，结合预计算的节点 embedding，构建 PyG Data 对象。

    边四通道：
        0 = temporal_forward   1 = temporal_reverse
        2 = causal_forward     3 = causal_reverse

    Parameters
    ----------
    graph_path : Path
        graph.json 的绝对路径。
    embeddings : Dict[str, torch.Tensor]
        该文档的 {event_id: Tensor[768]} 映射，来自 TextEncoder.encode_all()。

    Returns
    -------
    Data
        包含 x / edge_index / edge_type / node_ids /
        causal_edge_mask / temporal_edge_mask 的图对象。
    """
    raw = json.loads(graph_path.read_text(encoding="utf-8"))
    doc_id: str = raw["doc_id"]
    events: list = raw["events"]
    causal_edges: list = raw.get("causal_edges", [])
    temporal_edges: list = raw.get("temporal_edges", [])

    # ── 节点索引 ─────────────────────────────────────────────────────────────
    node_ids: List[str] = [e["event_id"] for e in events]
    id2idx: Dict[str, int] = {eid: i for i, eid in enumerate(node_ids)}
    N = len(node_ids)

    # ── 节点特征矩阵 [N, 768] ─────────────────────────────────────────────────
    x = torch.stack([embeddings[eid] for eid in node_ids])  # [N, 768]

    # ── 构边（正向 + 补反向） ────────────────────────────────────────────────
    src_list: List[int] = []
    dst_list: List[int] = []
    type_list: List[int] = []

    # 时序正向
    for e in temporal_edges:
        s, t = id2idx[e["source"]], id2idx[e["target"]]
        src_list.append(s);  dst_list.append(t);  type_list.append(TEMPORAL_FORWARD)

    # 时序反向
    for e in temporal_edges:
        s, t = id2idx[e["source"]], id2idx[e["target"]]
        src_list.append(t);  dst_list.append(s);  type_list.append(TEMPORAL_REVERSE)

    # 因果正向
    for e in causal_edges:
        s, t = id2idx[e["cause"]], id2idx[e["effect"]]
        src_list.append(s);  dst_list.append(t);  type_list.append(CAUSAL_FORWARD)

    # 因果反向
    for e in causal_edges:
        s, t = id2idx[e["cause"]], id2idx[e["effect"]]
        src_list.append(t);  dst_list.append(s);  type_list.append(CAUSAL_REVERSE)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)  # [2, E_total]
    edge_type  = torch.tensor(type_list, dtype=torch.long)             # [E_total]

    # ── 辅助掩码 ─────────────────────────────────────────────────────────────
    causal_edge_mask   = (edge_type == CAUSAL_FORWARD)   | (edge_type == CAUSAL_REVERSE)
    temporal_edge_mask = (edge_type == TEMPORAL_FORWARD) | (edge_type == TEMPORAL_REVERSE)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=N,
        doc_id=doc_id,
        node_ids=node_ids,
        causal_edge_mask=causal_edge_mask,
        temporal_edge_mask=temporal_edge_mask,
    )


class GraphDataset:
    """
    加载全部文档的事件图，对外暴露列表接口。

    Parameters
    ----------
    graph_dir : str | Path
        doc_results/ 目录，每个子目录包含一个 graph.json。
    embeddings : Dict[str, Dict[str, torch.Tensor]]
        TextEncoder.encode_all() 的返回值：{doc_id: {event_id: Tensor[768]}}。

    Usage
    -----
    dataset = GraphDataset(graph_dir, embeddings)
    for data in dataset:
        # data.x           [N, 768]
        # data.edge_index  [2, E_total]
        # data.edge_type   [E_total]  (0/1/2/3)
        # data.node_ids    ["E1", "E2", ...]
        # data.causal_edge_mask    bool Tensor [E_total]
        # data.temporal_edge_mask  bool Tensor [E_total]
        ...
    """

    def __init__(
        self,
        graph_dir: str | Path,
        embeddings: Dict[str, Dict[str, torch.Tensor]],
    ) -> None:
        graph_dir = Path(graph_dir)
        self._graphs: List[Data] = []

        doc_dirs = sorted(p for p in graph_dir.iterdir() if p.is_dir())
        for doc_dir in doc_dirs:
            graph_path = doc_dir / "graph.json"
            if not graph_path.exists():
                logger.warning("[跳过] %s 下没有 graph.json", doc_dir.name)
                continue
            doc_id = doc_dir.name
            if doc_id not in embeddings:
                logger.warning("[跳过] %s 没有对应的 embedding，请先运行 TextEncoder", doc_id)
                continue
            data = _build_one(graph_path, embeddings[doc_id])
            self._graphs.append(data)

        logger.info("[GraphDataset] 已加载 %d 个文档图", len(self._graphs))

    # ── 列表接口 ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> Data:
        return self._graphs[idx]

    def __iter__(self) -> Iterator[Data]:
        return iter(self._graphs)

    def doc_ids(self) -> List[str]:
        """返回所有已加载文档的 doc_id 列表。"""
        return [g.doc_id for g in self._graphs]


# ── 独立测试入口 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from TextEncoder import TextEncoder
    from config import EncoderConfig

    graph_dir = _PROJECT_ROOT / "data" / "output" / "doc_results"

    encoder = TextEncoder(EncoderConfig())
    embeddings = encoder.encode_all(graph_dir)

    dataset = GraphDataset(graph_dir, embeddings)

    print(f"\n=== GraphDataset 构图摘要 ===")
    for data in dataset:
        n = data.num_nodes
        e = data.edge_index.shape[1]
        nc = data.causal_edge_mask.sum().item()
        nt = data.temporal_edge_mask.sum().item()
        print(
            f"  {data.doc_id:8s}  nodes={n:2d}  "
            f"edges={e:2d}  (causal×2={int(nc):2d}, temporal×2={int(nt):2d})"
        )
