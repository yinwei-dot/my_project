from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, cast

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from config import HetGATConfig, ScorerConfig
from HetGAT import HetGAT

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent


def _minmax_normalize(values: torch.Tensor) -> torch.Tensor:
    """逐文档 min-max 归一化到 [0, 1]。"""
    v_min = values.min()
    v_max = values.max()
    denom = v_max - v_min
    if denom.abs() < 1e-8:
        return torch.zeros_like(values)
    return (values - v_min) / denom


class SemanticScorer:
    """
    语义评分器：加载训练好的 HetGAT，在原始图上推理，生成节点语义重要性评分。

    评分公式
    --------
    s_rich(i) = ‖z_i‖₂
        丰富性：节点嵌入的 L2 范数。VICReg 训练后，因果中心节点邻域稳定
        → 嵌入收敛 → 模长大；游离节点邻域在 V2 里不稳定 → 模长小。

    s_rare(i) = mean_{j≠i}(1 - cos(z_i, z_j))
        稀缺性：节点与其他节点的平均余弦距离。语义独特的节点距离远 → 分高。

    score(i) = w_rich × s_rich(i) + w_rare × s_rare(i)
        per-document min-max 归一化到 [0, 1]。

    注
    --
    w_rare 建议保持较小（默认 0.3），防止 M009-E6（梵蒂冈，语义独特但游离）
    等无关事件因稀缺性被过度高估。
    """

    def __init__(self, hetgat: HetGAT, cfg: ScorerConfig | None = None) -> None:
        self.hetgat = hetgat
        self.cfg = cfg or ScorerConfig()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        in_dim: int = 768,
        hetgat_cfg: HetGATConfig | None = None,
        scorer_cfg: ScorerConfig | None = None,
    ) -> "SemanticScorer":
        """
        从 checkpoint 文件（或目录）加载 HetGAT 权重并返回 SemanticScorer。

        checkpoint 格式：{"encoder": state_dict}（由 TrainModel 保存的 best_weights.pth）。
        传入目录时自动查找 best_weights.pth，不存在则退而用 latest_weights.pth。
        """
        hetgat_cfg = hetgat_cfg or HetGATConfig()
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_dir():
            for name in ("best_weights.pth", "latest_weights.pth"):
                p = checkpoint_path / name
                if p.exists():
                    checkpoint_path = p
                    break
            else:
                raise FileNotFoundError(f"在 {checkpoint_path} 中找不到 .pth 权重文件")

        model = HetGAT(in_dim, hetgat_cfg)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # best_weights.pth 存的是 {"encoder": state_dict}
        if isinstance(state, dict) and "encoder" in state:
            model.load_state_dict(state["encoder"])
        else:
            model.load_state_dict(state)

        model.eval()
        logger.info("[SemanticScorer] 已从 %s 加载权重", checkpoint_path.name)
        return cls(model, scorer_cfg)

    # ── 评分 ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score_details(self, data: Data) -> dict:
        """返回最终语义分以及丰富性/稀缺性分解。"""
        self.hetgat.eval()
        device = next(self.hetgat.parameters()).device
        x = cast(torch.Tensor, data.x)
        edge_index = cast(torch.Tensor, data.edge_index)
        edge_type = cast(torch.Tensor, data.edge_type)

        z = self.hetgat(
            x.to(device),
            edge_index.to(device),
            edge_type.to(device),
        )  # [N, hidden_dims[-1]]
        N = z.size(0)

        s_rich = z.norm(dim=1)

        if N > 1:
            z_norm = F.normalize(z, dim=1)
            cos_sim = z_norm @ z_norm.T
            cos_dist = 1.0 - cos_sim
            off_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
            s_rare = cos_dist[off_mask].view(N, N - 1).mean(dim=1)
        else:
            s_rare = torch.zeros(N, device=z.device)

        cfg = self.cfg
        s_raw = cfg.w_rich * s_rich + cfg.w_rare * s_rare

        rich_norm = _minmax_normalize(s_rich)
        rare_norm = _minmax_normalize(s_rare)
        score_norm = _minmax_normalize(s_raw)

        node_ids = list(data.node_ids)
        scores = {eid: round(float(score_norm[i]), 6) for i, eid in enumerate(node_ids)}
        rich_scores = {eid: round(float(rich_norm[i]), 6) for i, eid in enumerate(node_ids)}
        rare_scores = {eid: round(float(rare_norm[i]), 6) for i, eid in enumerate(node_ids)}
        ranking = sorted(scores, key=scores.__getitem__, reverse=True)

        relation_weights = None
        last_layer = self.hetgat.layers[-1] if getattr(self.hetgat, "layers", None) else None
        last_alpha = getattr(last_layer, "last_alpha", None)
        if last_alpha is not None:
            alpha = last_alpha.detach().cpu().tolist()
            labels = [
                "temporal_forward",
                "temporal_reverse",
                "causal_forward",
                "causal_reverse",
            ]
            relation_weights = {
                label: round(float(weight), 6)
                for label, weight in zip(labels, alpha)
            }

        return {
            "scores": scores,
            "ranking": ranking,
            "rich_scores": rich_scores,
            "rare_scores": rare_scores,
            "formula_weights": {
                "rich": float(cfg.w_rich),
                "rare": float(cfg.w_rare),
            },
            "relation_weights": relation_weights,
        }

    @torch.no_grad()
    def score(self, data: Data) -> Dict[str, float]:
        """
        单图评分。

        Returns
        -------
        {event_id: score}  score ∈ [0, 1]，per-document min-max 归一化
        """
        return self.score_details(data)["scores"]

    def score_all(
        self,
        dataset,              # GraphDataset
        output_dir: str | Path,
    ) -> Dict[str, Dict[str, float]]:
        """
        对全部文档评分，每个文档写入 output_dir/{doc_id}/semantic_scores.json。
        输出格式对标 Topo_model 的 topology_scores.json。

        Returns
        -------
        {doc_id: {event_id: score}}
        """
        output_dir = Path(output_dir)
        all_scores: Dict[str, Dict[str, float]] = {}

        for data in dataset:
            details = self.score_details(data)
            scores = details["scores"]
            ranking = details["ranking"]
            all_scores[data.doc_id] = scores

            doc_dir = output_dir / data.doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            out = {
                "doc_id":   data.doc_id,
                "num_nodes": data.num_nodes,
                "scores":   scores,
                "ranking":  ranking,
                "rich_scores": details["rich_scores"],
                "rare_scores": details["rare_scores"],
                "formula_weights": details["formula_weights"],
                "relation_weights": details["relation_weights"],
            }
            with open(doc_dir / "semantic_scores.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            logger.info("[SemanticScorer] %s → ranking %s", data.doc_id, ranking)

        return all_scores


# ── 独立测试入口 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

    from config import EncoderConfig, HetGATConfig, ScorerConfig
    from TextEncoder import TextEncoder
    from GraphDataset import GraphDataset

    _ROOT = Path(__file__).parent.parent
    graph_dir = _ROOT / "data" / "output" / "doc_results"

    print("[1] 加载 embedding 和图数据...")
    encoder = TextEncoder(EncoderConfig())
    embeddings = encoder.encode_all(graph_dir)
    dataset = GraphDataset(graph_dir, embeddings)

    print("[2] 使用随机初始化的 HetGAT 测试评分流程（未训练，仅验证形状和流程）...")
    hetgat = HetGAT(768, HetGATConfig())
    scorer = SemanticScorer(hetgat, ScorerConfig())

    for data in dataset:
        scores  = scorer.score(data)
        ranking = sorted(scores, key=scores.__getitem__, reverse=True)
        print(f"\n{data.doc_id}  (N={data.num_nodes})")
        print(f"  排名: {ranking}")
        for nid in ranking:
            print(f"    {nid}: {scores[nid]:.4f}")

    print("\n✓ SemanticScorer 流程正常（结果无意义，模型未训练）")
