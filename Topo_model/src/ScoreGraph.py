from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from config import ModelConfig
from FeatureEngineering import load_graph, compute_features
from Dataset import build_edge_index
from Model import TopoModel


# ── 1. 加载模型 ────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
) -> tuple[TopoModel, ModelConfig]:
    """
    从 best_weights.pth + best_meta.json 重建并加载模型。

    checkpoint_path 可以是权重文件本身，也可以是其所在目录
    （目录时自动查找 best_weights.pth，不存在则退而用 latest_weights.pth）。

    Returns:
        model     : 已加载权重、已设 eval() 的 TopoModel
        model_cfg : 对应的 ModelConfig（从同目录 *_meta.json 读取）
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        for candidate in ("best_weights.pth", "latest_weights.pth"):
            p = checkpoint_path / candidate
            if p.exists():
                checkpoint_path = p
                break
        else:
            raise FileNotFoundError(f"在 {checkpoint_path} 中找不到 .pth 权重文件")

    # 同目录下查找 meta JSON（best_meta → latest_meta → 任意 *meta*.json）
    ckpt_dir = checkpoint_path.parent
    model_cfg = ModelConfig()
    for meta_name in ("best_meta.json", "latest_meta.json"):
        meta_path = ckpt_dir / meta_name
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            # meta 中若保存了 feat_dim 则覆盖默认值
            if "feat_dim" in meta:
                model_cfg.feat_dim = int(meta["feat_dim"])
            if "feature_out_activation" in meta:
                model_cfg.feature_out_activation = str(meta["feature_out_activation"])
            if "feature_out_leaky_slope" in meta:
                model_cfg.feature_out_leaky_slope = float(meta["feature_out_leaky_slope"])
            if "scorer_output_norm" in meta:
                model_cfg.scorer_output_norm = str(meta["scorer_output_norm"])
            break

    model = TopoModel(cfg=model_cfg)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True),
        strict=False,
    )
    model.to(device)
    model.eval()
    return model, model_cfg


# ── 2. 单图推理 ────────────────────────────────────────────────────────────────

def score_one(
    graph_path: Path,
    model: TopoModel,
    device: torch.device | str = "cpu",
) -> dict:
    """
    对单个 graph.json 做推理，实时计算特征（不依赖预存 features.npy）。

    Returns:
        {
          "doc_id"     : str,
          "graph_path" : str,
          "num_nodes"  : int,
          "num_edges"  : int,
          "scores"     : {node_id: float}  原始预测分（未排名）
        }
    """
    graph_path = Path(graph_path)
    G, node_ids = load_graph(graph_path)
    n = len(node_ids)

    if n == 0:
        return {
            "doc_id":     graph_path.parent.name,
            "graph_path": str(graph_path),
            "num_nodes":  0,
            "num_edges":  0,
            "scores":     {},
        }

    matrix, _ = compute_features(G, node_ids)                              # [N, 8] float32，已归一化
    features   = torch.from_numpy(matrix).to(device)                       # [N, 8]
    edge_index = torch.from_numpy(build_edge_index(graph_path, node_ids)).to(device)  # [2, E]

    with torch.no_grad():
        preds = model(features, edge_index)               # [N]

    scores = {nid: float(preds[i].item()) for i, nid in enumerate(node_ids)}
    num_edges = int(edge_index.shape[1]) if edge_index.numel() > 0 else 0

    return {
        "doc_id":     graph_path.parent.name,
        "graph_path": str(graph_path),
        "num_nodes":  n,
        "num_edges":  num_edges,
        "scores":     scores,
    }


# ── 3. 排名转换 ────────────────────────────────────────────────────────────────

def rank_scores(scores_dict: dict[str, float]) -> dict[str, dict]:
    """
    将原始分转为 {node_id: {score, rank}} 格式。
    rank=1 表示最重要，分值相同的节点共享最高名次（dense rank）。

    Args:
        scores_dict : {node_id: float}

    Returns:
        {node_id: {"score": float, "rank": int}}
    """
    if not scores_dict:
        return {}

    # 去重分值排序（降序）
    sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    result: dict[str, dict] = {}
    rank = 1
    prev_score: float | None = None
    for i, (nid, score) in enumerate(sorted_items):
        if prev_score is not None and score < prev_score:
            rank = i + 1
        result[nid] = {"score": score, "rank": rank}
        prev_score = score

    return result


# ── 4. 批量推理 ────────────────────────────────────────────────────────────────

def score_all(
    graph_root: Path,
    checkpoint_path: Path,
    output_root: Path,
    device: torch.device | str = "cpu",
    logger: logging.Logger | None = None,
    doc_ids: set[str] | None = None,
) -> list[dict]:
    """
    遍历 graph_root 下所有 graph.json，逐图推理并写出结果。

    每个文档输出到 output_root/{doc_id}/topology_scores.json，格式：
    {
      "doc_id"     : str,
      "graph_path" : str,
      "num_nodes"  : int,
      "num_edges"  : int,
      "scores"     : {node_id: {"score": float, "rank": int}}
    }

    Returns:
        结果列表（每元素对应一个文档的输出 dict）
    """
    log = logger or logging.getLogger("score_all")
    model, _ = load_model(checkpoint_path, device)

    graph_paths = sorted(Path(graph_root).rglob("graph.json"))
    if doc_ids is not None:
        graph_paths = [p for p in graph_paths if p.parent.name in doc_ids]
    if not graph_paths:
        log.warning("在 %s 下未找到 graph.json", graph_root)
        return []

    results: list[dict] = []
    for gp in graph_paths:
        doc_id = gp.parent.name
        try:
            raw = score_one(gp, model, device)
            raw["scores"] = rank_scores(raw["scores"])

            out_path = output_root / doc_id / "topology_scores.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log.info("scored %s  nodes=%d  → %s", doc_id, raw["num_nodes"], out_path)
            results.append(raw)
        except Exception as exc:
            log.error("scored %s FAILED: %s", doc_id, exc)

    return results


# ── 命令行入口 ─────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用训练好的 TopoModel 对 graph.json 进行拓扑评分"
    )
    base_default = Path(__file__).resolve().parents[1]

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=base_default / "output" / "models" / "final",
        help="权重文件路径 (.pth) 或包含 best_weights.pth 的目录，默认 output/models/final",
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=base_default / "data" / "output" / "doc_results",
        help="批量模式：递归扫描 graph.json 的根目录",
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=None,
        help="单图模式：直接指定一个 graph.json 路径（优先于 --graph-root）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_default / "output" / "topology_scores",
        help="评分结果写出目录，默认 output/topology_scores",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="推理设备，如 cpu / cuda / cuda:0，默认 cpu",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    logger = logging.getLogger("score")
    device = torch.device(args.device)

    if args.graph_path is not None:
        # ── 单图模式 ──
        model, _ = load_model(args.checkpoint, device)
        raw = score_one(args.graph_path, model, device)
        raw["scores"] = rank_scores(raw["scores"])

        out_path = args.output_dir / raw["doc_id"] / "topology_scores.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("scored %s  nodes=%d  → %s", raw["doc_id"], raw["num_nodes"], out_path)
    else:
        # ── 批量模式 ──
        results = score_all(
            graph_root=args.graph_root,
            checkpoint_path=args.checkpoint,
            output_root=args.output_dir,
            device=device,
            logger=logger,
        )
        logger.info("完成 %d 个文档评分", len(results))


if __name__ == "__main__":
    main()
