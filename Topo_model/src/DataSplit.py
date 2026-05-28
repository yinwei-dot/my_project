from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from config import TopoConfig, TrainConfig

MANIFEST_NAME = "split_manifest.json"


def count_nodes_from_graph(graph_path: Path) -> int:
    """从 graph.json 统计有效事件节点数。"""
    with open(graph_path, encoding="utf-8") as f:
        payload = json.load(f)
    return len([e for e in payload.get("events", []) if e.get("event_id")])


def scan_graph_sizes(graph_root: Path) -> dict[str, tuple[int, Path]]:
    """
    扫描 graph_root 下全部 graph.json。

    Returns:
        {doc_id: (num_nodes, graph_path)}
    """
    result: dict[str, tuple[int, Path]] = {}
    for graph_path in sorted(graph_root.rglob("graph.json")):
        doc_id = graph_path.parent.name
        result[doc_id] = (count_nodes_from_graph(graph_path), graph_path)
    return result


def load_manifest(manifest_path: Path) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest_path: Path, data: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def assign_train_test(
    doc_ids: list[str],
    train_cfg: TrainConfig,
) -> tuple[list[str], list[str]]:
    """将 doc_id 列表按 test_ratio 划分为 train / test（可复现）。"""
    rng = random.Random(train_cfg.random_seed)
    shuffled = list(doc_ids)
    rng.shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * train_cfg.test_ratio)) if shuffled else 0
    if len(shuffled) > 1 and n_test >= len(shuffled):
        n_test = 1
    test_ids = sorted(shuffled[:n_test])
    train_ids = sorted(shuffled[n_test:])
    return train_ids, test_ids


def preview_split_from_graphs(
    graph_root: Path,
    topo_cfg: TopoConfig,
    train_cfg: TrainConfig,
) -> tuple[int, int, int, int]:
    """
    扫描全量图，返回预览统计（不依赖标签是否已生成）。

    Returns:
        (skip_lt3, skip_gt30, n_train, n_test)
    """
    graph_map = scan_graph_sizes(graph_root)
    labeled_candidates: list[str] = []
    skip_lt3 = skip_gt30 = 0

    for doc_id, (n, _) in graph_map.items():
        if n < topo_cfg.min_nodes:
            skip_lt3 += 1
        elif n > topo_cfg.labeled_max_nodes:
            skip_gt30 += 1
        else:
            labeled_candidates.append(doc_id)

    train_ids, test_ids = assign_train_test(labeled_candidates, train_cfg)
    return skip_lt3, skip_gt30, len(train_ids), len(test_ids)


def build_split_manifest(
    graph_root: Path,
    label_root: Path,
    manifest_path: Path,
    topo_cfg: TopoConfig,
    train_cfg: TrainConfig,
    *,
    failed: list[str] | None = None,
) -> dict:
    """
    根据全量图节点数 + 已生成标签，构建数据划分清单。

    规则：
      - n < min_nodes          → skipped_lt3
      - n > labeled_max_nodes  → inference_only_gt30（不生成标签）
      - min_nodes ≤ n ≤ max    且有 topo_labels.json → labeled 池，划分 train/test
      - 应有标签但缺失/失败     → failed
    """
    failed_set = set(failed or [])
    graph_map = scan_graph_sizes(graph_root)

    skipped_lt3: list[str] = []
    inference_only_gt30: list[str] = []
    labeled_candidates: list[str] = []

    for doc_id, (n, _) in sorted(graph_map.items()):
        if n < topo_cfg.min_nodes:
            skipped_lt3.append(doc_id)
        elif n > topo_cfg.labeled_max_nodes:
            inference_only_gt30.append(doc_id)
        else:
            labeled_candidates.append(doc_id)

    labeled_success: list[str] = []
    labeled_missing: list[str] = []
    for doc_id in labeled_candidates:
        if doc_id in failed_set:
            labeled_missing.append(doc_id)
            continue
        label_path = label_root / doc_id / "topo_labels.json"
        if label_path.exists():
            labeled_success.append(doc_id)
        else:
            labeled_missing.append(doc_id)

    train_ids, test_ids = assign_train_test(labeled_success, train_cfg)

    score_ids = sorted(
        doc_id for doc_id, (n, _) in graph_map.items() if n >= topo_cfg.min_nodes
    )

    manifest = {
        "total_graphs": len(graph_map),
        "min_nodes": topo_cfg.min_nodes,
        "labeled_max_nodes": topo_cfg.labeled_max_nodes,
        "test_ratio": train_cfg.test_ratio,
        "random_seed": train_cfg.random_seed,
        "skipped_lt3": sorted(skipped_lt3),
        "inference_only_gt30": sorted(inference_only_gt30),
        "failed": sorted(labeled_missing),
        "labeled": {
            "train": train_ids,
            "test": test_ids,
        },
        "score_ids": score_ids,
        "stats": {
            "skipped_lt3": len(skipped_lt3),
            "inference_only_gt30": len(inference_only_gt30),
            "failed": len(labeled_missing),
            "labeled_train": len(train_ids),
            "labeled_test": len(test_ids),
            "labeled_total": len(labeled_success),
            "score_total": len(score_ids),
        },
    }
    save_manifest(manifest_path, manifest)
    return manifest


def get_doc_ids(
    manifest: dict,
    *,
    split: Literal["train", "test", "labeled", "score", "inference_only"] | None = None,
) -> list[str]:
    """从 manifest 取 doc_id 列表。"""
    if split is None:
        return []
    if split == "train":
        return list(manifest["labeled"]["train"])
    if split == "test":
        return list(manifest["labeled"]["test"])
    if split == "labeled":
        return sorted(manifest["labeled"]["train"] + manifest["labeled"]["test"])
    if split == "score":
        return list(manifest["score_ids"])
    if split == "inference_only":
        return list(manifest["inference_only_gt30"])
    raise ValueError(f"未知 split: {split}")
