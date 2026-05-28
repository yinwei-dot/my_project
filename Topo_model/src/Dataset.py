from __future__ import annotations

import json
import logging
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config import TopoConfig
from DataSplit import get_doc_ids, load_manifest


# ── 1. 读取特征 ───────────────────────────────────────────────────────────────

def load_features(feature_dir: Path) -> tuple[np.ndarray, list[str]]:
    """
    读取单文档的节点特征矩阵和节点 ID 顺序。

    Returns:
        matrix   : [N×8] float32，已归一化（由 FeatureEngineering.py 生成）
        node_ids : 节点 ID 列表，决定矩阵行顺序
    """
    matrix: np.ndarray = np.load(feature_dir / "features.npy")  # [N×8]
    with open(feature_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    node_ids: list[str] = meta["node_ids"]
    return matrix, node_ids


# ── 2. 读取 rank 标签 ─────────────────────────────────────────────────────────

def load_label_payload(label_dir: Path) -> dict:
    with open(label_dir / "topo_labels.json", encoding="utf-8") as f:
        return json.load(f)


def load_rank_labels(label_dir: Path, node_ids: list[str]) -> np.ndarray:
    """
    rank_labels：tier 台阶序（D≻ND；D 内 c0+度，ND 内 s_nd+度）。
    """
    payload = load_label_payload(label_dir)
    if "rank_labels" not in payload:
        raise KeyError(
            f"topo_labels.json 缺少 rank_labels，请重跑 CreateGroundTruth：{label_dir}"
        )
    rank_labels: dict[str, float] = payload["rank_labels"]
    return np.array([rank_labels[nid] for nid in node_ids], dtype=np.float32)


def load_member_labels(label_dir: Path, node_ids: list[str]) -> np.ndarray:
    """member_labels：拆解=1，非拆解=0（schema v2）。"""
    payload = load_label_payload(label_dir)
    if "member_labels" not in payload:
        raise KeyError(
            f"topo_labels.json 缺少 member_labels（旧格式），"
            f"请重跑 CreateGroundTruth：{label_dir}"
        )
    member: dict[str, float] = payload["member_labels"]
    return np.array([member[nid] for nid in node_ids], dtype=np.float32)


# ── 3. 构建有向 edge_index ────────────────────────────────────────────────────

def build_edge_index(graph_path: Path, node_ids: list[str]) -> np.ndarray:
    """
    从 graph.json 读取因果边和时序边，映射为整数索引的有向 edge_index。
    不做双向补全，保留有向图方向性。

    边类型：
      - 因果边：cause → effect
      - 时序边：source → target
    因果边与时序边可能重叠，最终去重。

    Returns:
        edge_index : [2, E] int64，E 为去重后的有向边数
    """
    with open(graph_path, encoding="utf-8") as f:
        payload = json.load(f)

    # 节点 ID → 行索引映射
    id2idx: dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}

    src_list: list[int] = []
    dst_list: list[int] = []

    # 因果边
    for edge in payload.get("causal_edges", []) or []:
        src = str(edge.get("cause", ""))
        dst = str(edge.get("effect", ""))
        if src in id2idx and dst in id2idx:
            src_list.append(id2idx[src])
            dst_list.append(id2idx[dst])

    # 时序边
    for edge in payload.get("temporal_edges", []) or []:
        src = str(edge.get("source", ""))
        dst = str(edge.get("target", ""))
        if src in id2idx and dst in id2idx:
            src_list.append(id2idx[src])
            dst_list.append(id2idx[dst])

    if not src_list:
        return np.empty((2, 0), dtype=np.int64)

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    # 去重（因果边和时序边可能存在重复）
    edge_index = np.unique(edge_index.T, axis=0).T
    return edge_index


# ── 4. 构建单文档样本 ─────────────────────────────────────────────────────────

def build_record(
    doc_id: str,
    feature_root: Path,
    label_root: Path,
    graph_path: Path,
) -> dict:
    """
    组装单个文档的训练样本（numpy 格式），TopoDataset.__getitem__ 负责转 Tensor。

    Returns:
        {
          "doc_id":     str,
          "node_ids":   list[str],        # 推理时还原节点 ID 用
          "num_nodes":  int,
          "features":   np.ndarray [N×8] float32,
          "labels":     np.ndarray [N]    float32  rank 台阶
          "member":     np.ndarray [N]    float32  拆解=1
          "edge_index": np.ndarray [2, E] int64,
        }
    """
    feature_dir = feature_root / doc_id
    label_dir   = label_root   / doc_id

    features, node_ids = load_features(feature_dir)
    labels     = load_rank_labels(label_dir, node_ids)
    member     = load_member_labels(label_dir, node_ids)
    edge_index = build_edge_index(graph_path, node_ids)

    return {
        "doc_id":     doc_id,
        "node_ids":   node_ids,
        "num_nodes":  len(node_ids),
        "features":   features,
        "labels":     labels,
        "member":     member,
        "edge_index": edge_index,
    }


# ── 多进程任务单元（模块级函数，可被 pickle）────────────────────────────────────

def _worker_task(args: tuple) -> tuple[str, Optional[dict], Optional[str]]:
    """子进程执行单元，返回 (doc_id, record_or_None, error_or_None)。"""
    doc_id, feature_root_str, label_root_str, graph_path_str = args
    try:
        record = build_record(
            doc_id,
            Path(feature_root_str),
            Path(label_root_str),
            Path(graph_path_str),
        )
        return doc_id, record, None
    except Exception as exc:
        return doc_id, None, str(exc)


# ── 5. 批量加载所有文档样本 ───────────────────────────────────────────────────

def load_all_records(
    feature_root: Path,
    label_root: Path,
    graph_root: Path,
    config: TopoConfig,
    logger: logging.Logger,
    doc_ids: list[str] | None = None,
) -> list[dict]:
    """
    并行批量加载文档样本，支持动态进度条和日志。

    扫描策略：
      - doc_ids 由 split_manifest 指定（train / test / labeled）
      - 从 graph_root rglob 构建 doc_id → graph_path 映射
      - 同时检查 label_root、feature_root 下对应资源是否存在

    Returns:
        records : list[dict]，每项为一个文档图样本（numpy 格式）
    """
    graph_map: dict[str, Path] = {
        p.parent.name: p for p in sorted(graph_root.rglob("graph.json"))
    }
    logger.info("扫描到 %d 个 graph.json，graph_root=%s", len(graph_map), graph_root)

    if doc_ids is None:
        doc_ids = []
        for label_dir in sorted(label_root.iterdir()):
            if not label_dir.is_dir():
                continue
            if (label_dir / "topo_labels.json").exists():
                doc_ids.append(label_dir.name)

    filtered: list[str] = []
    for doc_id in sorted(doc_ids):
        if not (label_root / doc_id / "topo_labels.json").exists():
            logger.warning("[%s] 缺少 topo_labels.json，跳过。", doc_id)
            continue
        if not (feature_root / doc_id / "features.npy").exists():
            logger.warning("[%s] 缺少 features.npy，跳过。", doc_id)
            continue
        if doc_id not in graph_map:
            logger.warning("[%s] 找不到对应 graph.json，跳过。", doc_id)
            continue
        filtered.append(doc_id)
    doc_ids = filtered

    total = len(doc_ids)
    logger.info("有效文档 %d 个，开始构建样本。", total)
    tqdm.write(f"找到 {total} 个有效文档")

    if total == 0:
        return []

    task_args = [
        (doc_id, str(feature_root), str(label_root), str(graph_map[doc_id]))
        for doc_id in doc_ids
    ]

    records: list[dict] = []
    failed = 0
    num_workers = min(config.resolved_workers(), total)

    if num_workers == 1:
        # 串行：方便调试
        with tqdm(total=total, unit="doc", dynamic_ncols=True, desc="构建样本") as pbar:
            for args in task_args:
                doc_id, record, error = _worker_task(args)
                pbar.set_description(doc_id)
                if error:
                    logger.error("[%s] 构建失败: %s", doc_id, error)
                    tqdm.write(f"  ERROR [{doc_id}]: {error}")
                    failed += 1
                else:
                    assert record is not None
                    records.append(record)
                    logger.info(
                        "[%s] 节点 %d 个，边 %d 条。",
                        doc_id, record["num_nodes"], record["edge_index"].shape[1],
                    )
                pbar.update(1)
    else:
        with mp.Pool(processes=num_workers) as pool:
            with tqdm(total=total, unit="doc", dynamic_ncols=True, desc="构建样本") as pbar:
                for doc_id, record, error in pool.imap_unordered(_worker_task, task_args):
                    pbar.set_description(doc_id)
                    if error:
                        logger.error("[%s] 构建失败: %s", doc_id, error)
                        tqdm.write(f"  ERROR [{doc_id}]: {error}")
                        failed += 1
                    else:
                        assert record is not None
                        records.append(record)
                        logger.info(
                            "[%s] 节点 %d 个，边 %d 条。",
                            doc_id, record["num_nodes"], record["edge_index"].shape[1],
                        )
                    pbar.update(1)

    summary = f"样本构建完成：成功 {len(records)}，失败 {failed}，共 {total} 个"
    tqdm.write(summary)
    logger.info(summary)
    return records


# ── 6. Dataset 类 ─────────────────────────────────────────────────────────────

class TopoDataset(Dataset):
    """
    拓扑评分数据集，每个样本对应一个事件文档图。

    __getitem__ 返回 Tensor dict：
        features   : [N, 8]  float32  节点结构特征
        labels     : [N]     float32  rank 台阶标签
        member     : [N]     float32  拆解成员 0/1
        edge_index : [2, E]  int64    有向边索引
        num_nodes  : int              节点数（DataLoader collate 用）
        doc_id     : str              文档 ID
        node_ids   : list[str]        节点 ID 列表（推理时还原用）
    """

    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        return {
            "features":   torch.from_numpy(rec["features"]),
            "labels":     torch.from_numpy(rec["labels"]),
            "member":     torch.from_numpy(rec["member"]),
            "edge_index": torch.from_numpy(rec["edge_index"]),
            "num_nodes":  rec["num_nodes"],
            "doc_id":     rec["doc_id"],
            "node_ids":   rec["node_ids"],
        }

    @classmethod
    def build(
        cls,
        feature_root: Path,
        label_root: Path,
        graph_root: Path,
        config: TopoConfig,
        logger: logging.Logger,
        *,
        manifest_path: Path | None = None,
        split: str = "labeled",
    ) -> "TopoDataset":
        """工厂方法：按 split_manifest 划分加载样本。"""
        doc_ids: list[str] | None = None
        if manifest_path is not None and manifest_path.exists():
            manifest = load_manifest(manifest_path)
            doc_ids = get_doc_ids(manifest, split=split)  # type: ignore[arg-type]
            logger.info("从 manifest 加载 split=%s，共 %d 个文档。", split, len(doc_ids))
        records = load_all_records(
            feature_root, label_root, graph_root, config, logger, doc_ids=doc_ids,
        )
        return cls(records)


# ── 日志配置 ──────────────────────────────────────────────────────────────────

def _setup_logging(log_dir: Path, log_name: str = "dataset") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{log_name}_{timestamp}.log"

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    logger.addHandler(ch)

    return logger


# ── 入口（验证用）────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg  = TopoConfig()
    base = Path(__file__).resolve().parents[1]

    feature_root = base / "output" / "features"
    label_root   = base / "output" / "labels"
    graph_root   = base / "data"   / "output" / "doc_results"
    log_dir      = base / "log"

    logger = _setup_logging(log_dir)

    dataset = TopoDataset.build(feature_root, label_root, graph_root, cfg, logger)
    print(f"\n数据集大小: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"doc_id        : {sample['doc_id']}")
        print(f"num_nodes     : {sample['num_nodes']}")
        print(f"features shape: {sample['features'].shape}")
        print(f"labels shape  : {sample['labels'].shape}")
        print(f"edge_index    : {sample['edge_index'].shape}")
        print(f"labels        : {sample['labels']}")
