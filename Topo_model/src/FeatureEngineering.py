from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from tqdm import tqdm

from config import TopoConfig


# ── 特征名称（顺序即矩阵列顺序）────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # ── 纯拓扑特征，有向，无删点操作 ─────────────────────────────────────────
    "in_degree",            # 入度：被多少节点直接指向
    "out_degree",           # 出度：直接指向多少节点
    "pred_two_hop",         # 二跳上游邻居数：受多少间接上游影响
    "succ_two_hop",         # 二跳下游邻居数：间接影响多少下游
    "succ_sole_count",      # 入度=1 的后继数量：只能经过 v 到达的后继 → 瓶颈信号
    "pred_sole_count",      # 出度=1 的前驱数量：只能经过 v 继续传播的前驱 → 漏斗信号
    "pred_three_hop",       # 三跳上游邻居数（仅第三跳新增节点）
    "succ_three_hop",       # 三跳下游邻居数（仅第三跳新增节点）
    "reach_local_proxy",    # 局部可达代理：pred_two_hop × succ_two_hop（ΔR 近似，局部 O(deg²)）
    "pred_inv_degree_sum",  # 上游传播信号：Σ 1/deg(u) for u∈preds(v)（s_nd 的结构近似，无需标签）
    "succ_inv_degree_sum",  # 下游传播信号：Σ 1/deg(u) for u∈succs(v)（方向性传播出口代理）
]


# ── 1. 读取 graph.json，构建有向图 ─────────────────────────────────────────────

def load_graph(graph_path: Path) -> tuple[nx.DiGraph, list[str]]:
    """
    从 graph.json 构建有向图，节点名为事件 ID 字符串（如 "E1"）。
    因果边和时序边合并为同一张图，不区分类型。

    Returns:
        G        : nx.DiGraph
        node_ids : 按事件 ID 顺序排列的节点列表
    """
    with open(graph_path, encoding="utf-8") as f:
        payload = json.load(f)

    events = payload.get("events", [])
    node_ids: list[str] = [str(e["event_id"]) for e in events if e.get("event_id")]

    G = nx.DiGraph()
    G.add_nodes_from(node_ids)

    # 因果边
    for edge in payload.get("causal_edges", []) or []:
        src, dst = str(edge.get("cause", "")), str(edge.get("effect", ""))
        if src in G and dst in G:
            G.add_edge(src, dst)

    # 时序边
    for edge in payload.get("temporal_edges", []) or []:
        src, dst = str(edge.get("source", "")), str(edge.get("target", ""))
        if src in G and dst in G:
            G.add_edge(src, dst)

    return G, node_ids


# ── 2. 计算 10 个结构特征 ──────────────────────────────────────────────────────

def compute_features(G: nx.DiGraph, node_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    计算 11 个 DAG 结构特征，结果按列做 min-max 归一化。

    保留：in_degree, out_degree, pred_two_hop, succ_two_hop, succ_sole_count, pred_sole_count
    删除：topo_depth, topo_height（全局 DP，不可扩展）；ancestor_count, descendant_count（全局 BFS O(N(N+E))）
    新增：pred_three_hop, succ_three_hop（局部 O(deg³)）；reach_local_proxy = pred_two_hop × succ_two_hop（ΔR 近似）
          pred_inv_degree_sum, succ_inv_degree_sum（传播方向性特征，s_nd 结构近似）

    所有特征均为局部/半局部拓扑 O(deg²~deg³)，不含全局算法，不含删点操作。

    Returns:
        matrix       : [N×11] float32，已归一化
        feature_names: 列名列表（与 FEATURE_NAMES 相同）
    """
    n = len(node_ids)
    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), list(FEATURE_NAMES)

    # ── 度特征 ──
    in_deg  = np.array([G.in_degree(v)  for v in node_ids], dtype=np.float32)
    out_deg = np.array([G.out_degree(v) for v in node_ids], dtype=np.float32)

    # ── 二跳特征 ──
    def two_hop(v: str, first_fn, second_fn) -> int:
        """沿 first_fn 方向的二跳邻居数（排除一跳和自身）。"""
        first = set(first_fn(v))
        second: set[str] = set()
        for nb in first:
            second.update(second_fn(nb))
        second.difference_update(first)
        second.discard(v)
        return len(second)

    pred_two_hop = np.array([two_hop(v, G.predecessors, G.predecessors) for v in node_ids], dtype=np.float32)
    succ_two_hop = np.array([two_hop(v, G.successors,   G.successors)   for v in node_ids], dtype=np.float32)

    # ── 三跳特征（局部 O(deg³)）──
    def three_hop(v: str, direction_fn) -> int:
        """三跳方向邻居数（仅统计第三跳新增节点，排除前两跳集合和自身）。"""
        first = set(direction_fn(v))
        second: set[str] = set()
        for nb in first:
            second.update(direction_fn(nb))
        second.difference_update(first)
        second.discard(v)
        third: set[str] = set()
        for nb in second:
            third.update(direction_fn(nb))
        third.difference_update(second)
        third.difference_update(first)
        third.discard(v)
        return len(third)

    pred_three_hop = np.array([three_hop(v, G.predecessors) for v in node_ids], dtype=np.float32)
    succ_three_hop = np.array([three_hop(v, G.successors)   for v in node_ids], dtype=np.float32)

    # ── 局部可达代理（ΔR 近似）──
    # reach_local_proxy ≈ pred_two_hop × succ_two_hop（raw 乘积，归一化前）
    # 近似捕捉删除 v 后可达对减少量，比全局 ancestor_count×descendant_count 快 O(N) 倍
    reach_local_proxy = pred_two_hop * succ_two_hop  # elementwise, raw counts

    # ── 新增：succ_sole_count / pred_sole_count（瓶颈/漏斗 信号）──
    # succ_sole_count[v]：入度=1 的后继数量
    #   → 这些后继只能经过 v 到达，v 是它们的唯一通道 → 实质瓶颈
    # pred_sole_count[v]：出度=1 的前驱数量
    #   → 这些前驱只能经过 v 继续传播 → 上游被迫漏斗至 v
    succ_sole_count = np.array(
        [float(sum(1 for s in G.successors(v)   if G.in_degree(s)  == 1)) for v in node_ids],
        dtype=np.float32,
    )
    pred_sole_count = np.array(
        [float(sum(1 for p in G.predecessors(v) if G.out_degree(p) == 1)) for v in node_ids],
        dtype=np.float32,
    )

    # ── 方向性传播特征（s_nd 结构近似，无需标签）──
    # pred_inv_degree_sum[v] = Σ 1/max(1, deg(u)) for u∈preds(v)
    #   → 上游节点中度越小者贡献越大，近似衡量 v 接收到的传播权重
    #   → 结构上等同于 s_nd 公式（去掉 D 标签判断），令模型可近似学习 ND 层内部排序
    # succ_inv_degree_sum[v] = Σ 1/max(1, deg(u)) for u∈succs(v)
    #   → 下游节点度倒数之和，衡量 v 向下游分发影响的集中度
    pred_inv_degree_sum = np.array(
        [sum(1.0 / max(1, G.degree(u)) for u in G.predecessors(v)) for v in node_ids],
        dtype=np.float32,
    )
    succ_inv_degree_sum = np.array(
        [sum(1.0 / max(1, G.degree(u)) for u in G.successors(v)) for v in node_ids],
        dtype=np.float32,
    )

    # ── 拼接为矩阵 ──
    raw = np.column_stack([
        in_deg, out_deg,
        pred_two_hop, succ_two_hop,
        succ_sole_count, pred_sole_count,
        pred_three_hop, succ_three_hop,
        reach_local_proxy,
        pred_inv_degree_sum, succ_inv_degree_sum,
    ])  # [N×11]

    # ── 按列 min-max 归一化 ──
    matrix = _normalize_columns(raw)
    return matrix, list(FEATURE_NAMES)


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """
    对矩阵每列做 min-max 归一化到 [0,1]。
    列全相同（max==min）则置 0，NaN/Inf 先替换为 0。
    """
    out = matrix.astype(np.float32).copy()
    for col in range(out.shape[1]):
        col_vals = out[:, col]
        col_vals[~np.isfinite(col_vals)] = 0.0
        mn, mx = float(col_vals.min()), float(col_vals.max())
        if math.isclose(mx, mn):
            out[:, col] = 0.0
        else:
            out[:, col] = (col_vals - mn) / (mx - mn)
    return out


# ── 3. 单文档流程 ──────────────────────────────────────────────────────────────

def process_one(graph_path: Path, output_dir: Path) -> dict:
    """
    单文档：load → compute_features → 保存 features.npy + meta.json。

    Returns:
        {"doc_id": ..., "num_nodes": ..., "num_features": ...}
    """
    G, node_ids = load_graph(graph_path)
    matrix, feature_names = compute_features(G, node_ids)

    doc_id = graph_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存特征矩阵
    np.save(output_dir / "features.npy", matrix)

    # 保存 meta：记录节点顺序和特征列顺序，训练时用于对齐标签
    meta = {
        "doc_id": doc_id,
        "node_ids": node_ids,
        "feature_names": feature_names,
        "num_nodes": len(node_ids),
        "num_features": len(feature_names),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"doc_id": doc_id, "num_nodes": len(node_ids), "num_features": len(feature_names)}


# ── 4. 多进程任务单元（模块级，可被 pickle）────────────────────────────────────

def _worker_task(args: tuple) -> tuple[str, Optional[dict], Optional[str]]:
    """子进程执行单元，返回 (doc_id, result_or_None, error_or_None)。"""
    graph_path_str, output_dir_str = args
    graph_path = Path(graph_path_str)
    output_dir = Path(output_dir_str)
    doc_id = graph_path.parent.name
    try:
        result = process_one(graph_path, output_dir)
        return doc_id, result, None
    except Exception as exc:
        return doc_id, None, str(exc)


# ── 5. 串行批量处理 ────────────────────────────────────────────────────────────

def process_all(
    train_root: Path,
    output_root: Path,
    config: TopoConfig,
    logger: logging.Logger,
    resume: bool = True,
    doc_ids: set[str] | None = None,
) -> None:
    """串行批量处理，num_workers=1 时使用。"""
    graph_paths = sorted(train_root.rglob("graph.json"))
    if doc_ids is not None:
        graph_paths = [p for p in graph_paths if p.parent.name in doc_ids]
    total = len(graph_paths)
    tqdm.write(f"找到 {total} 个 graph.json")
    logger.info("开始串行特征工程，共 %d 个，train_root=%s", total, train_root)

    skipped = processed = failed = 0

    with tqdm(total=total, unit="doc", dynamic_ncols=True) as pbar:
        for graph_path in graph_paths:
            doc_id = graph_path.parent.name
            out_dir = output_root / doc_id
            pbar.set_description(doc_id)

            # 断点续跑：features.npy 和 meta.json 均存在则跳过
            if resume and (out_dir / "features.npy").exists() and (out_dir / "meta.json").exists():
                logger.info("[%s] 已存在，跳过。", doc_id)
                skipped += 1
                pbar.update(1)
                continue

            try:
                result = process_one(graph_path, out_dir)
                logger.info(
                    "[%s] nodes=%d, features=%d, saved → %s",
                    doc_id, result["num_nodes"], result["num_features"], out_dir,
                )
                processed += 1
            except Exception as exc:
                logger.error("[%s] 处理失败: %s", doc_id, exc, exc_info=True)
                tqdm.write(f"  ERROR [{doc_id}]: {exc}")
                failed += 1

            pbar.update(1)

    summary = f"完成：处理 {processed}，跳过 {skipped}，失败 {failed}，共 {total} 个"
    tqdm.write(summary)
    logger.info(summary)


# ── 6. 并行批量处理 ────────────────────────────────────────────────────────────

def process_all_parallel(
    train_root: Path,
    output_root: Path,
    config: TopoConfig,
    logger: logging.Logger,
    resume: bool = True,
    doc_ids: set[str] | None = None,
) -> None:
    """多进程并行批量处理，num_workers>1 时使用。"""
    graph_paths = sorted(train_root.rglob("graph.json"))
    if doc_ids is not None:
        graph_paths = [p for p in graph_paths if p.parent.name in doc_ids]
    total = len(graph_paths)
    tqdm.write(f"找到 {total} 个 graph.json")
    logger.info("开始并行特征工程，共 %d 个，workers=%d，train_root=%s",
                total, config.resolved_workers(), train_root)

    # 断点续跑：过滤已完成的文档
    todo: list[Path] = []
    skipped = 0
    for p in graph_paths:
        out_dir = output_root / p.parent.name
        if resume and (out_dir / "features.npy").exists() and (out_dir / "meta.json").exists():
            logger.info("[%s] 已存在，跳过。", p.parent.name)
            skipped += 1
        else:
            todo.append(p)

    if skipped:
        tqdm.write(f"断点续跑：跳过 {skipped} 个，待处理 {len(todo)} 个")

    processed = failed = 0
    num_workers = min(config.resolved_workers(), len(todo)) if todo else 1

    task_args = [
        (str(p), str(output_root / p.parent.name))
        for p in todo
    ]

    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(todo), unit="doc", dynamic_ncols=True) as pbar:
            for doc_id, result, error in pool.imap_unordered(_worker_task, task_args):
                pbar.set_description(doc_id)
                if error is not None:
                    logger.error("[%s] 处理失败: %s", doc_id, error)
                    tqdm.write(f"  ERROR [{doc_id}]: {error}")
                    failed += 1
                else:
                    assert result is not None  # _worker_task 保证：error=None 时 result 非 None
                    logger.info(
                        "[%s] nodes=%d, features=%d, saved → %s",
                        doc_id, result["num_nodes"], result["num_features"],
                        output_root / doc_id,
                    )
                    processed += 1
                pbar.update(1)

    summary = f"完成：处理 {processed}，跳过 {skipped}，失败 {failed}，共 {total} 个"
    tqdm.write(summary)
    logger.info(summary)


# ── 入口 ───────────────────────────────────────────────────────────────────────

def _setup_logging(log_dir: Path, log_name: str = "feature_engineering") -> logging.Logger:
    """控制台仅输出 WARNING+，日志文件记录 INFO+。"""
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


if __name__ == "__main__":
    cfg = TopoConfig()
    base = Path(__file__).resolve().parents[1]
    train_root  = base / "data" / "output" / "doc_results"
    output_root = base / "output" / "features"
    log_dir     = base / "log"

    logger = _setup_logging(log_dir)

    if cfg.resolved_workers() == 1:
        process_all(train_root, output_root, cfg, logger, resume=True)
    else:
        process_all_parallel(train_root, output_root, cfg, logger, resume=True)
