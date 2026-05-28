from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional

import networkx as nx
from tqdm import tqdm

from config import TopoConfig, TrainConfig
from DataSplit import (
    build_split_manifest,
    count_nodes_from_graph,
    preview_split_from_graphs,
)


# ── 日志配置 ──────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path, log_name: str = "ground_truth") -> logging.Logger:
    """
    控制台：仅输出 WARNING 及以上（tqdm 进度条不受干扰）。
    日志文件：输出 INFO 及以上（完整明细）。
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{log_name}_{timestamp}.log"

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # 文件 handler：INFO+，完整格式
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(fh)

    # 控制台 handler：WARNING+，不污染 tqdm 进度条
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    logger.addHandler(ch)

    return logger


# ── 1. 读取 graph.json ────────────────────────────────────────────────────────

def load_graph_json(graph_path: Path) -> dict:
    with open(graph_path, encoding="utf-8") as f:
        return json.load(f)


# ── 2. 构建有向图（节点用事件 ID 字符串）────────────────────────────────────────

def build_digraph(payload: dict) -> tuple[nx.DiGraph, list[str]]:
    """
    返回:
        G        : nx.DiGraph，节点名为事件 ID 字符串（如 "E1"）
        node_ids : 按事件 ID 排序的节点列表
    """
    events = payload.get("events", [])
    node_ids: list[str] = [str(e["event_id"]) for e in events if e.get("event_id")]

    G = nx.DiGraph()
    G.add_nodes_from(node_ids)

    for edge in payload.get("causal_edges", []) or []:
        src, dst = str(edge.get("cause", "")), str(edge.get("effect", ""))
        if src in G and dst in G:
            G.add_edge(src, dst)

    for edge in payload.get("temporal_edges", []) or []:
        src, dst = str(edge.get("source", "")), str(edge.get("target", ""))
        if src in G and dst in G:
            G.add_edge(src, dst)

    return G, node_ids


# ── 3. 计算可达对数 ────────────────────────────────────────────────────────────

def count_reachable_pairs(G: nx.DiGraph) -> int:
    """
    R(G) = 所有有序节点对 (u,v), u≠v，且 v 从 u 有向可达的数量。
    对每个节点做一次 BFS/DFS，时间复杂度 O(N*(N+E))。
    """
    total = 0
    for node in G.nodes():
        # nx.descendants 返回从 node 出发可到达的所有节点（不含自身）
        total += len(nx.descendants(G, node))
    return total


# ── 4. 搜索最小移除集合 ────────────────────────────────────────────────────────

def find_minimal_removal_sets(
    G: nx.DiGraph,
    config: TopoConfig,
) -> tuple[list[list[str]], int, int]:
    """
    暴力枚举最小移除集合（3–30 节点图，不设组合数上限）。

    返回:
        removal_sets   : 所有满足条件的最小移除集合（节点 ID 字符串列表）
        original_pairs : R(G) 原始可达对数
        final_pairs    : 移除后的可达对数（取所有集合中最小值）
    """
    nodes = list(G.nodes())
    n = len(nodes)
    original_pairs = count_reachable_pairs(G)

    if original_pairs == 0:
        return [[]], 0, 0

    target = math.ceil(config.theta * original_pairs)

    for k in range(1, n + 1):
        matches: list[list[str]] = []
        best_final = original_pairs
        for removal_set in combinations(nodes, k):
            G_temp = G.copy()
            G_temp.remove_nodes_from(removal_set)
            pairs = count_reachable_pairs(G_temp)
            if pairs <= target:
                matches.append(list(removal_set))
                best_final = min(best_final, pairs)

        if matches:
            return matches, original_pairs, best_final

    removal, final = _greedy(G, target)
    return [removal], original_pairs, final


def _greedy(
    G: nx.DiGraph,
    target: int,
) -> tuple[list[str], int]:
    """贪心：每次移除使可达对数下降最多的节点。"""
    temp = G.copy()
    removal: list[str] = []

    while True:
        pairs = count_reachable_pairs(temp)
        if pairs <= target or temp.number_of_nodes() == 0:
            return removal, pairs

        best_node: str | None = None
        best_pairs = pairs + 1
        # 试验移除每个节点，选使可达对数下降最多的（贪心最优子结构，O(N*(N+E)) per step）
        for node in list(temp.nodes()):
            trial = temp.copy()
            trial.remove_node(node)
            p = count_reachable_pairs(trial)
            if p < best_pairs:
                best_pairs = p
                best_node = node

        if best_node is None:
            break
        removal.append(best_node)
        temp.remove_node(best_node)

    return removal, count_reachable_pairs(temp)


# ── 5. 生成归一化标签 ─────────────────────────────────────────────────────────

def compute_labels(
    node_ids: list[str],
    removal_sets: list[list[str]],
) -> dict[str, float]:
    """
    统计每个节点在所有移除集合中的出现频次，归一化到 [0,1]。
    """
    freq: dict[str, float] = {nid: 0.0 for nid in node_ids}
    valid_sets = [s for s in removal_sets if s]
    if not valid_sets:
        return freq

    for removal_set in valid_sets:
        for nid in removal_set:
            if nid in freq:
                freq[nid] += 1.0

    max_freq = max(freq.values())
    if max_freq > 0:
        freq = {nid: v / max_freq for nid, v in freq.items()}
    return freq


# ── 6. 双阶段标签：c0 成员 + ND 传播 + tier 台阶 rank ───────────────────────

LABEL_SCHEMA_VERSION = 2
_SCORE_EPS  = 1e-9
_ND_CEILING = 0.3   # ND 层标签上界：ND 节点标签 ∈ (0, 0.3]
_D_FLOOR    = 0.7   # D  层标签下界：D  节点标签 ∈ [0.7, 1.0]


def compute_c0_scores(
    node_ids: list[str],
    removal_sets: list[list[str]],
) -> dict[str, float]:
    """c0[v] = 最优拆除集出现频次 / max_freq；非拆解节点为 0。"""
    return compute_labels(node_ids, removal_sets)


def compute_member_labels(c0: dict[str, float]) -> dict[str, float]:
    """拆解节点=1.0，非拆解=0.0（Stage1 二分类）。"""
    return {nid: (1.0 if c0[nid] > _SCORE_EPS else 0.0) for nid in c0}


def compute_s_nd_scores(
    G: nx.DiGraph,
    node_ids: list[str],
    c0: dict[str, float],
) -> dict[str, float]:
    """
    仅对非拆解节点 v∈ND：从相邻拆解节点 u∈D 接收 c0[u]/deg(u)。
    deg(u) = in+out；前驱与后继均参与（不区分边类型）。
    """
    d_set = {nid for nid in node_ids if c0.get(nid, 0.0) > _SCORE_EPS}
    s_nd: dict[str, float] = {nid: 0.0 for nid in node_ids}
    for v in node_ids:
        if v in d_set:
            continue
        score = 0.0
        for u in G.predecessors(v):
            if u in d_set:
                score += c0[u] / max(1, int(G.degree(u)))
        for w in G.successors(v):
            if w in d_set:
                score += c0[w] / max(1, int(G.degree(w)))
        s_nd[v] = score
    return s_nd


def _compute_local_reach_proxies(G: nx.DiGraph, node_ids: list[str]) -> dict[str, int]:
    """局部可达代理：pred_two_hop × succ_two_hop，用于同层同分同度节点的细分排序（替代nid字典序）。"""
    proxies: dict[str, int] = {}
    for v in node_ids:
        preds1 = set(G.predecessors(v))
        preds2: set[str] = set()
        for p in preds1:
            preds2.update(G.predecessors(p))
        preds2 -= preds1
        preds2.discard(v)

        succs1 = set(G.successors(v))
        succs2: set[str] = set()
        for s in succs1:
            succs2.update(G.successors(s))
        succs2 -= succs1
        succs2.discard(v)

        proxies[v] = len(preds2) * len(succs2)
    return proxies


def _group_score(
    nid: str,
    c0: dict[str, float],
    s_nd: dict[str, float],
    d_set: set[str],
) -> float:
    return c0[nid] if nid in d_set else s_nd[nid]


def _same_tier_bucket(
    a: str,
    b: str,
    c0: dict[str, float],
    s_nd: dict[str, float],
    degrees: dict[str, int],
    d_set: set[str],
    reach_proxies: dict[str, int],
) -> bool:
    if (a in d_set) != (b in d_set):
        return False
    sa = _group_score(a, c0, s_nd, d_set)
    sb = _group_score(b, c0, s_nd, d_set)
    if abs(sa - sb) > _SCORE_EPS:
        return False
    if degrees[a] != degrees[b]:
        return False
    # D 节点：用 reach_proxy 进一步细分，使 D 排名可学习
    # ND 节点：保留原始桶定义，(s_nd, degree) 决定桶，与标签语义一致
    if a in d_set:
        return reach_proxies[a] == reach_proxies[b]
    return True


def compute_tiered_rank_labels(
    G: nx.DiGraph,
    node_ids: list[str],
    removal_sets: list[list[str]],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """
    tier 台阶 rank_labels（含层间 gap）：
      - ND tier：标签 ∈ (0, _ND_CEILING]，按 (s_nd, degree, reach_proxy) 排序
      - D  tier：标签 ∈ [_D_FLOOR,  1.0]，按 (c0,  degree, reach_proxy) 排序
      - 层间 dead-zone：(_ND_CEILING, _D_FLOOR)，无节点落入 → 强化分层梯度信号
    """
    c0 = compute_c0_scores(node_ids, removal_sets)
    s_nd = compute_s_nd_scores(G, node_ids, c0)
    member = compute_member_labels(c0)
    d_set = {nid for nid in node_ids if c0[nid] > _SCORE_EPS}
    degrees = {nid: int(G.degree(nid)) for nid in node_ids}
    reach_proxies = _compute_local_reach_proxies(G, node_ids)

    n = len(node_ids)
    if n == 0:
        return c0, s_nd, member, {}

    order = sorted(
        node_ids,
        key=lambda nid: (
            1 if nid in d_set else 0,
            _group_score(nid, c0, s_nd, d_set),
            degrees[nid],
            reach_proxies[nid] if nid in d_set else 0,  # D: reach_proxy 细分; ND: 保留原始 nid 字典序
            nid,
        ),
    )

    n_nd = sum(1 for nid in node_ids if nid not in d_set)
    n_d  = n - n_nd

    rank_labels: dict[str, float] = {}
    i = 0
    while i < n:
        j = i + 1
        while j < n and _same_tier_bucket(
            order[i], order[j], c0, s_nd, degrees, d_set, reach_proxies
        ):
            j += 1

        if order[i] in d_set:
            # D tier：映射到 [_D_FLOOR, 1.0]
            tier_pos  = i - n_nd + 1          # 1-based within D tier
            tier_frac = tier_pos / n_d if n_d > 0 else 1.0
            label_val = _D_FLOOR + tier_frac * (1.0 - _D_FLOOR)
        else:
            # ND tier：映射到 (0, _ND_CEILING]
            tier_pos  = i + 1                  # 1-based within ND tier
            tier_frac = tier_pos / n_nd if n_nd > 0 else 1.0
            label_val = tier_frac * _ND_CEILING

        for k in range(i, j):
            rank_labels[order[k]] = label_val
        i = j
    return c0, s_nd, member, rank_labels


def compute_propagation_labels(
    G: nx.DiGraph,
    node_ids: list[str],
    removal_sets: list[list[str]],
) -> dict[str, float]:
    """兼容旧调用：返回 tier 台阶 rank_labels。"""
    _, _, _, rank_labels = compute_tiered_rank_labels(G, node_ids, removal_sets)
    return rank_labels


# ── 主流程：处理单个 graph.json ───────────────────────────────────────────────

def process_one(graph_path: Path, config: TopoConfig) -> dict:
    """
    单文档完整标签生成流程：构图 → 搜索最小移除集合 → 生成频次标签 → 传播标签。

    Returns:
        {
          "doc_id"        : str,
          "theta"         : float,            # 本次使用的阈值（来自 config.theta）
          "original_pairs": int,              # 原始可达对数 R(G)
          "final_pairs"   : int,              # 移除后的最小可达对数
          "removal_sets"  : list[list[str]],  # 所有最优最小移除集合（可能多个）
          "labels"        : dict[str, float], # 频次归一化标签（仅关键节点 > 0）
                    "rank_labels"   : dict[str, float], # tier 台阶序（D≻ND）
                    "member_labels" : dict[str, float], # 拆解=1 / 非拆解=0
                    "s_nd_scores"   : dict[str, float], # ND 传播分
                    "c0_scores"     : dict[str, float], # 频次归一化
        }
    """
    payload = load_graph_json(graph_path)
    G, node_ids = build_digraph(payload)
    removal_sets, original_pairs, final_pairs = find_minimal_removal_sets(G, config)
    c0_scores, s_nd_scores, member_labels, rank_labels = compute_tiered_rank_labels(
        G, node_ids, removal_sets,
    )

    return {
        "doc_id": payload.get("doc_id", graph_path.parent.name),
        "num_nodes": len(node_ids),
        "theta": config.theta,
        "original_pairs": original_pairs,
        "final_pairs": final_pairs,
        "removal_sets": removal_sets,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "labels": c0_scores,
        "c0_scores": c0_scores,
        "s_nd_scores": s_nd_scores,
        "member_labels": member_labels,
        "rank_labels": rank_labels,
    }


# ── 批量处理：日志与进度条格式 ─────────────────────────────────────────────────

def _print_gt_preamble(
    total: int,
    skip_lt3: int,
    skip_gt30: int,
    n_train: int,
    n_test: int,
) -> None:
    tqdm.write(
        f"找到 {total} 个 graph.json，跳过{skip_lt3}条（n<3），跳过{skip_gt30}条（n>30）"
    )
    tqdm.write(
        f"划分: train={n_train} test={n_test} inference_only={skip_gt30}"
    )


def _print_gt_summary(
    processed: int,
    skipped_resume: int,
    failed: int,
    total: int,
) -> None:
    tqdm.write(
        f"完成：新处理 {processed}，断点跳过 {skipped_resume}，"
        f"失败 {failed}，共 {total} 个"
    )


def _collect_gt_todo(
    graph_paths: list[Path],
    output_root: Path,
    config: TopoConfig,
    resume: bool,
) -> tuple[int, int, list[Path], int]:
    """返回 (skip_lt3, skip_gt30, todo, skipped_resume)。"""
    skip_lt3 = skip_gt30 = skipped_resume = 0
    todo: list[Path] = []

    for graph_path in graph_paths:
        n_nodes = count_nodes_from_graph(graph_path)
        if n_nodes < config.min_nodes:
            skip_lt3 += 1
            continue
        if n_nodes > config.labeled_max_nodes:
            skip_gt30 += 1
            continue
        out_path = output_root / graph_path.parent.name / "topo_labels.json"
        if resume and out_path.exists():
            skipped_resume += 1
        else:
            todo.append(graph_path)

    return skip_lt3, skip_gt30, todo, skipped_resume


# ── 批量处理 train_data（支持断点续跑）────────────────────────────────────────

def process_all(
    train_root: Path,
    output_root: Path,
    config: TopoConfig,
    logger: logging.Logger,
    resume: bool = True,
    *,
    manifest_path: Path | None = None,
    train_cfg: TrainConfig | None = None,
) -> None:
    """
    串行批量处理：仅对 min_nodes ≤ n ≤ labeled_max_nodes 的图生成标签。
    结束后写入 split_manifest.json。
    """
    graph_paths = sorted(train_root.rglob("graph.json"))
    total = len(graph_paths)
    logger.info("开始批量处理，共 %d 个 graph.json，train_root=%s", total, train_root)

    skip_lt3, skip_gt30, todo, skipped_resume = _collect_gt_todo(
        graph_paths, output_root, config, resume,
    )

    if train_cfg is not None:
        p_lt3, p_gt30, n_train, n_test = preview_split_from_graphs(
            train_root, config, train_cfg,
        )
        _print_gt_preamble(total, p_lt3, p_gt30, n_train, n_test)
    else:
        _print_gt_preamble(total, skip_lt3, skip_gt30, 0, 0)

    processed = 0
    failed_ids: list[str] = []
    todo_total = len(todo)

    if todo_total > 0:
        with tqdm(total=todo_total, unit="doc", dynamic_ncols=True) as pbar:
            for graph_path in todo:
                doc_id = graph_path.parent.name
                out_dir = output_root / doc_id
                out_path = out_dir / "topo_labels.json"
                pbar.set_description(doc_id)

                try:
                    result = process_one(graph_path, config)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

                    logger.info(
                        "[%s] nodes=%d pairs %d → %d, removal_sets=%d, saved → %s",
                        doc_id, result["num_nodes"],
                        result["original_pairs"],
                        result["final_pairs"],
                        len(result["removal_sets"]),
                        out_path,
                    )
                    processed += 1
                except Exception as exc:
                    logger.error("[%s] 处理失败: %s", doc_id, exc, exc_info=True)
                    tqdm.write(f"  ERROR [{doc_id}]: {exc}")
                    failed_ids.append(doc_id)

                pbar.update(1)

    _print_gt_summary(processed, skipped_resume, len(failed_ids), total)
    logger.info(
        "完成：新处理 %d，断点跳过 %d，失败 %d，共 %d 个",
        processed, skipped_resume, len(failed_ids), total,
    )

    if manifest_path is not None and train_cfg is not None:
        build_split_manifest(
            train_root, output_root, manifest_path, config, train_cfg, failed=failed_ids,
        )
        logger.info("划分清单已保存 → %s", manifest_path)


# ── 多进程任务单元（模块级函数，可被 pickle）────────────────────────────────────

def _worker_task(args: tuple) -> tuple[str, Optional[dict], Optional[str]]:
    """
    子进程执行单元。返回 (doc_id, result_or_None, error_msg_or_None)。
    不在子进程中写文件，由主进程统一写入，避免并发写冲突。
    """
    graph_path_str, config_theta, config_min_nodes, config_labeled_max = args
    graph_path = Path(graph_path_str)
    cfg = TopoConfig(
        theta=config_theta,
        min_nodes=config_min_nodes,
        labeled_max_nodes=config_labeled_max,
    )
    doc_id = graph_path.parent.name
    try:
        result = process_one(graph_path, cfg)
        return doc_id, result, None
    except Exception as exc:
        return doc_id, None, str(exc)


# ── 并行批量处理（支持断点续跑）──────────────────────────────────────────────────

def process_all_parallel(
    train_root: Path,
    output_root: Path,
    config: TopoConfig,
    logger: logging.Logger,
    resume: bool = True,
    *,
    manifest_path: Path | None = None,
    train_cfg: TrainConfig | None = None,
) -> None:
    graph_paths = sorted(train_root.rglob("graph.json"))
    total = len(graph_paths)
    logger.info("开始并行处理，共 %d 个，workers=%d，train_root=%s",
                total, config.resolved_workers(), train_root)

    skip_lt3, skip_gt30, todo, skipped_resume = _collect_gt_todo(
        graph_paths, output_root, config, resume,
    )

    if train_cfg is not None:
        p_lt3, p_gt30, n_train, n_test = preview_split_from_graphs(
            train_root, config, train_cfg,
        )
        _print_gt_preamble(total, p_lt3, p_gt30, n_train, n_test)
    else:
        _print_gt_preamble(total, skip_lt3, skip_gt30, 0, 0)

    processed = 0
    failed_ids: list[str] = []
    todo_total = len(todo)
    num_workers = min(config.resolved_workers(), todo_total) if todo_total else 1

    task_args = [
        (str(p), config.theta, config.min_nodes, config.labeled_max_nodes)
        for p in todo
    ]

    if todo_total > 0:
        with mp.Pool(processes=num_workers) as pool:
            with tqdm(total=todo_total, unit="doc", dynamic_ncols=True) as pbar:
                for doc_id, result, error in pool.imap_unordered(_worker_task, task_args):
                    pbar.set_description(doc_id)
                    if error is not None:
                        logger.error("[%s] 处理失败: %s", doc_id, error)
                        tqdm.write(f"  ERROR [{doc_id}]: {error}")
                        failed_ids.append(doc_id)
                    else:
                        assert result is not None
                        out_dir = output_root / doc_id
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / "topo_labels.json"
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        logger.info(
                            "[%s] nodes=%d pairs %d → %d, removal_sets=%d, saved → %s",
                            doc_id, result["num_nodes"],
                            result["original_pairs"], result["final_pairs"],
                            len(result["removal_sets"]), out_path,
                        )
                        processed += 1
                    pbar.update(1)

    _print_gt_summary(processed, skipped_resume, len(failed_ids), total)
    logger.info(
        "完成：新处理 %d，断点跳过 %d，失败 %d，共 %d 个",
        processed, skipped_resume, len(failed_ids), total,
    )

    if manifest_path is not None and train_cfg is not None:
        build_split_manifest(
            train_root, output_root, manifest_path, config, train_cfg, failed=failed_ids,
        )
        logger.info("划分清单已保存 → %s", manifest_path)


if __name__ == "__main__":
    cfg = TopoConfig()
    train_cfg = TrainConfig()
    base = Path(__file__).resolve().parents[1]
    train_root = base / "data" / "output" / "doc_results"
    output_root = base / "output" / "labels"
    manifest_path = base / "output" / "split_manifest.json"
    log_dir = base / "log"

    logger = setup_logging(log_dir)

    if cfg.resolved_workers() == 1:
        process_all(
            train_root, output_root, cfg, logger, resume=True,
            manifest_path=manifest_path, train_cfg=train_cfg,
        )
    else:
        process_all_parallel(
            train_root, output_root, cfg, logger, resume=True,
            manifest_path=manifest_path, train_cfg=train_cfg,
        )
