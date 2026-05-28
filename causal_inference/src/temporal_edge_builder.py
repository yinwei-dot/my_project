"""时序边构建：因果边是特殊的时序边。

流程：
1. 阶段0：预计算所有事件对的嵌入相似度矩阵。
2. 阶段1：LLM 时序边（Judge > discourse > commonsense），按相似度降序加入，跳过已有路径的对。
3. 阶段2：嵌入相似度 ≥ 阈值的候选对，按相似度降序加入，跳过已有路径。
4. 传递规约：移除可由其他边组合覆盖的冗余长边。
5. 阶段3：孤立点消除——degree=0 的节点连接前一个编号节点（首节点连后一个）。
6. 阶段4：子图桥接——各连通分量按最小事件编号排序，依次连接相邻分量的最小编号节点。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any

from .infra import event_id_num

if TYPE_CHECKING:
    from .models import DocumentInput, Edge


TEMPORAL_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
EMBEDDING_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "embeddings"
TEMPORAL_EMBEDDING_MIN_SIM = 0.70

_EMBEDDING_MODEL_LOCK = threading.Lock()


@dataclass(frozen=True)
class TemporalEdge:
    source: str
    target: str
    reason: str
    score: float = 0.0
    span: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "reason": self.reason,
            "score": round(self.score, 4),
            "span": self.span,
        }


@lru_cache(maxsize=1)
def _get_embedding_model() -> Any:
    with _EMBEDDING_MODEL_LOCK:
        EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(EMBEDDING_CACHE_DIR))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(EMBEDDING_CACHE_DIR))
        # 若本地已缓存模型权重，强制离线模式，避免每次都发网络请求
        _model_cached = any(EMBEDDING_CACHE_DIR.rglob("pytorch_model.bin")) or any(
            EMBEDDING_CACHE_DIR.rglob("model.safetensors")
        )
        if _model_cached:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import logging as _logging
        # 压制 sentence_transformers / transformers 的 LOAD REPORT 和 tqdm 进度条
        _st_logger = _logging.getLogger("sentence_transformers")
        _tr_logger = _logging.getLogger("transformers")
        _prev_st = _st_logger.level
        _prev_tr = _tr_logger.level
        _st_logger.setLevel(_logging.ERROR)
        _tr_logger.setLevel(_logging.ERROR)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            from sentence_transformers import SentenceTransformer
            import transformers as _transformers
            _prev_verbosity = _transformers.logging.get_verbosity()
            _transformers.logging.set_verbosity_error()
            model = SentenceTransformer(TEMPORAL_EMBEDDING_MODEL, cache_folder=str(EMBEDDING_CACHE_DIR))
            _transformers.logging.set_verbosity(_prev_verbosity)
        finally:
            _st_logger.setLevel(_prev_st)
            _tr_logger.setLevel(_prev_tr)
        return model


# ── Union-Find（弱连通判断）──────────────────────────────────────────────────

class _UnionFind:
    """路径压缩 + 按秩合并的 Union-Find，用于弱连通判断。"""

    def __init__(self, nodes: list[str]) -> None:
        self._parent: dict[str, str] = {n: n for n in nodes}
        self._rank: dict[str, int] = {n: 0 for n in nodes}

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # 路径减半
            x = self._parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def all_connected(self) -> bool:
        if not self._parent:
            return True
        root = self.find(next(iter(self._parent)))
        return all(self.find(n) == root for n in self._parent)


def _has_path(adj: dict[str, set[str]], source: str, target: str) -> bool:
    if source == target:
        return True
    seen = {source}
    stack = [source]
    while stack:
        node = stack.pop()
        for nxt in adj.get(node, set()):
            if nxt == target:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _orient(left: str, right: str) -> tuple[str, str]:
    """按事件编号取向：小编号 → 大编号。"""
    if event_id_num(left) <= event_id_num(right):
        return left, right
    return right, left


def _semantic_reason(source: str, target: str, similarity: float) -> str:
    return (
        f"语义嵌入相似度 {similarity:.3f}，按事件编号 {source} 在 {target} 之前，"
        "补一条仅表达时序的边。"
    )


def _sequential_reason(source: str, target: str) -> str:
    return f"孤立点消除：{source} 与前一事件 {target} 构成时序关系，仅表达时序，不表达因果。"


def _bridge_reason(source: str, target: str) -> str:
    return f"子图桥接：{source} → {target} 为跨连通分量最小跨度对，仅表达时序，不表达因果。"


class TemporalEdgeBuilder:
    def __init__(self) -> None:
        pass

    # ── 主入口 ──────────────────────────────────────────────────────────────

    def build(
        self,
        doc: "DocumentInput",
        causal_edges: list["Edge"],
        llm_temporal_edges: list[TemporalEdge] | None = None,
        llm_temporal_priorities: list[int] | None = None,
    ) -> list[TemporalEdge]:
        """构造时序边。

        Args:
            doc: 文档（含事件列表和文本）。
            causal_edges: 已确定的因果边，时序边不重复这些。
            llm_temporal_edges: LLM 分类的时序边，优先采纳。
            llm_temporal_priorities: 与 llm_temporal_edges 平行，优先级数字（越小越高），
                用于排序：同一优先级内按相似度降序，不同优先级按优先级升序。
                若为 None，则退化为全局按相似度降序。
        """
        event_ids = [event.event_id for event in doc.events]
        if not event_ids:
            return []

        causal_pair_keys = {frozenset((e.cause, e.effect)) for e in causal_edges}

        # 有向邻接表（含因果边），用于路径冗余检测
        adj = self._initial_adjacency(event_ids, causal_edges)

        # 无向 Union-Find，用于弱连通判断
        uf = _UnionFind(event_ids)
        for edge in causal_edges:
            if edge.cause in uf._parent and edge.effect in uf._parent:
                uf.union(edge.cause, edge.effect)

        result: list[TemporalEdge] = []
        added_keys: set[frozenset] = set()  # 已加入的时序边（frozenset 无向键）

        # ── 阶段 0：预计算嵌入相似度矩阵 ───────────────────────────────────────
        sim_lookup = self._compute_sim_lookup(doc)

        # ── 阶段1：LLM 时序边（Judge > discourse > commonsense 优先级，同优先级内按相似度降序）───
        scored_llm = []
        for i, te in enumerate(llm_temporal_edges or []):
            source, target = _orient(te.source, te.target)
            sim = sim_lookup.get((source, target), 0.0)
            pri = llm_temporal_priorities[i] if llm_temporal_priorities else 0
            scored_llm.append((pri, sim, source, target, te.reason))
        scored_llm.sort(key=lambda x: (x[0], -x[1]))  # 优先级升序，同优先级相似度降序

        for pri, sim, source, target, reason in scored_llm:
            key = frozenset((source, target))
            if key in causal_pair_keys or key in added_keys:
                continue
            if _has_path(adj, source, target) or _has_path(adj, target, source):
                continue
            result.append(TemporalEdge(
                source=source,
                target=target,
                reason=reason,
                score=sim,
                span=abs(event_id_num(source) - event_id_num(target)),
            ))
            added_keys.add(key)
            adj.setdefault(source, set()).add(target)
            uf.union(source, target)

        # ── 阶段 2：嵌入相似度候选（≥ 阈值，降序）──────────────────────────
        for similarity, left, right in self._embedding_candidates_from_lookup(sim_lookup, event_ids):
            source, target = _orient(left, right)
            key = frozenset((source, target))
            if key in causal_pair_keys or key in added_keys:
                continue
            if _has_path(adj, source, target) or _has_path(adj, target, source):
                continue
            result.append(TemporalEdge(
                source=source,
                target=target,
                reason=_semantic_reason(source, target, similarity),
                score=similarity,
                span=abs(event_id_num(source) - event_id_num(target)),
            ))
            added_keys.add(key)
            adj.setdefault(source, set()).add(target)
            uf.union(source, target)

        # 传递规约——移除可由其他边组合覆盖的冗余长边
        result = self._transitive_reduce(result, causal_edges, event_ids)

        # 重建邻接表与 Union-Find（基于规约后的结果）
        adj = self._initial_adjacency(event_ids, causal_edges)
        uf = _UnionFind(event_ids)
        for edge in causal_edges:
            if edge.cause in uf._parent and edge.effect in uf._parent:
                uf.union(edge.cause, edge.effect)
        added_keys = set()
        for te in result:
            adj.setdefault(te.source, set()).add(te.target)
            uf.union(te.source, te.target)
            added_keys.add(frozenset((te.source, te.target)))

        if uf.all_connected():
            return result

        # ── 阶段 3：孤立点消除（degree=0 → 连前一个编号，E1 → 连 E2）────────
        degree: dict[str, int] = {eid: 0 for eid in event_ids}
        for e in causal_edges:
            if e.cause in degree:
                degree[e.cause] += 1
            if e.effect in degree:
                degree[e.effect] += 1
        for te in result:
            degree[te.source] += 1
            degree[te.target] += 1

        for i, eid in enumerate(event_ids):
            if degree[eid] > 0:
                continue
            neighbor = event_ids[i - 1] if i > 0 else (event_ids[1] if len(event_ids) > 1 else None)
            if neighbor is None:
                continue
            source, target = _orient(eid, neighbor)
            key = frozenset((source, target))
            if key not in added_keys and not _has_path(adj, source, target) and not _has_path(adj, target, source):
                te = TemporalEdge(
                    source=source,
                    target=target,
                    reason=_sequential_reason(eid, neighbor),
                    score=0.0,
                    span=abs(event_id_num(source) - event_id_num(target)),
                )
                result.append(te)
                added_keys.add(key)
                adj.setdefault(source, set()).add(target)
                uf.union(source, target)
                degree[source] += 1
                degree[target] += 1

        if uf.all_connected():
            return result

        # ── 阶段 4：子图桥接（各分量按最小编号排序，依次连接相邻分量首事件）──
        while not uf.all_connected():
            # 找出所有连通分量并按最小事件编号排序
            comp_map: dict[str, list[str]] = {}
            for eid in event_ids:
                root = uf.find(eid)
                comp_map.setdefault(root, []).append(eid)
            sorted_comps = sorted(
                comp_map.values(),
                key=lambda comp: min(event_id_num(e) for e in comp),
            )
            if len(sorted_comps) < 2:
                break
            # 连接第一分量的最小节点 → 第二分量的最小节点
            comp0_min = min(sorted_comps[0], key=event_id_num)
            comp1_min = min(sorted_comps[1], key=event_id_num)
            source, target = _orient(comp0_min, comp1_min)
            result.append(TemporalEdge(
                source=source,
                target=target,
                reason=_bridge_reason(source, target),
                score=0.0,
                span=abs(event_id_num(source) - event_id_num(target)),
            ))
            adj.setdefault(source, set()).add(target)
            uf.union(comp0_min, comp1_min)

        return result

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _transitive_reduce(
        self,
        temporal_edges: list[TemporalEdge],
        causal_edges: list["Edge"],
        event_ids: list[str],
    ) -> list[TemporalEdge]:
        """移除时序边中可经由其他边（时序或因果）到达的传递冗余边。

        按 span 降序检查：长跨度边最可能是传递冗余；先尝试移除它，
        若移除后仍有路径则确认冗余；否则恢复并保留。
        """
        # 构建完整有向邻接表（因果边 + 时序边）
        adj: dict[str, set[str]] = {eid: set() for eid in event_ids}
        for e in causal_edges:
            adj.setdefault(e.cause, set()).add(e.effect)
        for te in temporal_edges:
            adj.setdefault(te.source, set()).add(te.target)

        removed_indices: set[int] = set()
        # span 降序：长边先判断，避免短边被误判为冗余
        indexed = list(enumerate(sorted(temporal_edges, key=lambda x: -x.span)))
        for orig_idx, te in indexed:
            adj[te.source].discard(te.target)
            if _has_path(adj, te.source, te.target):
                removed_indices.add(orig_idx)  # 有替代路径 → 冗余，不恢复
            else:
                adj[te.source].add(te.target)  # 无替代路径 → 保留

        sorted_edges = [te for _, te in indexed]
        return [te for i, te in enumerate(sorted_edges) if i not in removed_indices]

    def _compute_sim_lookup(self, doc: "DocumentInput") -> dict[tuple[str, str], float]:
        """计算所有前向事件对（按编号升序）的嵌入相似度，返回 (source, target) → sim 字典。"""
        event_ids = [event.event_id for event in doc.events]
        if len(event_ids) < 2:
            return {}
        model = _get_embedding_model()
        texts = [event.text for event in doc.events]
        embeddings = model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False,
        )
        from sentence_transformers import util
        sim_matrix = util.cos_sim(embeddings, embeddings).cpu().tolist()
        lookup: dict[tuple[str, str], float] = {}
        for i, left in enumerate(event_ids):
            for j in range(i + 1, len(event_ids)):
                source, target = _orient(left, event_ids[j])
                lookup[(source, target)] = float(sim_matrix[i][j])
        return lookup

    def _embedding_candidates_from_lookup(
        self,
        sim_lookup: dict[tuple[str, str], float],
        event_ids: list[str],
    ) -> list[tuple[float, str, str]]:
        """从相似度字典中筛选 ≥ 阈值的对，按相似度降序返回。"""
        pairs = [
            (sim, src, tgt)
            for (src, tgt), sim in sim_lookup.items()
            if sim >= TEMPORAL_EMBEDDING_MIN_SIM
        ]
        pairs.sort(key=lambda x: (-x[0], abs(event_id_num(x[1]) - event_id_num(x[2]))))
        return pairs

    def _initial_adjacency(
        self,
        event_ids: list[str],
        causal_edges: list["Edge"],
    ) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {eid: set() for eid in event_ids}
        for edge in causal_edges:
            adj.setdefault(edge.cause, set()).add(edge.effect)
            adj.setdefault(edge.effect, set())
        return adj


__all__ = ["TemporalEdge", "TemporalEdgeBuilder"]
