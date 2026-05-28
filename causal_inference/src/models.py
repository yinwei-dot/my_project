"""Data models and minimal payload normalization for the causal pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .infra import EXPERTS, event_id_num, extract_doc_id, read_json_utf8


@dataclass(frozen=True)
class Event:
    event_id: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"event_id": self.event_id, "text": self.text}


@dataclass
class DocumentInput:
    doc_id: str
    doc_name: str
    doc_text: str
    events: list[Event]

    @property
    def event_ids(self) -> list[str]:
        return [event.event_id for event in self.events]

    @property
    def event_map(self) -> dict[str, Event]:
        return {event.event_id: event for event in self.events}

    @property
    def forward_edge_keys(self) -> set[tuple[str, str]]:
        ids = self.event_ids
        return {(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))}

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "doc_text": self.doc_text,
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(frozen=True)
class Edge:
    cause: str
    effect: str
    reason: str
    supporters: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> tuple[str, str]:
        return (self.cause, self.effect)

    def to_dict(self, *, include_supporters: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cause": self.cause,
            "effect": self.effect,
            "reason": self.reason,
        }
        if include_supporters and self.supporters:
            payload["supporters"] = list(self.supporters)
        return payload


@dataclass
class ExpertResult:
    expert: str
    edges: list[Edge]
    temporal_edges: list[Edge] = field(default_factory=list)  # round 1+ 二分类标为 temporal 的边

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "expert": self.expert,
            "causal_edges": [edge.to_dict(include_supporters=False) for edge in self.edges],
        }
        if self.temporal_edges:
            d["temporal_edges"] = [edge.to_dict(include_supporters=False) for edge in self.temporal_edges]
        return d


@dataclass
class JudgeResult:
    kept: list[Edge]
    rejected: list[Edge]
    temporal_kept: list[Edge] = field(default_factory=list)

    @property
    def final_edges(self) -> list[Edge]:
        return list(self.kept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "causal": [edge.to_dict() for edge in self.kept],
            "temporal": [edge.to_dict() for edge in self.temporal_kept],
        }


# ── 输入文档规范化 ──────────────────────────────────────────────────────────

def normalize_document_payload(payload: Any, *, source_name: str = "") -> DocumentInput:
    if not isinstance(payload, dict):
        raise ValueError("输入文件不是 JSON 对象")
    doc_name = payload.get("doc_name")
    doc_text = payload.get("doc_text")
    events = payload.get("events")
    if not isinstance(doc_name, str) or not doc_name.strip():
        raise ValueError("doc_name 缺失或非法")
    if not isinstance(doc_text, str) or not doc_text.strip():
        raise ValueError("doc_text 缺失或非法")
    if not isinstance(events, dict) or not events:
        raise ValueError("events 缺失或非法")

    normalized: list[Event] = []
    for key, text in sorted(events.items(), key=lambda item: event_id_num(str(item[0]))):
        if not isinstance(key, str) or event_id_num(key) == 10**9:
            raise ValueError(f"非法事件编号: {key}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"事件文本为空: {key}")
        normalized.append(Event(event_id=key.strip(), text=text.strip()))

    return DocumentInput(
        doc_id=extract_doc_id(doc_name or source_name),
        doc_name=doc_name.strip(),
        doc_text=doc_text.strip(),
        events=normalized,
    )


def load_document_input(path: Path) -> DocumentInput:
    return normalize_document_payload(read_json_utf8(path), source_name=path.name)


# ── 边规范化 ────────────────────────────────────────────────────────────────

def _normalize_edge(
    item: Any,
    forward_keys: set[tuple[str, str]],
    *,
    allow_supporters: bool,
) -> Edge | None:
    """单条边规范化。强制前向（cause id < effect id），跳过非法项。"""
    if not isinstance(item, dict):
        return None
    cause = str(item.get("cause", "")).strip()
    effect = str(item.get("effect", "")).strip()
    reason = str(item.get("reason", "")).strip()
    if cause == effect or not reason:
        return None
    if (cause, effect) not in forward_keys:
        return None
    supporters: tuple[str, ...] = ()
    if allow_supporters:
        raw = item.get("supporters", [])
        if isinstance(raw, list):
            supporters = tuple(s for s in raw if isinstance(s, str) and s in EXPERTS)
    return Edge(cause=cause, effect=effect, reason=reason, supporters=supporters)


def _dedupe(edges: Iterable[Edge]) -> list[Edge]:
    seen: dict[tuple[str, str], Edge] = {}
    for edge in edges:
        seen.setdefault(edge.key, edge)
    return sorted(seen.values(), key=lambda e: (event_id_num(e.cause), event_id_num(e.effect)))


def normalize_expert_payload(
    payload: Any,
    expert_name: str,
    forward_keys: set[tuple[str, str]],
) -> ExpertResult:
    edges: list[Edge] = []
    if isinstance(payload, dict):
        # LLM 原始输出用 "edges"；落盘后的 to_dict() 用 "causal_edges"——两者都要支持
        raw = payload.get("edges") or payload.get("causal_edges") or []
        if isinstance(raw, list):
            for item in raw:
                edge = _normalize_edge(item, forward_keys, allow_supporters=False)
                if edge is not None:
                    edges.append(edge)
    return ExpertResult(expert=expert_name, edges=_dedupe(edges))


_KEEP_LABELS = {"causal", "casual", "keep", "yes", "true", "因果", "保留"}


def normalize_expert_classification_payload(
    payload: Any,
    expert_name: str,
    candidate_keys: set[tuple[str, str]],
) -> ExpertResult:
    """解析 Round 1+ 的二分类输出：

    形如 ``{"edge1": ["E1", "E3", "causal", "reason"], "edge2": [..., "temporal", ...]}``。

    - label 为 causal/keep 系列 → edges；temporal → temporal_edges；其余视为剔除。
    - 仅在 candidate_keys 内的边被接受。
    """
    edges: list[Edge] = []
    temporal_edges: list[Edge] = []
    if isinstance(payload, dict):
        for value in payload.values():
            if not isinstance(value, list) or len(value) < 2:
                continue
            cause = str(value[0]).strip()
            effect = str(value[1]).strip()
            if (cause, effect) not in candidate_keys:
                continue
            label = str(value[2]).strip().lower() if len(value) >= 3 else ""
            reason = str(value[3]).strip() if len(value) >= 4 else ""
            if label in _KEEP_LABELS:
                if not reason:
                    reason = "保留为因果。"
                edges.append(Edge(cause=cause, effect=effect, reason=reason))
            elif label == "temporal":
                if not reason:
                    reason = "时序关系。"
                temporal_edges.append(Edge(cause=cause, effect=effect, reason=reason))
            # 其他标签（no/false/remove/drop）静默丢弃
    return ExpertResult(expert=expert_name, edges=_dedupe(edges), temporal_edges=_dedupe(temporal_edges))


# ── Judge 输出规范化 ────────────────────────────────────────────────────────

def build_candidate_pool(final_round_results: dict[str, ExpertResult]) -> list[dict[str, Any]]:
    """合并最后一轮所有专家边集，得到 Judge 的候选边池。"""
    pool: dict[tuple[str, str], dict[str, Any]] = {}
    for expert_name, result in final_round_results.items():
        for edge in result.edges:
            entry = pool.setdefault(
                edge.key,
                {"cause": edge.cause, "effect": edge.effect, "supporters": [], "reasons": []},
            )
            if expert_name not in entry["supporters"]:
                entry["supporters"].append(expert_name)
            entry["reasons"].append({"expert": expert_name, "reason": edge.reason})
    return sorted(pool.values(), key=lambda x: (event_id_num(x["cause"]), event_id_num(x["effect"])))


def _candidate_supporters(candidate: dict[str, Any] | None) -> tuple[str, ...]:
    if candidate is None:
        return ()
    raw = candidate.get("supporters", [])
    if not isinstance(raw, list):
        return ()
    return tuple(s for s in raw if isinstance(s, str) and s in EXPERTS)


_JUDGE_CAUSAL_LABELS = {"causal", "casual", "keep", "yes"}


def normalize_judge_payload(
    payload: Any,
    candidate_pool: list[dict[str, Any]],
    forward_keys: set[tuple[str, str]],
) -> JudgeResult:
    """解析 Judge 二分类输出：{"edgeN": [cause, effect, label, reason]}

    label "causal" → kept；"temporal" → temporal_kept；其余 / 未出现 → rejected。
    """
    lookup = {(c["cause"], c["effect"]): c for c in candidate_pool}
    causal_edges: list[Edge] = []
    temporal_edges: list[Edge] = []
    seen_keys: set[tuple[str, str]] = set()
    other_reject_reasons: dict[tuple[str, str], str] = {}

    if isinstance(payload, dict):
        for val in payload.values():
            if not isinstance(val, list) or len(val) < 2:
                continue
            cause = str(val[0]).strip()
            effect = str(val[1]).strip()
            label = str(val[2]).strip().lower() if len(val) >= 3 else ""
            if cause == effect or (cause, effect) not in forward_keys:
                continue
            seen_keys.add((cause, effect))
            candidate = lookup.get((cause, effect))
            reason = str(val[3]).strip() if len(val) >= 4 else ""
            supporters = _candidate_supporters(candidate)
            if label in _JUDGE_CAUSAL_LABELS:
                if not reason and candidate:
                    reasons = candidate.get("reasons", [])
                    if isinstance(reasons, list) and reasons:
                        reason = str(reasons[0].get("reason", "")).strip()
                reason = reason or "Judge 分类保留。"
                causal_edges.append(Edge(cause=cause, effect=effect, reason=reason, supporters=supporters))
            elif label == "temporal":
                reason = reason or "Judge 分类为时序关系。"
                temporal_edges.append(Edge(cause=cause, effect=effect, reason=reason, supporters=supporters))
            else:
                # 其他标签（no/false/remove/drop）→ 归入 rejected；顺手收集原因
                if reason:
                    other_reject_reasons[(cause, effect)] = reason

    classified_keys = {e.key for e in causal_edges} | {e.key for e in temporal_edges}

    rejected_edges: list[Edge] = [
        Edge(
            cause=c["cause"],
            effect=c["effect"],
            reason=other_reject_reasons.get((c["cause"], c["effect"]), "Judge 未保留此边。"),
            supporters=_candidate_supporters(c),
        )
        for c in candidate_pool
        if (c["cause"], c["effect"]) not in classified_keys
    ]

    return JudgeResult(
        kept=_dedupe(causal_edges),
        rejected=_dedupe(rejected_edges),
        temporal_kept=_dedupe(temporal_edges),
    )


def expert_result_from_dict(
    payload: Any,
    expert_name: str,
    forward_keys: set[tuple[str, str]],
) -> ExpertResult:
    """从已落盘的 expert_results JSON 还原 ExpertResult（用于 resume）。"""
    result = normalize_expert_payload(payload, expert_name, forward_keys)
    # 还原 temporal_edges（round 1+ 二分类时落盘的时序边）
    if isinstance(payload, dict):
        raw = payload.get("temporal_edges", [])
        if isinstance(raw, list):
            temporal = [_normalize_edge(item, forward_keys, allow_supporters=False) for item in raw]
            result.temporal_edges = [e for e in temporal if e is not None]
    return result


def edges_to_json(edges: Iterable[Edge]) -> str:
    return json.dumps([edge.to_dict() for edge in edges], ensure_ascii=False, indent=2)


__all__ = [
    "DocumentInput",
    "Edge",
    "Event",
    "ExpertResult",
    "JudgeResult",
    "build_candidate_pool",
    "edges_to_json",
    "expert_result_from_dict",
    "load_document_input",
    "normalize_document_payload",
    "normalize_expert_payload",
    "normalize_expert_classification_payload",
    "normalize_judge_payload",
]
