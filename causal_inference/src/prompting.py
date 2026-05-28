"""拼提示词：system = 角色 + 全局约束；user = 动态模板。

设计原则：
- 专家 prompt 不出现 Judge 的输出示例；Judge prompt 不出现专家示例。
- 第 1 轮不拼"你上一轮" / "其他专家意见"等空段落。
- Judge prompt 传全文、事件列表与专家边集，要求基于原文做强过滤。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .infra import event_id_num, read_text_utf8, render_prompt_template
from .models import DocumentInput, ExpertResult, build_candidate_pool


@lru_cache(maxsize=64)
def _read_md(path_str: str) -> str:
    return read_text_utf8(Path(path_str)).strip()


def _section(*parts: str) -> str:
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def _events_json(doc: DocumentInput) -> str:
    return json.dumps([event.to_dict() for event in doc.events], ensure_ascii=False, indent=2)


def _candidate_pool_block(allowed_keys: set[tuple[str, str]]) -> str:
    """生成候选边集约束提示段：列出本轮允许判断的边，限制 LLM 不得新增候选集外的边。"""
    if not allowed_keys:
        return ""
    candidates = sorted(allowed_keys, key=lambda k: (event_id_num(k[0]), event_id_num(k[1])))
    compact = {f"edge{i + 1}": [c, e] for i, (c, e) in enumerate(candidates)}
    lines = [
        "## 候选边集约束（本轮仅可在此范围内判断）",
        "",
        "本轮你只能保留或删除以下边，**不得提议此列表之外的边**。",
        "",
        "```json",
        json.dumps(compact, ensure_ascii=False),
        "```",
    ]
    return "\n".join(lines)


def _candidate_list_for_classify(
    allowed_keys: set[tuple[str, str]],
    event_map: dict[str, Any] | None = None,
) -> str:
    """二分类复审专用：把候选边以 edgeN 顺序列出，每条边内联事件描述，便于 LLM 逐条分类。"""
    if not allowed_keys:
        return "（候选池为空）"
    candidates = sorted(allowed_keys, key=lambda k: (event_id_num(k[0]), event_id_num(k[1])))
    payload: dict[str, Any] = {}
    for i, (cause, effect) in enumerate(candidates):
        if event_map is not None:
            cause_text = event_map.get(cause, cause)
            effect_text = event_map.get(effect, effect)
            payload[f"edge{i + 1}"] = {cause: cause_text, effect: effect_text}
        else:
            payload[f"edge{i + 1}"] = [cause, effect]
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def _result_json(result: ExpertResult | None, fallback_expert: str) -> str:
    payload = result.to_dict() if result is not None else {"expert": fallback_expert, "causal_edges": []}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _peers_json(peers: dict[str, ExpertResult]) -> str:
    return json.dumps(
        {name: result.to_dict() for name, result in peers.items()},
        ensure_ascii=False,
        indent=2,
    )


def _judge_candidates_json(peers: dict[str, ExpertResult]) -> str:
    """把所有专家提案边合并为匿名候选列表，隐藏专家来源，避免 Judge 被专家身份影响。
    格式：{"edge1": ["E1", "E2"], "edge2": ["E3", "E5"], ...}
    """
    seen: dict[tuple[str, str], str] = {}
    for result in peers.values():
        for edge in result.edges:
            key = (edge.cause, edge.effect)
            if key not in seen:
                seen[key] = edge.reason or ""
    output = {
        f"edge{i + 1}": [cause, effect]
        for i, (cause, effect) in enumerate(seen.keys())
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


_ROUND_UPDATE_INSTRUCTION = (
    "## 本轮任务\n\n"
    "参考其他专家的意见，从你的视角决定每条边是否应当保留。"
    "输出你本轮认为应当保留的完整边集；不再支持的边直接删除即可。"
)


def _history_block(
    round_index: int,
    expert_name: str,
    previous_self: ExpertResult | None,
    previous_peers: dict[str, ExpertResult],
) -> str:
    if round_index == 0:
        return ""

    # previous_self 为 None 表示本专家上一轮未参与（换届专家），不展示空占位
    self_block = (
        "## 你上一轮的边集（含原因）\n\n" + _result_json(previous_self, expert_name)
        if previous_self is not None
        else ""
    )
    return _section(
        _ROUND_UPDATE_INSTRUCTION,
        self_block,
        "## 其他专家上一轮的边集（含原因）\n\n" + _peers_json(previous_peers),
    )


def build_expert_prompts(
    prompt_dir: Path,
    doc: DocumentInput,
    expert_name: str,
    round_index: int,
    previous_self: ExpertResult | None,
    previous_peers: dict[str, ExpertResult],
    *,
    allowed_keys: set[tuple[str, str]] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    role_path = prompt_dir / "experts" / f"{expert_name}.md"
    schema_path = prompt_dir / "shared" / "output_schema.md"
    propose_template_path = prompt_dir / "templates" / "expert_user.md"
    refine_template_path = prompt_dir / "templates" / "expert_user_refine.md"

    is_refine = round_index >= 1 and allowed_keys is not None

    # 第 1 轮使用通用 schema；第 2+ 轮改为二分类，避免通用 schema 中
    # "edges 数组" 的格式约束与 edgeN 字典格式冲突。
    if is_refine:
        system_prompt = _read_md(str(role_path))
    else:
        system_prompt = _section(_read_md(str(role_path)), _read_md(str(schema_path)))

    history = _history_block(round_index, expert_name, previous_self, previous_peers)

    if is_refine:
        # event_map: event_id -> text，用于候选边内联描述
        event_map = {e.event_id: e.text for e in doc.events}
        candidate_list = _candidate_list_for_classify(allowed_keys or set(), event_map)
        user_prompt = render_prompt_template(
            _read_md(str(refine_template_path)),
            {
                "EXPERT_NAME": expert_name,
                "ROUND_INDEX": str(round_index + 1),
                "DOC_ID": doc.doc_id,
                "DOC_NAME": doc.doc_name,
                "DOC_TEXT": doc.doc_text,
                "CANDIDATE_LIST": candidate_list,
            },
        ).strip()
    else:
        # 第 1 轮自由提案；保留旧的候选池提示作为兜底（一般 allowed_keys=None）。
        candidate_block = (
            _candidate_pool_block(allowed_keys)
            if allowed_keys is not None and not previous_peers
            else ""
        )
        user_prompt = render_prompt_template(
            _read_md(str(propose_template_path)),
            {
                "EXPERT_NAME": expert_name,
                "ROUND_INDEX": str(round_index + 1),
                "DOC_ID": doc.doc_id,
                "DOC_NAME": doc.doc_name,
                "EVENTS_FULL": _events_json(doc),
                "DOC_TEXT": doc.doc_text,
                "HISTORY_BLOCK": ("\n" + history + "\n") if history else "",
                "CANDIDATE_POOL": ("\n" + candidate_block + "\n") if candidate_block else "",
            },
        ).strip()

    metadata = {
        "expert": expert_name,
        "doc_id": doc.doc_id,
    }
    return system_prompt, user_prompt, metadata


def build_judge_prompts(
    prompt_dir: Path,
    doc: DocumentInput,
    final_round_results: dict[str, "ExpertResult"],
) -> tuple[str, str, dict[str, Any]]:
    role_path = prompt_dir / "experts" / "judge.md"
    template_path = prompt_dir / "templates" / "judge_user.md"

    system_prompt = _read_md(str(role_path))
    user_prompt = render_prompt_template(
        _read_md(str(template_path)),
        {
            "DOC_ID": doc.doc_id,
            "DOC_NAME": doc.doc_name,
            "EVENTS_FULL": _events_json(doc),
            "DOC_TEXT": doc.doc_text,
            "EXPERT_RESULTS": _judge_candidates_json(final_round_results),
        },
    ).strip()

    metadata = {"doc_id": doc.doc_id}
    return system_prompt, user_prompt, metadata


__all__ = ["build_expert_prompts", "build_judge_prompts"]
