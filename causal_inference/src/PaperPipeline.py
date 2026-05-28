"""LLM 主流程：四专家多轮讨论 + Judge 裁决，输出因果边。

业务边界：
1. 只做 LLM 调用与结果落盘；prompt 拼接交给 prompting.py，规范化交给 models.py。
2. 边集合的合法性约束统一为：cause 事件编号必须严格小于 effect。
3. checkpoint / resume / 进度条 / TraceExporter 等机制走 infra.py。
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from .infra import (
    AppConfig,
    BatchProgressDisplay,
    OpenAICompatibleClient,
    SlidingWindowRateLimiter,
    TraceExporter,
    checkpoint_matches,
    create_checkpoint,
    discover_input_files,
    event_id_num,
    load_checkpoint,
    log_file_only,
    mark_expert_completed,
    mark_judge_completed,
    read_json_utf8,
    save_checkpoint,
    sha256_file,
    write_json_utf8,
)
from .models import (
    DocumentInput,
    Edge,
    ExpertResult,
    JudgeResult,
    build_candidate_pool,
    expert_result_from_dict,
    load_document_input,
    normalize_expert_classification_payload,
    normalize_expert_payload,
    normalize_judge_payload,
)
from .prompting import build_expert_prompts, build_judge_prompts
from .temporal_edge_builder import _get_embedding_model

# ── 辅助 ────────────────────────────────────────────────────────────────────────────

def _edge_keys_union(results: dict[str, ExpertResult]) -> set[tuple[str, str]]:
    """计算某轮所有专家边集的并集。用于生成下一轮的候选边集约束。"""
    keys: set[tuple[str, str]] = set()
    for result in results.values():
        for edge in result.edges:
            keys.add(edge.key)
    return keys


def _experts_agree(results: dict[str, ExpertResult]) -> bool:
    """判断本轮 discourse 与 commonsense 的因果边集是否完全一致。

    两者一致说明已收敛，可以提前停止讨论轮次。
    若任一专家缺席则视为未收敛。
    """
    if "discourse" not in results or "commonsense" not in results:
        return False
    return (
        {e.key for e in results["discourse"].edges}
        == {e.key for e in results["commonsense"].edges}
    )


def _collect_llm_temporal_edges(
    judge_result: "JudgeResult",
    final_round_results: dict[str, "ExpertResult"],
) -> tuple[list[Any], list[int]]:
    """按优先级收集 LLM 分类的时序边：Judge > discourse > commonsense。

    返回 (edges, priorities) 两个平行列表：
    - edges: TemporalEdge 列表，同一边只保留优先级最高来源的理由。
    - priorities: 与 edges 平行，0=Judge, 1=discourse, 2=commonsense。
    """
    from .temporal_edge_builder import TemporalEdge
    result: list[TemporalEdge] = []
    priorities: list[int] = []
    seen: set[tuple[str, str]] = set()

    def _add(source: str, target: str, reason: str, priority: int) -> None:
        src, tgt = (source, target) if event_id_num(source) <= event_id_num(target) else (target, source)
        if (src, tgt) not in seen:
            seen.add((src, tgt))
            result.append(TemporalEdge(
                source=src, target=tgt, reason=reason, score=0.0,
                span=abs(event_id_num(src) - event_id_num(tgt)),
            ))
            priorities.append(priority)

    # 1. Judge temporal（最高优先级）
    for edge in judge_result.temporal_kept:
        _add(edge.cause, edge.effect, edge.reason, 0)

    # 2. discourse temporal
    er_disc = final_round_results.get("discourse")
    if er_disc is not None:
        for edge in er_disc.temporal_edges:
            _add(edge.cause, edge.effect, edge.reason, 1)

    # 3. commonsense temporal
    er_comm = final_round_results.get("commonsense")
    if er_comm is not None:
        for edge in er_comm.temporal_edges:
            _add(edge.cause, edge.effect, edge.reason, 2)

    return result, priorities


# ── 单步 LLM 调用 ───────────────────────────────────────────────────────────

def _call_expert(
    *,
    config: AppConfig,
    client: OpenAICompatibleClient,
    doc: DocumentInput,
    expert_name: str,
    round_index: int,
    previous_self: ExpertResult | None,
    previous_peers: dict[str, ExpertResult],
    allowed_keys: set[tuple[str, str]] | None = None,
) -> tuple[ExpertResult, dict[str, Any]]:
    system_prompt, user_prompt, metadata = build_expert_prompts(
        config.prompt_dir, doc, expert_name, round_index, previous_self, previous_peers,
        allowed_keys=allowed_keys,
    )
    label = f"{doc.doc_id} R{round_index} {expert_name}"
    llm_result = client.chat_json(
        model=config.expert_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        request_label=label,
        max_tokens=config.expert_max_tokens,
    )
    if not llm_result.success or not isinstance(llm_result.parsed_json, dict):
        raise RuntimeError(f"{label} 调用失败: {llm_result.error or 'LLM 输出不是合法 JSON'}")

    # 第 1 轮（自由提案）：用通用 ExpertResult schema 解析。
    # 第 2+ 轮：候选边二分类（参照 Judge 的输入输出），按 edgeN 字典解析。
    if round_index >= 1 and allowed_keys is not None:
        result = normalize_expert_classification_payload(
            llm_result.parsed_json, expert_name, allowed_keys,
        )
    else:
        effective_keys = allowed_keys if allowed_keys is not None else doc.forward_edge_keys
        result = normalize_expert_payload(llm_result.parsed_json, expert_name, effective_keys)
    trace = {
        "expert": expert_name,
        "round": round_index,
        "model": config.expert_model,
        "prompt_metadata": metadata,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": llm_result.raw_text,
        "parsed_result": result.to_dict(),
    }
    return result, trace


def _call_judge(
    *,
    config: AppConfig,
    client: OpenAICompatibleClient,
    doc: DocumentInput,
    final_round_results: dict[str, ExpertResult],
    candidate_pool: list[dict[str, Any]],
) -> tuple[JudgeResult, dict[str, Any]]:
    """调用 Judge LLM，以各专家原始边集送审，candidate_pool 仅用于输出规范化。"""
    system_prompt, user_prompt, metadata = build_judge_prompts(
        config.prompt_dir, doc, final_round_results,
    )
    label = f"{doc.doc_id} Judge"
    llm_result = client.chat_json(
        model=config.judge_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        request_label=label,
        max_tokens=config.judge_max_tokens,
        timeout=config.judge_timeout,
    )
    if not llm_result.success or not isinstance(llm_result.parsed_json, dict):
        raise RuntimeError(f"{label} 调用失败: {llm_result.error or 'LLM 输出不是合法 JSON'}")

    judge_result = normalize_judge_payload(
        llm_result.parsed_json,
        candidate_pool,
        # Judge 将输出限制在候选池键内，不得新增 pool 之外的边
        {(c["cause"], c["effect"]) for c in candidate_pool} if candidate_pool else doc.forward_edge_keys,
    )
    trace = {
        "model": config.judge_model,
        "prompt_metadata": metadata,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": llm_result.raw_text,
        "parsed_result": judge_result.to_dict(),
    }
    return judge_result, trace


# ── Pipeline ────────────────────────────────────────────────────────────────

class PaperPipeline:
    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

        def _make_limiter(rpm: int, tpm: int) -> SlidingWindowRateLimiter | None:
            if rpm > 0 or tpm > 0:
                return SlidingWindowRateLimiter(
                    rpm_limit=rpm,
                    tpm_limit=tpm,
                    window_seconds=config.rate_limit_window_seconds,
                    logger=logger,
                )
            return None

        self.expert_client = OpenAICompatibleClient(
            config, logger,
            rate_limiter=_make_limiter(config.expert_rpm_limit, config.expert_tpm_limit),
        )
        self.judge_client = OpenAICompatibleClient(
            config, logger,
            rate_limiter=_make_limiter(config.judge_rpm_limit, config.judge_tpm_limit),
        )
        self.trace_exporter = TraceExporter(config.output_dir)

    # ── 入口 ────────────────────────────────────────────────────────────────

    def run(self) -> list[dict[str, Any]]:
        files = discover_input_files(self.config)
        steps_per_doc = sum(
            len(self.config.experts_for_round(r)) for r in range(self.config.rounds)
        ) + 1
        progress = BatchProgressDisplay(
            total_docs=len(files),
            total_steps=max(1, len(files) * steps_per_doc),
        )
        # 提前加载嵌入模型，避免在文档处理中途触发下载/初始化日志
        self.logger.info("预加载语义嵌入模型…")
        _get_embedding_model()
        self.logger.info("嵌入模型就绪。")

        workers = self.config.workers
        if workers <= 1:
            summaries: list[dict[str, Any]] = []
            for path in files:
                try:
                    summaries.append(self.run_document(path, progress=progress))
                except Exception as exc:  # noqa: BLE001
                    self.logger.error("[%s] 处理失败，跳过：%s", path.stem, exc)
                    summaries.append({"doc_id": path.stem, "status": "failed", "error": str(exc)})
                    if progress is not None:
                        progress.skip_doc(path.stem, steps_per_doc, reason="失败跳过")
            return summaries

        # 并发模式
        futures_map = {}
        summaries_map: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for path in files:
                fut = executor.submit(self.run_document, path, progress=progress)
                futures_map[fut] = path
            for fut in as_completed(futures_map):
                path = futures_map[fut]
                try:
                    result = fut.result()
                    summaries_map[path.stem] = result
                except Exception as exc:  # noqa: BLE001
                    self.logger.error("[%s] 处理失败，跳过：%s", path.stem, exc)
                    summaries_map[path.stem] = {"doc_id": path.stem, "status": "failed", "error": str(exc)}
                    progress.skip_doc(path.stem, steps_per_doc, reason="失败跳过")
        # 按原始顺序返回
        return [summaries_map[path.stem] for path in files if path.stem in summaries_map]

    def run_document(self, input_path: Path, *, progress: BatchProgressDisplay | None = None) -> dict[str, Any]:
        doc = load_document_input(input_path)
        paths = self._doc_paths(doc.doc_id)

        steps_per_doc = sum(
            len(self.config.experts_for_round(r)) for r in range(self.config.rounds)
        ) + 1

        source_hash = sha256_file(input_path)
        loaded_ckpt = load_checkpoint(paths["checkpoint"])
        ckpt_hit = checkpoint_matches(
            loaded_ckpt,
            source_hash=source_hash,
        )
        if (
            self.config.resume
            and ckpt_hit
            and cast(dict, loaded_ckpt).get("judge_done")
            and paths["graph"].exists()
        ):
            log_file_only(self.logger, logging.INFO, "[%s] 已完成，按 resume 跳过。", doc.doc_id)
            if progress is not None:
                progress.skip_doc(doc.doc_id, steps_per_doc, reason="缓存命中")
            return {"doc_id": doc.doc_id, "status": "skipped"}

        if self.config.resume and ckpt_hit:
            checkpoint = cast(dict, loaded_ckpt)
        else:
            checkpoint = create_checkpoint(source_hash=source_hash, run_signature=self.config.run_signature)

        rounds_payload: dict[str, Any] = {"rounds": []}

        write_json_utf8(paths["run_config"], self.config.to_dict())
        write_json_utf8(paths["normalized_input"], doc.to_dict())

        if progress is not None:
            progress.start_doc(doc.doc_id, steps_per_doc)

        # 多轮专家（第 1 轮独立提案 + 第 2+ 轮基于上轮结果更新）
        self._run_expert_rounds(doc, rounds_payload, checkpoint, paths, progress=progress)

        # ── 最后一轮边集送 Judge 裁决 ────────────────────────────────────────
        if progress is not None:
            progress.set_stage("Judge 裁决")

        final_round_results = self._final_round_results(doc, rounds_payload)
        candidate_pool = build_candidate_pool(final_round_results)

        if candidate_pool:
            judge_result, judge_trace = _call_judge(
                config=self.config, client=self.judge_client, doc=doc,
                final_round_results=final_round_results,
                candidate_pool=candidate_pool,
            )
        else:
            judge_result = JudgeResult(kept=[], rejected=[])
            judge_trace = {
                "note": "候选边为空，跳过 Judge。",
                "parsed_result": judge_result.to_dict(),
            }

        mark_judge_completed(checkpoint)
        save_checkpoint(paths["checkpoint"], checkpoint)
        if progress is not None:
            progress.advance_step("Judge 裁决")

        # 导出 trace + 渲染（内部写 final_edges.json 和 graph.json）
        self._export_artifacts(doc, rounds_payload, judge_result, judge_trace, paths, final_round_results)

        summary = {
            "doc_id": doc.doc_id,
            "status": "completed",
            "final_edge_count": len(judge_result.final_edges),
            "judge_kept_count": len(judge_result.kept),
            "candidate_pool_size": len(candidate_pool),
            "output_dir": str(paths["doc_dir"]),
            "html": str(paths["html"]),
        }
        if progress is not None:
            progress.finish_doc(stage_label="完成")
        log_file_only(
            self.logger, logging.INFO,
            "[%s] 完成，最终边数=%s（Judge保留=%s），候选数=%s",
            doc.doc_id, summary["final_edge_count"],
            summary["judge_kept_count"],
            summary["candidate_pool_size"],
        )
        return summary

    # ── 辅助 ────────────────────────────────────────────────────────────────

    def _run_expert_rounds(
        self,
        doc: DocumentInput,
        rounds_payload: dict[str, Any],
        checkpoint: dict[str, Any],
        paths: dict[str, Path],
        *,
        progress: BatchProgressDisplay | None = None,
    ) -> None:
        rounds = rounds_payload.setdefault("rounds", [])

        for round_index in range(self.config.rounds):
            # ── 初始化本轮 entry ──────────────────────────────────────────────
            while len(rounds) <= round_index:
                rounds.append({"round": round_index, "expert_results": {}, "expert_traces": {}})
            entry = rounds[round_index]

            # 本轮应参与的专家列表
            round_expert_list = list(self.config.experts_for_round(round_index))

            # 取上一轮结果作为历史（第 1 轮无历史）
            if round_index > 0:
                prev_results = self._round_results(doc, rounds[round_index - 1], round_index - 1)
            else:
                prev_results: dict[str, ExpertResult] = {}

            # Round 1+ 将候选边限制在上轮结果的并集内
            allowed_keys: set[tuple[str, str]] | None = (
                _edge_keys_union(prev_results) if round_index > 0 else None
            )

            pending_experts = [
                name for name in round_expert_list
                if not (self.config.resume and name in entry.get("expert_results", {}))
            ]

            def _call_proposal(
                expert_name: str,
                _prev: dict[str, ExpertResult] = prev_results,
                _ri: int = round_index,
                _allowed: set[tuple[str, str]] | None = allowed_keys,
            ) -> tuple[str, Any, dict[str, Any]]:
                previous_self = _prev.get(expert_name)
                previous_peers = {k: v for k, v in _prev.items() if k != expert_name}
                res, tr = _call_expert(
                    config=self.config, client=self.expert_client, doc=doc,
                    expert_name=expert_name, round_index=_ri,
                    previous_self=previous_self, previous_peers=previous_peers,
                    allowed_keys=_allowed,
                )
                return expert_name, res, tr

            self._run_experts_parallel(
                pending_experts, _call_proposal,
                entry, "expert_results", "expert_traces",
                round_index, checkpoint, paths, rounds_payload, progress,
                mark_fn=mark_expert_completed,
            )

            checkpoint["last_completed_round"] = round_index
            save_checkpoint(paths["checkpoint"], checkpoint)

            # ── 提前停止：当 discourse 与 commonsense 因果边集完全一致时收敛 ────
            if round_index > 0 and round_index < self.config.rounds - 1:
                curr_results = self._round_results(doc, entry, round_index)
                if _experts_agree(curr_results):
                    log_file_only(
                        self.logger, logging.INFO,
                        "[%s] R%s discourse 与 commonsense 因果边一致，提前停止讨论。",
                        doc.doc_id, round_index,
                    )
                    break

    def _run_experts_parallel(
        self,
        pending_experts: list[str],
        call_fn,   # (expert_name) -> (expert_name, result, trace)
        entry: dict[str, Any],
        results_key: str,
        traces_key: str,
        round_index: int,
        checkpoint: dict[str, Any],
        paths: dict[str, Path],
        rounds_payload: dict[str, Any],
        progress: BatchProgressDisplay | None,
        *,
        mark_fn,   # mark_expert_completed
    ) -> None:
        """串行或并发执行一轮内所有专家调用，统一落盘与进度更新。"""
        if self.config.expert_workers <= 1 or len(pending_experts) <= 1:
            for idx, expert_name in enumerate(pending_experts, start=1):
                stage = f"R{round_index + 1}·{idx}/{len(pending_experts)}·{expert_name}"
                if progress is not None:
                    progress.set_stage(stage)
                try:
                    _, result, trace = call_fn(expert_name)
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(
                        "[%s] R%s %s 调用失败，跳过（结果为空）: %s",
                        entry.get("round", round_index), round_index, expert_name, exc,
                    )
                    if progress is not None:
                        progress.advance_step(stage)
                    continue
                entry.setdefault(results_key, {})[expert_name] = result.to_dict()
                entry.setdefault(traces_key, {})[expert_name] = trace
                mark_fn(checkpoint, round_index, expert_name)
                save_checkpoint(paths["checkpoint"], checkpoint)
                if progress is not None:
                    progress.advance_step(stage)
        else:
            write_lock = threading.Lock()
            stage_label = f"R{round_index + 1}·并发·{len(pending_experts)}专家"
            if progress is not None:
                progress.set_stage(stage_label)
            with ThreadPoolExecutor(max_workers=self.config.expert_workers) as ex:
                futures = {ex.submit(call_fn, name): name for name in pending_experts}
                for fut in as_completed(futures):
                    expert_name = futures[fut]
                    try:
                        _, result, trace = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        self.logger.warning(
                            "[%s] R%s %s 调用失败，跳过（结果为空）: %s",
                            entry.get("round", round_index), round_index, expert_name, exc,
                        )
                        if progress is not None:
                            progress.advance_step(f"R{round_index + 1}·{expert_name}·失败")
                        continue
                    with write_lock:
                        entry.setdefault(results_key, {})[expert_name] = result.to_dict()
                        entry.setdefault(traces_key, {})[expert_name] = trace
                        mark_fn(checkpoint, round_index, expert_name)
                        save_checkpoint(paths["checkpoint"], checkpoint)
                    if progress is not None:
                        progress.advance_step(f"R{round_index + 1}·{expert_name}·完成")

    def _final_round_results(self, doc: DocumentInput, rounds_payload: dict[str, Any]) -> dict[str, ExpertResult]:
        last_index = len(rounds_payload["rounds"]) - 1
        last_entry = rounds_payload["rounds"][last_index]
        return self._round_results(doc, last_entry, last_index)

    def _round_results(self, doc: DocumentInput, entry: dict[str, Any], round_index: int) -> dict[str, ExpertResult]:
        forward_keys = doc.forward_edge_keys
        out: dict[str, ExpertResult] = {}
        for name in self.config.experts_for_round(round_index):
            payload = entry.get("expert_results", {}).get(name)
            if payload is not None:
                out[name] = expert_result_from_dict(payload, name, forward_keys)
        return out

    def _doc_paths(self, doc_id: str) -> dict[str, Path]:
        doc_dir = self.config.output_dir / "doc_results" / doc_id
        vis_dir = self.config.output_dir / "visualization" / doc_id
        return {
            "doc_dir": doc_dir,
            "vis_dir": vis_dir,
            "run_config": doc_dir / "run_config.json",
            "checkpoint": doc_dir / "checkpoint.json",
            "normalized_input": doc_dir / "normalized_input.json",
            "graph": doc_dir / "graph.json",
            "html": vis_dir / "graph.html",
        }

    def _export_artifacts(
        self,
        doc: DocumentInput,
        rounds_payload: dict[str, Any],
        judge_result: JudgeResult,
        judge_trace: dict[str, Any],
        paths: dict[str, Path],
        final_round_results: dict[str, ExpertResult],
    ) -> None:
        bundle = {
            "doc_id": doc.doc_id,
            "doc_name": doc.doc_name,
            "doc_text": doc.doc_text,
            "rounds": rounds_payload["rounds"],
            "judge": judge_trace,
        }
        self.trace_exporter.export_json(doc.doc_id, bundle)
        self.trace_exporter.export_text(doc.doc_id, bundle)
        self.trace_exporter.export_summary(doc.doc_id, bundle)

        # 构造时序边并与因果边一起落盘为 graph.json
        from .temporal_edge_builder import TemporalEdgeBuilder
        from .render_graph import render_graph

        llm_temporal_edges, llm_temporal_priorities = _collect_llm_temporal_edges(judge_result, final_round_results)
        try:
            builder_temporal = TemporalEdgeBuilder().build(
                doc, list(judge_result.final_edges), llm_temporal_edges, llm_temporal_priorities,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("[%s] 时序边构造失败，仅输出因果边：%s", doc.doc_id, exc)
            builder_temporal = []

        all_temporal_dicts = [te.to_dict() for te in builder_temporal]

        # graph.json 包含全部信息（因果边 + 时序边 + 事件 + 全文）
        graph_payload = {
            "doc_id": doc.doc_id,
            "doc_name": doc.doc_name,
            "doc_text": doc.doc_text,
            "events": [event.to_dict() for event in doc.events],
            "causal_edges": [edge.to_dict() for edge in judge_result.final_edges],
            "temporal_edges": all_temporal_dicts,
            "manual_review_edges": [],
        }
        write_json_utf8(paths["graph"], graph_payload)
        paths["vis_dir"].mkdir(parents=True, exist_ok=True)
        render_graph(paths["graph"], paths["html"])


def run_pipeline(config: AppConfig, logger: logging.Logger) -> list[dict[str, Any]]:
    return PaperPipeline(config, logger).run()


__all__ = ["PaperPipeline", "run_pipeline"]
