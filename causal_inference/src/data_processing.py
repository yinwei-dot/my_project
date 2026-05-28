from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.infra import configure_logging, extract_doc_id, log_file_only, write_json_utf8
else:
    from .infra import configure_logging, extract_doc_id, log_file_only, write_json_utf8


def collect_txt_by_id(folder: Path) -> tuple[dict[str, Path], list[str]]:
    mapping: dict[str, Path] = {}
    issues: list[str] = []
    for file_path in sorted(folder.glob("*.txt")):
        try:
            doc_id = extract_doc_id(file_path.name)
        except ValueError:
            issues.append(f"Cannot parse doc_id: {file_path.name}")
            continue
        if doc_id in mapping:
            issues.append(f"Duplicate doc_id in {folder.name}: {doc_id} -> {mapping[doc_id].name} | {file_path.name}")
            continue
        mapping[doc_id] = file_path
    return mapping, issues


def read_non_empty_lines(file_path: Path) -> list[str]:
    return [l.strip() for l in file_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_call_llm_dataset(
    cleaned_dir: Path,
    timeline_dir: Path,
    output_dir: Path,
    *,
    doc_ids: set[str] | None = None,
) -> dict[str, Any]:
    cleaned_dir, timeline_dir, output_dir = cleaned_dir.resolve(), timeline_dir.resolve(), output_dir.resolve()

    if not cleaned_dir.is_dir():
        raise FileNotFoundError(f"Invalid cleaned_dir: {cleaned_dir}")
    if not timeline_dir.is_dir():
        raise FileNotFoundError(f"Invalid timeline_dir: {timeline_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_map, cleaned_issues = collect_txt_by_id(cleaned_dir)
    timeline_map, timeline_issues = collect_txt_by_id(timeline_dir)
    common_ids = sorted(set(cleaned_map) & set(timeline_map))
    if doc_ids:
        common_ids = [d for d in common_ids if d in doc_ids]

    jsonl_path = output_dir / "all_docs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for doc_id in common_ids:
            events = read_non_empty_lines(timeline_map[doc_id])
            payload = {
                "doc_name": cleaned_map[doc_id].name,
                "doc_text": cleaned_map[doc_id].read_text(encoding="utf-8").strip(),
                "events": {f"E{i}": t for i, t in enumerate(events, 1)},
            }
            write_json_utf8(output_dir / f"{doc_id}.json", payload)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = {
        "written_docs": len(common_ids),
        "output_dir": str(output_dir),
        "jsonl": str(jsonl_path),
        "cleaned_count": len(cleaned_map),
        "timeline_count": len(timeline_map),
        "missing_in_timeline": sorted(set(cleaned_map) - set(timeline_map)),
        "missing_in_cleaned": sorted(set(timeline_map) - set(cleaned_map)),
        "cleaned_issues": cleaned_issues,
        "timeline_issues": timeline_issues,
        "doc_ids": common_ids,
    }
    write_json_utf8(output_dir / "build_summary.json", summary)
    return summary


# ── 入口 ──────────────────────────────────────────────────────────────────────

def prepare_data(
    cleaned_dir: Path,
    timeline_dir: Path,
    output_dir: Path,
    *,
    doc_ids: set[str] | None = None,
) -> dict[str, Any]:
    return build_call_llm_dataset(cleaned_dir, timeline_dir, output_dir, doc_ids=doc_ids)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="组织 call_llm 数据集。")
    parser.add_argument("--doc-ids", nargs="*", default=[], help="仅处理指定文档 ID，可传多个。")
    parser.add_argument("--cleaned-dir", type=Path, default=project_root / "data" / "cleaned")
    parser.add_argument("--timeline-dir", type=Path, default=project_root / "data" / "processed_timeline")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data" / "call_llm")
    parser.add_argument("--log-dir", type=Path, default=project_root / "log")
    args = parser.parse_args()

    doc_ids = {s.strip() for s in args.doc_ids if s.strip()}
    log_path = configure_logging(args.log_dir.resolve(), log_name="data_processing")
    logger = logging.getLogger("data_processing")

    build_summary = prepare_data(
        args.cleaned_dir, args.timeline_dir, args.output_dir,
        doc_ids=doc_ids or None,
    )
    log_file_only(logger, logging.INFO,
        "data_processing 完成：written_docs=%s doc_ids=%s",
        build_summary.get("written_docs", 0), sorted(doc_ids) if doc_ids else "ALL",
    )
    print(f"处理完成。call_llm 输出文档数：{build_summary.get('written_docs', 0)}")
    print(f"日志：{log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())