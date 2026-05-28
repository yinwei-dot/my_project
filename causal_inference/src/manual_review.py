"""manual_review.py — 人工审核本地服务，为可视化 HTML 提供 /save_graph 端点。

修改流程
---------
1. 可选手工启动服务：
       cd causal_inference_v5_分类版
       python src/manual_review.py --doc-id D013
   或指定完整路径：
       python src/manual_review.py --graph-path output/doc_results/D013/graph.json

2. 服务启动后会自动打开对应的 graph.html（浏览器）。
3. 在浏览器中点击"修改"按钮进入编辑模式，点击边进行删除或类型转换。
4. 点击"保存"按钮：页面会优先连接自动拉起的本地保存服务，graph.json 写回后同步重渲染 graph.html；
    若服务短暂未就绪，前端会自动等待并重试，仍失败才降级为下载 graph.json 备份。

ManualReviewServer
------------------
亦可在代码中直接实例化：

    from src.manual_review import ManualReviewServer
    from pathlib import Path
    ManualReviewServer(Path("output/doc_results/D013/graph.json")).start()
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

_DEFAULT_PORT = 7760
_SERVICE_VERSION = "manual-review-v2"
_EDGE_TYPE_LABELS = {"causal": "因果边", "temporal": "时序边"}


def _edge_key(e: dict[str, Any], is_causal: bool) -> tuple[str, str]:
    if is_causal:
        return (str(e.get("cause", "")), str(e.get("effect", "")))
    return (str(e.get("source", "")), str(e.get("target", "")))


def _edge_label(key: tuple[str, str], edge_type: str) -> str:
    return f"{_edge_type_name(edge_type)} {key[0]} -> {key[1]}"


def _edge_type_name(edge_type: str) -> str:
    return _EDGE_TYPE_LABELS.get(edge_type, edge_type)


def _event_sort_value(value: str) -> tuple[int, int | str]:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return (0, int(digits))
    return (1, text)


def _edge_sort_key(key: tuple[str, str]) -> tuple[tuple[int, int | str], tuple[int, int | str]]:
    return (_event_sort_value(key[0]), _event_sort_value(key[1]))


def _collect_edges(data: dict[str, Any]) -> dict[tuple[str, str], tuple[str, dict[str, Any]]]:
    edges: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for edge in data.get("causal_edges", []) or []:
        if isinstance(edge, dict):
            edges[_edge_key(edge, True)] = ("causal", edge)
    for edge in data.get("temporal_edges", []) or []:
        if isinstance(edge, dict):
            edges[_edge_key(edge, False)] = ("temporal", edge)
    return edges


def _format_review_record(record: dict[str, Any]) -> str:
    source = str(record.get("source", ""))
    target = str(record.get("target", ""))
    ts = str(record.get("timestamp", ""))
    action = record.get("action")
    if action == "modified":
        return (
            f"{_edge_type_name(str(record.get('old_type', '')))} {source} -> {target} "
            f"修改为 {_edge_type_name(str(record.get('new_type', '')))} [{ts}]"
        )
    if action == "added":
        return f"新增 {_edge_type_name(str(record.get('edge_type', '')))} {source}->{target} [{ts}]"
    if action == "deleted":
        return f"删除 {_edge_type_name(str(record.get('edge_type', '')))} {source}->{target} [{ts}]"
    return f"{action or 'unknown'} {source}->{target} [{ts}]"


def _manual_review_records(
    old_data: dict[str, Any],
    new_data: dict[str, Any],
) -> list[dict[str, Any]]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    old_all = _collect_edges(old_data)
    new_all = _collect_edges(new_data)
    records: list[dict[str, Any]] = []

    for key in sorted(old_all.keys(), key=_edge_sort_key):
        old_type, old_edge = old_all[key]
        new_entry = new_all.get(key)
        if new_entry is None:
            records.append(
                {
                    "action": "deleted",
                    "edge_type": old_type,
                    "source": key[0],
                    "target": key[1],
                    "reason": old_edge.get("reason", ""),
                    "timestamp": timestamp,
                }
            )
            continue
        new_type, new_edge = new_entry
        if new_type != old_type:
            records.append(
                {
                    "action": "modified",
                    "old_type": old_type,
                    "new_type": new_type,
                    "source": key[0],
                    "target": key[1],
                    "reason": new_edge.get("reason", ""),
                    "timestamp": timestamp,
                }
            )

    for key in sorted(new_all.keys(), key=_edge_sort_key):
        if key in old_all:
            continue
        new_type, new_edge = new_all[key]
        records.append(
            {
                "action": "added",
                "edge_type": new_type,
                "source": key[0],
                "target": key[1],
                "reason": new_edge.get("reason", ""),
                "timestamp": timestamp,
            }
        )

    for record in records:
        record["summary"] = _format_review_record(record)
    return records


def _merge_manual_review_edges(
    old_data: dict[str, Any],
    new_data: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    history: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in (old_data.get("manual_review_edges"), new_data.get("manual_review_edges")):
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            marker = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if marker in seen:
                continue
            history.append(item)
            seen.add(marker)
    history.extend(records)
    new_data["manual_review_edges"] = history


def _preserve_edge_metadata(old_data: dict[str, Any], new_data: dict[str, Any]) -> None:
    old_edges = _collect_edges(old_data)
    for edge_type, field_name, is_causal in (
        ("causal", "causal_edges", True),
        ("temporal", "temporal_edges", False),
    ):
        for edge in new_data.get(field_name, []) or []:
            if not isinstance(edge, dict):
                continue
            old_entry = old_edges.get(_edge_key(edge, is_causal))
            if old_entry is None:
                continue
            old_type, old_edge = old_entry
            if old_type != edge_type:
                continue
            for key, value in old_edge.items():
                edge.setdefault(key, value)


def _append_manual_review_summary(
    graph_path: Path,
    records: list[dict[str, Any]],
) -> None:
    """将人工审核记录追加到 discussion_summary.txt。"""
    if not records:
        return
    # 找到 discussion_summary.txt 的位置
    doc_id = graph_path.parent.name
    # graph.json 在 output/doc_results/DOC_ID/graph.json
    # discussion_summary 在 output/discussion_traces/DOC_ID_discussion_summary.txt
    traces_dir = graph_path.parents[2] / "discussion_traces"
    # 找匹配的 summary 文件
    summary_files = list(traces_dir.glob(f"{doc_id}_discussion_summary.txt"))
    if not summary_files:
        # 尝试宽松匹配
        summary_files = list(traces_dir.glob(f"*_discussion_summary.txt"))
        summary_files = [f for f in summary_files if doc_id in f.stem]
    if not summary_files:
        print(f"[ManualReview] 未找到 discussion_summary.txt，跳过追加审核摘要")
        return
    summary_path = summary_files[0]

    section_lines = ["manual_review"]
    section_lines.extend(str(record.get("summary") or _format_review_record(record)) for record in records)

    with summary_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(section_lines) + "\n")
    print(f"[ManualReview] 已追加审核摘要 → {summary_path}")

def _output_html_path(graph_path: Path) -> Path:
    """根据 graph.json 路径定位对应的 graph.html。"""
    doc_id = graph_path.parent.name
    return graph_path.parents[2] / "visualization" / doc_id / "graph.html"

def _regen_html(graph_path: Path) -> Path:
    """保存后同步调用 render_graph()，确保 graph.html 与 graph.json 立即一致。"""
    output_html = _output_html_path(graph_path)
    try:
        try:
            from .render_graph import render_graph
        except ImportError:
            from render_graph import render_graph
        render_graph(graph_path, output_html)
        print(f"[ManualReview] HTML 已重新生成 → {output_html}")
        return output_html
    except Exception as exc:
        print(f"[ManualReview] HTML 重生成失败: {exc}")
        raise


def _resolve_graph_path(default_path: Path | None, raw_path: str = "") -> Path:
    """优先使用请求中携带的路径，否则回退到服务启动时配置的路径。"""
    if raw_path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        if default_path is not None:
            return (default_path.parents[3] / candidate).resolve()
        return candidate.resolve()
    if default_path is not None:
        return default_path
    raise ValueError("未配置 graph.json 路径")


_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


class _Handler(BaseHTTPRequestHandler):
    """轻量 HTTP 处理器，只支持 OPTIONS 预检和 POST /save_graph。"""

    server: "ManualReviewServer"  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # 抑制默认访问日志

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        for k, v in _CORS.items():
            self.send_header(k, v)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for k, v in _CORS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send(
                200,
                json.dumps(
                    {"ok": True, "service": _SERVICE_VERSION, "graph_path": str(getattr(self.server, "graph_path", "") or "")},
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
            return
        if parsed.path != "/load_graph":
            self._send(404, b'{"ok":false,"error":"not found"}')
            return
        try:
            query = parse_qs(parsed.query)
            raw_path = query.get("path", [""])[0]
            graph_path = _resolve_graph_path(getattr(self.server, "graph_path", None), raw_path)
            if not graph_path.exists():
                raise FileNotFoundError(f"graph.json 不存在: {graph_path}")
            data = json.loads(graph_path.read_text(encoding="utf-8"))
            self._send(
                200,
                json.dumps({"ok": True, "data": data}, ensure_ascii=False).encode("utf-8"),
            )
        except Exception as exc:
            print(f"[ManualReview] 读取 graph.json 失败: {exc}")
            self._send(500, json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False).encode("utf-8"))

    def do_POST(self) -> None:
        if self.path != "/save_graph":
            self._send(404, b'{"ok":false,"error":"not found"}')
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body)
            data = payload.get("data")
            if data is None:
                raise ValueError("payload 缺少 data 字段")
            if not isinstance(data, dict):
                raise ValueError("payload.data 必须是 JSON 对象")

            # 优先使用 HTML 传来的路径（当前文档），其次使用服务端配置的路径
            save_path = _resolve_graph_path(
                getattr(self.server, "graph_path", None),
                str(payload.get("path", "")),
            )

            # 读取旧数据用于 diff
            old_data: dict[str, Any] = {}
            if save_path.exists():
                try:
                    old_data = json.loads(save_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            _preserve_edge_metadata(old_data, data)
            review_records = _manual_review_records(old_data, data)
            _merge_manual_review_edges(old_data, data, review_records)

            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[ManualReview] 已保存 → {save_path}")

            # 将 diff 追加到 discussion_summary.txt
            try:
                _append_manual_review_summary(save_path, review_records)
            except Exception as exc:
                print(f"[ManualReview] 追加审核摘要失败: {exc}")

            # 同步重新生成 HTML，保证刷新页面时看到的就是最新网络
            output_html = _regen_html(save_path)

            # 把保存后的 data 返回，让浏览器直接重载可视化（无需刷新页面）
            self._send(
                200,
                json.dumps(
                    {"ok": True, "data": data, "html_path": str(output_html)},
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
        except Exception as exc:
            print(f"[ManualReview] 保存失败: {exc}")
            self._send(500, json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False).encode("utf-8"))


class ManualReviewServer(HTTPServer):
    """本地 HTTP 服务，监听指定端口，为可视化 HTML 提供 graph.json 写回端点。

    Parameters
    ----------
    graph_path : Path | None
        graph.json 的绝对路径。若为 None，则从请求体的 path 字段获取。
    port : int
        监听端口，默认 7760（与 render_graph._SAVE_PORT 保持一致）。
    """

    def __init__(self, graph_path: Path | None = None, port: int = _DEFAULT_PORT) -> None:
        super().__init__(("localhost", port), _Handler)
        self.graph_path: Path | None = graph_path

    def start(self, *, open_browser: bool = True) -> None:
        """启动服务（阻塞，Ctrl+C 停止）。"""
        port = self.server_address[1]
        print(f"[ManualReview] 服务已启动 → http://localhost:{port}")

        # 尝试定位 graph.html
        html_path = self._find_html()
        if html_path:
            print(f"[ManualReview] 可视化文件 → {html_path}")
            if open_browser:
                webbrowser.open(html_path.as_uri())
        else:
            print("[ManualReview] 未找到 graph.html，请手动打开浏览器")

        print("[ManualReview] 按 Ctrl+C 停止服务")
        try:
            self.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.server_close()
            print("[ManualReview] 服务已停止")

    def _find_html(self) -> Path | None:
        if self.graph_path is None:
            return None
        doc_id = self.graph_path.parent.name
        root = self.graph_path.parents[3]  # output/doc_results/DOC_ID/graph.json -> root
        output_root = self.graph_path.parents[2]
        candidates = [
            output_root / "visualization" / doc_id / "graph.html",
            root / "visualization" / doc_id / "graph.html",
            self.graph_path.parent / "graph.html",
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_graph_path(doc_id: str | None, graph_path_str: str | None, root: Path) -> Path | None:
    if graph_path_str:
        return Path(graph_path_str).resolve()
    if doc_id:
        candidates = [
            root / "output" / "doc_results" / doc_id / "graph.json",
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="人工审核本地 HTTP 服务")
    parser.add_argument("--doc-id", default=None, help="文档 ID（如 D013）")
    parser.add_argument("--graph-path", default=None, help="graph.json 完整路径")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT, help=f"监听端口（默认 {_DEFAULT_PORT}）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    graph_path = _find_graph_path(args.doc_id, args.graph_path, root)
    if graph_path is None:
        print("[ManualReview] 警告：未找到 graph.json，将仅启动服务（由 HTML 提供路径）")

    ManualReviewServer(graph_path=graph_path, port=args.port).start(
        open_browser=not args.no_browser
    )


if __name__ == "__main__":
    main()

__all__ = ["ManualReviewServer"]
