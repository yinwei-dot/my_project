"""key_nodes/render.py — 读 graph.json + 拓扑/语义分，渲染只读可视化 HTML。

设计原则：
- 本文件自包含，不依赖 causal_inference 模块。
- 节点按综合得分着色：低分浅蓝 (#dbeeff) → 高分橙红 (#d62728)。
- 因果边：红色实线；时序边：蓝色虚线。
- 右侧面板显示得分排名表格。
- 不含任何编辑/保存功能。
- 使用 Path.write_text(encoding='utf-8') 规避 Windows GBK 问题。
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any, cast

# ── 颜色常量 ─────────────────────────────────────────────────────────────────
CAUSAL_COLOR = "#d62728"
TEMPORAL_COLOR = "#1f77b4"
NODE_FONT_SIZE = 12
NODE_WRAP_CHARS = 14

# 低分 → 高分颜色（RGB）
_COLOR_LOW = (219, 238, 255)   # #dbeeff
_COLOR_HIGH = (214,  39,  40)  # #d62728


# ── 颜色插值 ─────────────────────────────────────────────────────────────────

def _score_to_color(score: float) -> str:
    """线性插值: 0.0 → #dbeeff, 1.0 → #d62728。"""
    t = max(0.0, min(1.0, float(score)))
    r = int(_COLOR_LOW[0] + t * (_COLOR_HIGH[0] - _COLOR_LOW[0]))
    g = int(_COLOR_LOW[1] + t * (_COLOR_HIGH[1] - _COLOR_LOW[1]))
    b = int(_COLOR_LOW[2] + t * (_COLOR_HIGH[2] - _COLOR_LOW[2]))
    return f"#{r:02x}{g:02x}{b:02x}"


# ── 节点标签 ─────────────────────────────────────────────────────────────────

def _node_label(event_id: str, text: str, combined: float, combined_rank: int,
                wrap: int = NODE_WRAP_CHARS) -> str:
    """详细模式标签：'E3: 事件描述\n▲ 0.84 (#1)'"""
    raw = f"{event_id}: {text.strip().replace(chr(10), ' ')}"
    lines: list[str] = []
    while len(raw) > wrap:
        lines.append(raw[:wrap])
        raw = raw[wrap:]
    if raw:
        lines.append(raw)
    lines.append(f"\u25b2 {combined:.2f} (#{combined_rank})")
    return "\n".join(lines)


def _node_label_simple(event_id: str, combined: float) -> str:
    """简洁模式标签：'E3 \u2605 0.84'"""
    return f"{event_id} \u2605{combined:.2f}"


# ── 综合得分计算 ──────────────────────────────────────────────────────────────

def _compute_combined(
    topo_scores: dict[str, dict],   # {node_id: {"score": float, "rank": int}}
    sem_scores: dict[str, float],   # {node_id: float}
    sem_ranking: list[str],         # node_id 列表，按语义分降序
    sem_rich_scores: dict[str, float] | None,
    sem_rare_scores: dict[str, float] | None,
    sem_formula_weights: dict[str, float] | None,
    w_topo: float,
    w_sem: float,
) -> dict[str, dict]:
    """返回 {node_id: {topo, topo_rank, semantic, semantic_rank, combined, combined_rank}}"""
    all_ids = sorted(set(list(topo_scores.keys()) + list(sem_scores.keys())))

    # 语义排名（从 sem_ranking 构建，允许缺失）
    sem_rank_map: dict[str, int] = {nid: i + 1 for i, nid in enumerate(sem_ranking)}

    result: dict[str, dict] = {}
    for nid in all_ids:
        topo_s = float(topo_scores.get(nid, {}).get("score", 0.0))
        topo_r = int(topo_scores.get(nid, {}).get("rank", 999))
        sem_s  = float(sem_scores.get(nid, 0.0))
        sem_r  = sem_rank_map.get(nid, 999)
        sem_rich = float((sem_rich_scores or {}).get(nid, 0.0))
        sem_rare = float((sem_rare_scores or {}).get(nid, 0.0))
        comb   = w_topo * topo_s + w_sem * sem_s
        result[nid] = {
            "topo": topo_s, "topo_rank": topo_r,
            "semantic": sem_s, "semantic_rank": sem_r,
            "semantic_rich": sem_rich,
            "semantic_rare": sem_rare,
            "semantic_w_rich": float((sem_formula_weights or {}).get("rich", 1.0)),
            "semantic_w_rare": float((sem_formula_weights or {}).get("rare", 0.3)),
            "combined": comb, "combined_rank": 0,   # 后续填充
        }

    # 计算综合排名
    sorted_by_comb = sorted(result.keys(), key=lambda k: result[k]["combined"], reverse=True)
    for rank, nid in enumerate(sorted_by_comb, start=1):
        result[nid]["combined_rank"] = rank

    return result


# ── JS 数据块 ─────────────────────────────────────────────────────────────────

def _js_data_block(
    causal_edges: list[dict],
    temporal_edges: list[dict],
    events: list[dict],
    scores: dict[str, dict],  # combined scores dict
) -> str:
    events_js = json.dumps(
        [{"id": ev["event_id"], "text": ev.get("text", "")} for ev in events],
        ensure_ascii=False,
    )
    causal_js = json.dumps(
        [{"from": e["cause"], "to": e["effect"], "reason": e.get("reason", "")}
         for e in causal_edges],
        ensure_ascii=False,
    )
    temporal_js = json.dumps(
        [{"from": e["source"], "to": e["target"], "reason": e.get("reason", "")}
         for e in temporal_edges],
        ensure_ascii=False,
    )
    scores_js = json.dumps(scores, ensure_ascii=False)
    return (
        f"<script>"
        f"var _pyEvents={events_js};"
        f"var _pyCausal={causal_js};"
        f"var _pyTemporal={temporal_js};"
        f"var _pyScores={scores_js};"
        f"</script>"
    )


# ── JS 逻辑（只读，含模式切换 + 权重动态调节） ────────────────────────────────

_JS_LOGIC = r"""
var _simpActive = false;
var _simpOrig = {};
var _currentWTopo = 0.5;
var _currentCombined = {};
var _currentRanks = {};

function _esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/* 颜色插值：0→#dbeeff，1→#d62728 */
function _scoreToColor(score) {
    var t = Math.max(0, Math.min(1, parseFloat(score) || 0));
    var r = Math.round(219 + t * (214 - 219));
    var g = Math.round(238 + t * (39  - 238));
    var b = Math.round(255 + t * (40  - 255));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
}

/* 详细模式节点标签 */
function _nodeLabelFull(eid, text, combined, rank) {
    var wrap = 14;
    var raw = eid + ': ' + (text || '').replace(/\n/g, ' ');
    var lines = [];
    while (raw.length > wrap) { lines.push(raw.slice(0, wrap)); raw = raw.slice(wrap); }
    if (raw) lines.push(raw);
    lines.push('\u25b2 ' + combined.toFixed(2) + ' (#' + rank + ')');
    return lines.join('\n');
}

/* 根据当前权重初始化 _currentCombined / _currentRanks（页面加载时调用） */
function _initCurrentScores() {
    var scores = _pyScores || {};
    var combined = {};
    Object.keys(scores).forEach(function(nid) {
        var sc = scores[nid];
        combined[nid] = _currentWTopo * sc.topo + (1 - _currentWTopo) * sc.semantic;
    });
    var sorted = Object.keys(combined).sort(function(a, b) { return combined[b] - combined[a]; });
    var ranks = {};
    sorted.forEach(function(nid, i) { ranks[nid] = i + 1; });
    _currentCombined = combined;
    _currentRanks = ranks;
}

/* 重新计算并更新节点颜色/标签/排名表 */
function _recomputeAndUpdate(wTopo) {
    _currentWTopo = wTopo;
    var wSem = 1.0 - wTopo;
    var scores = _pyScores || {};
    var combined = {};
    Object.keys(scores).forEach(function(nid) {
        var sc = scores[nid];
        combined[nid] = wTopo * sc.topo + wSem * sc.semantic;
    });
    var sorted = Object.keys(combined).sort(function(a, b) { return combined[b] - combined[a]; });
    var ranks = {};
    sorted.forEach(function(nid, i) { ranks[nid] = i + 1; });
    _currentCombined = combined;
    _currentRanks = ranks;

    if (typeof network !== 'undefined') {
        var upd = [];
        Object.keys(combined).forEach(function(nid) {
            var sc = scores[nid];
            var comb = combined[nid];
            var rank = ranks[nid];
            var color = _scoreToColor(comb);
            var border = rank === 1 ? '#c0910a' : (rank <= 3 ? '#e07b20' : '#3a7bd5');
            var ev = (_pyEvents || []).find(function(e) { return e.id === nid; });
            var text = ev ? ev.text : '';
            var title = '\u62d3\u6251\u5206: ' + sc.topo.toFixed(3) + ' (#' + sc.topo_rank + ')\n'
                      + '\u8bed\u4e49\u5206: ' + sc.semantic.toFixed(3) + ' (#' + sc.semantic_rank + ')\n'
                      + '  = ' + sc.semantic_w_rich.toFixed(2) + '×丰富性 ' + sc.semantic_rich.toFixed(3) + ' + '
                      + sc.semantic_w_rare.toFixed(2) + '×稀缺性 ' + sc.semantic_rare.toFixed(3) + '\n'
                      + '\u7efc\u5408\u5206: ' + comb.toFixed(3) + ' (#' + rank + ')';
            var label = _simpActive
                ? (nid + ' \u2605' + comb.toFixed(2))
                : _nodeLabelFull(nid, text, comb, rank);
            /* 简洁模式下同步更新 _simpOrig，以便切回详细时使用最新标签 */
            if (_simpActive && _simpOrig[nid]) {
                _simpOrig[nid] = {label: _nodeLabelFull(nid, text, comb, rank), title: title};
            }
            upd.push({
                id: nid, label: label, title: title,
                color: {background: color, border: border,
                        highlight: {background: color, border: '#1a1a1a'},
                        hover:     {background: color, border: '#1a1a1a'}}
            });
        });
        network.body.data.nodes.update(upd);
        network.redraw();
    }
    _updateScorePanel(combined, ranks);
}

/* 刷新右侧排名表 tbody */
function _updateScorePanel(combined, ranks) {
    var tbody = document.getElementById('score-tbody');
    if (!tbody) return;
    var scores = _pyScores || {};
    var sorted = Object.keys(combined).sort(function(a, b) { return combined[b] - combined[a]; });
    var html = '';
    sorted.forEach(function(nid) {
        var sc = scores[nid];
        var comb = combined[nid];
        var rank = ranks[nid];
        var ev = (_pyEvents || []).find(function(e) { return e.id === nid; });
        var text = ev ? ev.text : '';
        var short = text.length > 16 ? text.slice(0, 16) + '\u2026' : text;
        html += '<tr title="' + _esc(text) + '" style="border-bottom:1px solid #f0f0f0;">'
              + '<td style="padding:4px 6px;font-weight:600;color:#333;">' + _esc(nid) + '</td>'
              + '<td style="padding:4px 6px;font-size:11px;color:#555;max-width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + _esc(short) + '</td>'
              + '<td style="padding:4px 6px;text-align:center;">' + sc.topo.toFixed(3) + ' <span style="color:#888;font-size:10px">(#' + sc.topo_rank + ')</span></td>'
              + '<td style="padding:4px 6px;text-align:center;line-height:1.35;">' + sc.semantic.toFixed(3) + ' <span style="color:#888;font-size:10px">(#' + sc.semantic_rank + ')</span><div style="color:#888;font-size:10px">丰 ' + sc.semantic_rich.toFixed(2) + ' / 稀 ' + sc.semantic_rare.toFixed(2) + '</div></td>'
              + '<td style="padding:4px 6px;text-align:center;font-weight:600;color:#c0392b;">' + comb.toFixed(3) + ' <span style="color:#888;font-size:10px">(#' + rank + ')</span></td>'
              + '</tr>';
    });
    tbody.innerHTML = html;
}

/* 权重滑块事件 */
function onWeightChange() {
    var slider = document.getElementById('slider-topo');
    if (!slider) return;
    var wTopo = parseInt(slider.value, 10) / 100;
    var wSem  = 1.0 - wTopo;
    var lblT = document.getElementById('label-topo');
    var lblS = document.getElementById('label-sem');
    if (lblT) lblT.textContent = wTopo.toFixed(2);
    if (lblS) lblS.textContent = wSem.toFixed(2);
    _recomputeAndUpdate(wTopo);
}

function toggleSimplifiedMode() {
    _simpActive = !_simpActive;
    var btn = document.getElementById('btn-mode');
    if (typeof network === 'undefined') return;
    var ids = network.body.data.nodes.getIds();
    var upd = [];
    if (_simpActive) {
        ids.forEach(function(id) {
            var n = network.body.data.nodes.get(id);
            _simpOrig[id] = {label: n.label, title: n.title};
            var comb = _currentCombined[id] !== undefined ? _currentCombined[id]
                     : ((_pyScores[id] || {}).combined || 0);
            upd.push({id: id, label: id + ' \u2605' + comb.toFixed(2)});
        });
        if (btn) btn.textContent = '\u7b80\u6d01\u6a21\u5f0f';
    } else {
        ids.forEach(function(id) {
            var o = _simpOrig[id];
            if (o) upd.push({id: id, label: o.label});
        });
        if (btn) btn.textContent = '\u8be6\u7ec6\u6a21\u5f0f';
    }
    network.body.data.nodes.update(upd);
    network.redraw();
}

window.addEventListener('load', function() { _initCurrentScores(); });
"""


# ── 右侧面板：得分排名表 ───────────────────────────────────────────────────────

def _score_panel_html(scores: dict[str, dict], events: list[dict]) -> str:
    """生成固定在右侧的得分排名表面板 HTML。"""
    # 按综合排名排序
    ranked = sorted(scores.keys(), key=lambda k: scores[k]["combined_rank"])

    # 建立 event_id → text 映射
    ev_map = {ev["event_id"]: ev.get("text", "") for ev in events}

    rows = ""
    for nid in ranked:
        sc = scores[nid]
        text = ev_map.get(nid, "")
        short = text[:16] + "…" if len(text) > 16 else text
        tooltip = _html.escape(text)
        rows += (
            f'<tr title="{tooltip}" style="border-bottom:1px solid #f0f0f0;">'
            f'<td style="padding:4px 6px;font-weight:600;color:#333;">{_html.escape(nid)}</td>'
            f'<td style="padding:4px 6px;font-size:11px;color:#555;max-width:100px;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{_html.escape(short)}</td>'
            f'<td style="padding:4px 6px;text-align:center;">'
            f'{sc["topo"]:.3f} <span style="color:#888;font-size:10px">(#{sc["topo_rank"]})</span></td>'
            f'<td style="padding:4px 6px;text-align:center;">'
            f'{sc["semantic"]:.3f} <span style="color:#888;font-size:10px">(#{sc["semantic_rank"]})</span>'
            f'<div style="color:#888;font-size:10px">丰 {sc.get("semantic_rich", 0.0):.2f} / 稀 {sc.get("semantic_rare", 0.0):.2f}</div></td>'
            f'<td style="padding:4px 6px;text-align:center;font-weight:600;color:#c0392b;">'
            f'{sc["combined"]:.3f} <span style="color:#888;font-size:10px">(#{sc["combined_rank"]})</span></td>'
            f'</tr>'
        )

    return (
        '<div style="position:fixed;top:14px;right:14px;bottom:14px;width:360px;'
        'z-index:9999;display:flex;flex-direction:column;">'
        '<div style="background:#ffffffee;border:1px solid #c0c0c0;border-radius:8px;'
        'padding:10px 14px;font-family:sans-serif;font-size:13px;line-height:1.5;'
        'box-shadow:0 2px 6px rgba(0,0,0,0.10);overflow-y:auto;flex:1;">'
        '<div style="font-weight:700;font-size:13px;color:#333;border-bottom:1px solid #e0e0e0;'
        'padding-bottom:6px;margin-bottom:8px;">\u5173\u952e\u8282\u70b9\u5f97\u5206\u6392\u540d</div>'
        '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
        '<thead>'
        '<tr style="background:#f5f8ff;font-size:11px;color:#555;">'
        '<th style="padding:4px 6px;text-align:left;">节点</th>'
        '<th style="padding:4px 6px;text-align:left;">描述</th>'
        '<th style="padding:4px 6px;text-align:center;">拓扑</th>'
        '<th style="padding:4px 6px;text-align:center;">语义</th>'
        '<th style="padding:4px 6px;text-align:center;">综合</th>'
        '</tr>'
        '</thead>'
        '<tbody id="score-tbody">'
        + rows +
        '</tbody>'
        '</table>'
        '</div>'
        '</div>'
    )


# ── 左上信息框 ────────────────────────────────────────────────────────────────

def _info_box_html(
    doc_title: str,
    causal_count: int,
    temporal_count: int,
    sem_formula_weights: dict[str, float] | None = None,
    relation_weights: dict[str, float] | None = None,
) -> str:
    sem_formula_weights = sem_formula_weights or {"rich": 1.0, "rare": 0.3}
    relation_html = ""
    if relation_weights:
        relation_html = (
            '<div style="margin-top:6px;font-size:11px;color:#666;line-height:1.5;">'
            '<div><b>关系权重</b></div>'
            f'<div>T+ {relation_weights.get("temporal_forward", 0.0):.2f} / '
            f'T- {relation_weights.get("temporal_reverse", 0.0):.2f}</div>'
            f'<div>C+ {relation_weights.get("causal_forward", 0.0):.2f} / '
            f'C- {relation_weights.get("causal_reverse", 0.0):.2f}</div>'
            '</div>'
        )
    return (
        '<div style="position:fixed;top:14px;left:14px;z-index:9999;'
        'background:#ffffffcc;border:1px solid #c0c0c0;border-radius:8px;'
        'padding:10px 14px;font-family:sans-serif;font-size:13px;line-height:1.5;'
        'box-shadow:0 2px 6px rgba(0,0,0,0.08);max-width:320px;">'
        f'<div style="font-weight:600;font-size:14px;margin-bottom:4px;">{_html.escape(doc_title)}</div>'
        f'<div>\u56e0\u679c\u8fb9\uff1a<b>{causal_count}</b> \u6761'
        f'&emsp;\u65f6\u5e8f\u8fb9\uff1a<b>{temporal_count}</b> \u6761</div>'
        '<div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;">'
        '<button id="btn-mode" onclick="toggleSimplifiedMode()" '
        'style="cursor:pointer;padding:3px 10px;font-size:12px;border:1px solid #aaa;'
        'border-radius:4px;background:#f5f5f5;">\u8be6\u7ec6\u6a21\u5f0f</button>'
        '</div>'
        '<div style="margin-top:8px;font-size:11px;color:#888;">'
        '<span style="display:inline-block;width:14px;height:3px;background:#d62728;'
        'vertical-align:middle;margin-right:4px;border-radius:1px;"></span>\u56e0\u679c\u8fb9'
        '&emsp;'
        '<span style="display:inline-block;width:14px;height:0;border-top:2px dashed #1f77b4;'
        'vertical-align:middle;margin-right:4px;"></span>\u65f6\u5e8f\u8fb9'
        f'<div style="margin-top:6px;font-size:11px;color:#666;line-height:1.5;">'
        f'<div><b>语义公式</b></div>'
        f'<div>语义分 = {sem_formula_weights.get("rich", 1.0):.2f} × 丰富性 + {sem_formula_weights.get("rare", 0.3):.2f} × 稀缺性</div>'
        f'</div>'
        f'{relation_html}'
        '</div>'
        '</div>'
    )


# ── 权重调节面板 ─────────────────────────────────────────────────────────────

def _weight_panel_html() -> str:
    """底部居中的拓扑/语义权重滑块面板。"""
    return (
        '<div id="weight-panel" style="position:fixed;bottom:14px;left:50%;'
        'transform:translateX(-50%);z-index:9999;'
        'background:#ffffffee;border:1px solid #c0c0c0;border-radius:8px;'
        'padding:10px 18px;font-family:sans-serif;font-size:13px;'
        'box-shadow:0 2px 6px rgba(0,0,0,0.10);min-width:360px;'
        'display:flex;flex-direction:column;gap:6px;">'
        '<div style="font-weight:700;font-size:12px;color:#555;margin-bottom:2px;">'
        '\u6743\u91cd\u8c03\u8282\uff08\u62d3\u6251 + \u8bed\u4e49 = 1\uff09</div>'
        '<div style="display:flex;align-items:center;gap:10px;">'
        '<span style="font-size:12px;color:#333;white-space:nowrap;min-width:56px;">'
        '\u62d3\u6251 <b id="label-topo">0.50</b></span>'
        '<input type="range" id="slider-topo" min="0" max="100" value="50" step="1"'
        ' style="flex:1;cursor:pointer;accent-color:#d62728;" oninput="onWeightChange()">'
        '<span style="font-size:12px;color:#333;white-space:nowrap;min-width:56px;text-align:right;">'
        '<b id="label-sem">0.50</b> \u8bed\u4e49</span>'
        '</div>'
        '</div>'
    )


# ── 颜色图例 ──────────────────────────────────────────────────────────────────

def _legend_html() -> str:
    stops = " ".join(
        f'<div style="width:16px;height:16px;background:{_score_to_color(i/9)};'
        f'border-radius:2px;" title="{i/9:.1f}"></div>'
        for i in range(10)
    )
    return (
        '<div style="position:fixed;bottom:14px;left:14px;z-index:9999;'
        'background:#ffffffcc;border:1px solid #c0c0c0;border-radius:8px;'
        'padding:8px 12px;font-family:sans-serif;font-size:11px;color:#555;'
        'box-shadow:0 2px 6px rgba(0,0,0,0.08);">'
        '<div style="margin-bottom:4px;font-weight:600;">\u8282\u70b9\u5f97\u5206\uff08\u7efc\u5408\uff09</div>'
        '<div style="display:flex;align-items:center;gap:2px;">'
        '<span style="margin-right:4px;">0.0</span>'
        + stops +
        '<span style="margin-left:4px;">1.0</span>'
        '</div>'
        '</div>'
    )


# ── pyvis 渲染核心 ────────────────────────────────────────────────────────────

def _render_html(
    graph: dict[str, Any],
    scores: dict[str, dict],
    output_path: Path,
    sem_formula_weights: dict[str, float] | None = None,
    relation_weights: dict[str, float] | None = None,
) -> None:
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise RuntimeError("需要安装 pyvis: pip install pyvis") from exc

    import re as _re

    events = sorted(graph.get("events", []), key=lambda ev: _event_id_num(ev["event_id"]))
    causal_edges = graph.get("causal_edges", [])
    temporal_edges = graph.get("temporal_edges", [])

    net = Network(height="100vh", width="100%", directed=True, notebook=False)
    net.set_options(
        """
        {
          "physics": {
            "enabled": true,
            "stabilization": { "iterations": 200, "fit": true },
            "barnesHut": { "gravitationalConstant": -14000, "springLength": 220,
                           "springConstant": 0.006, "damping": 0.92 }
          },
          "edges": { "smooth": { "type": "cubicBezier" } },
          "interaction": { "hover": true, "tooltipDelay": 120, "dragNodes": true },
          "nodes": {
            "shape": "box",
            "borderWidth": 2,
            "font": { "size": 12, "color": "#222", "multi": false },
            "margin": 8
          }
        }
        """
    )

    for event in events:
        eid = event["event_id"]
        text = event.get("text", "")
        sc = scores.get(eid, {})
        combined = sc.get("combined", 0.0)
        combined_rank = sc.get("combined_rank", 999)
        topo = sc.get("topo", 0.0)
        topo_rank = sc.get("topo_rank", 999)
        semantic = sc.get("semantic", 0.0)
        semantic_rank = sc.get("semantic_rank", 999)
        semantic_rich = sc.get("semantic_rich", 0.0)
        semantic_rare = sc.get("semantic_rare", 0.0)
        semantic_w_rich = sc.get("semantic_w_rich", 1.0)
        semantic_w_rare = sc.get("semantic_w_rare", 0.3)

        color = _score_to_color(combined)
        # 节点边框：综合排名第1用金色，前3用深橙
        if combined_rank == 1:
            border = "#c0910a"
        elif combined_rank <= 3:
            border = "#e07b20"
        else:
            border = "#3a7bd5"

        label = _node_label(eid, text, combined, combined_rank)
        title = (
            f"\u62d3\u6251\u5206: {topo:.3f} (#{topo_rank})\n"
            f"\u8bed\u4e49\u5206: {semantic:.3f} (#{semantic_rank})\n"
            f"  = {semantic_w_rich:.2f}×丰富性 {semantic_rich:.3f} + {semantic_w_rare:.2f}×稀缺性 {semantic_rare:.3f}\n"
            f"\u7efc\u5408\u5206: {combined:.3f} (#{combined_rank})"
        )

        net.add_node(
            eid,
            label=label,
            shape="box",
            color=cast(Any, {
                "background": color,
                "border": border,
                "highlight": {"background": color, "border": "#1a1a1a"},
                "hover": {"background": color, "border": "#1a1a1a"},
            }),
            title=title,
        )

    for i, edge in enumerate(causal_edges):
        net.add_edge(
            edge["cause"], edge["effect"],
            id=f"c{i}",
            color=CAUSAL_COLOR, width=3,
            title=_html.escape(edge.get("reason", "")),
            arrows="to",
        )
    for i, edge in enumerate(temporal_edges):
        net.add_edge(
            edge["source"], edge["target"],
            id=f"t{i}",
            color=TEMPORAL_COLOR, width=1.6, dashes=True,
            title=_html.escape(edge.get("reason", "")),
            arrows="to",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = net.generate_html(notebook=False)

    # 清理不需要的元素
    html = _re.sub(r"<hr[^>]*>", "", html, flags=_re.IGNORECASE)
    html = html.replace(
        "</head>",
        '<style>'
        'html,body{margin:0;padding:0;overflow:hidden}'
        '#config,div.vis-configuration-wrapper{display:none!important}'
        '#mynetwork{border:none!important;height:100vh!important}'
        'hr{display:none!important}'
        '</style></head>',
        1,
    )

    # 注入数据 + 逻辑 + UI
    data_js = _js_data_block(causal_edges, temporal_edges, events, scores)
    logic_js = f"<script>{_JS_LOGIC}</script>"
    doc_title = _strip_doc_suffix(graph.get("doc_name") or graph.get("doc_id", ""))
    info_html = _info_box_html(
        doc_title,
        len(causal_edges),
        len(temporal_edges),
        sem_formula_weights,
        relation_weights,
    )
    score_panel = _score_panel_html(scores, events)
    legend = _legend_html()

    insert = data_js + logic_js + info_html + score_panel + legend + _weight_panel_html()
    if "</body>" in html:
        html = html.replace("</body>", insert + "</body>", 1)
    else:
        html += insert

    output_path.write_text(html, encoding="utf-8")


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _event_id_num(event_id: str) -> int:
    """从 'E3' 提取数字 3，用于排序。"""
    import re as _re
    m = _re.search(r"\d+", event_id)
    return int(m.group()) if m else 0


def _strip_doc_suffix(name: str) -> str:
    if name.lower().endswith(".txt"):
        return name[:-4]
    return name


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default_semantic_score_root() -> Path:
    """优先使用 semantic_model 的正式输出；不存在时退回 key_nodes/data。"""
    project_root = Path(__file__).resolve().parent.parent
    model_root = project_root / "semantic_model" / "output" / "semantic_scores"
    if model_root.exists():
        return model_root
    return Path(__file__).parent / "data" / "semantic_scores"


# ── 公开 API ──────────────────────────────────────────────────────────────────

def render_doc(
    graph_path: Path,
    topo_scores: dict,    # {node_id: {"score": float, "rank": int}}
    sem_scores: dict,     # {node_id: float}
    sem_ranking: list,    # node_id 列表，按语义分降序
    sem_rich_scores: dict[str, float] | None,
    sem_rare_scores: dict[str, float] | None,
    sem_formula_weights: dict[str, float] | None,
    relation_weights: dict[str, float] | None,
    output_path: Path,
    w_topo: float = 0.5,
    w_sem: float = 0.5,
) -> None:
    """渲染单个文档的关键节点可视化 HTML。"""
    graph = _read_json(graph_path)
    scores = _compute_combined(
        topo_scores,
        sem_scores,
        sem_ranking,
        sem_rich_scores,
        sem_rare_scores,
        sem_formula_weights,
        w_topo,
        w_sem,
    )
    _render_html(graph, scores, output_path, sem_formula_weights, relation_weights)
    print(f"  [render] {output_path.relative_to(output_path.parents[3])}")


def render_all(
    graph_root: Path,
    topo_root: Path,
    sem_root: Path,
    output_root: Path,
    w_topo: float = 0.5,
    w_sem: float = 0.5,
) -> None:
    """遍历所有文档，批量渲染。

    目录约定：
      graph_root/{doc_id}/graph.json
      topo_root/{doc_id}/topology_scores.json   → {"doc_id":..., "scores":{...}}
      sem_root/{doc_id}/semantic_scores.json     → {"doc_id":..., "scores":{...}, "ranking":[...]}
      output_root/{doc_id}_scores.html
    """
    doc_ids = sorted(p.name for p in graph_root.iterdir() if p.is_dir())
    total = len(doc_ids)
    print(f"[render_all] 共 {total} 个文档")

    ok, skipped = 0, 0
    for doc_id in doc_ids:
        graph_path = graph_root / doc_id / "graph.json"
        topo_path  = topo_root  / doc_id / "topology_scores.json"
        sem_path   = sem_root   / doc_id / "semantic_scores.json"

        missing = [p for p in (graph_path, topo_path, sem_path) if not p.exists()]
        if missing:
            print(f"  [skip] {doc_id}：缺少 {[str(m) for m in missing]}")
            skipped += 1
            continue

        topo_data = _read_json(topo_path)
        sem_data  = _read_json(sem_path)
        topo_scores = topo_data.get("scores", {})
        sem_scores  = sem_data.get("scores", {})
        sem_ranking = sem_data.get("ranking", [])
        sem_rich_scores = sem_data.get("rich_scores", {})
        sem_rare_scores = sem_data.get("rare_scores", {})
        sem_formula_weights = sem_data.get("formula_weights", {})
        relation_weights = sem_data.get("relation_weights")

        out = output_root / f"{doc_id}_scores.html"
        try:
            render_doc(
                graph_path,
                topo_scores,
                sem_scores,
                sem_ranking,
                sem_rich_scores,
                sem_rare_scores,
                sem_formula_weights,
                relation_weights,
                out,
                w_topo,
                w_sem,
            )
            ok += 1
        except Exception as exc:
            print(f"  [error] {doc_id}: {exc}")
            skipped += 1

    print(f"[render_all] 完成 {ok}/{total}，跳过 {skipped}")


__all__ = ["render_doc", "render_all"]


# ── 快速测试入口 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _base = Path(__file__).parent / "data"
    render_all(
        graph_root=_base / "output" / "doc_results",
        topo_root =_base / "topology_scores",
        sem_root  =_default_semantic_score_root(),
        output_root=Path(__file__).parent / "output",
    )
