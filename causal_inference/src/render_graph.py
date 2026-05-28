"""读 graph.json 渲染为交互式 HTML 。

设计原则：
- 本文件不依赖嵌入模型、不调 LLM，只负责画图。
- 节点默认显示"E1: 事件描述"；鼠标 hover 边显示原因。
- 因果：红色实线；时序：蓝色虚线；人工审核修改：黄色。
- 左上角信息框显示文档标题、边数量、当前模式及操作按钮。
- 右侧事件列表在两种模式下均可见，支持展开/折叠全文。
- 编辑模式下点击边可删除或修改（类型转换），修改结果可保存回 graph.json。
    生成 HTML 时会尽量自动拉起本地保存服务（端口 7760）；
    若服务暂未就绪，前端保存时会自动等待并重试，仍失败才降级为下载 graph.json。
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any

try:
    from .infra import event_id_num, read_json_utf8, write_text_utf8
except ImportError:  # standalone mode (called directly as a script)
    from infra import event_id_num, read_json_utf8, write_text_utf8  # type: ignore[no-redef]

CAUSAL_COLOR = "#d62728"
TEMPORAL_COLOR = "#1f77b4"
MANUAL_REVIEW_COLOR = "#f5a623"
NODE_BG_COLOR = "#dbeeff"         # 方框填充色
NODE_BORDER_COLOR = "#3a7bd5"     # 方框边框色
NODE_FONT_SIZE = 12
NODE_WRAP_CHARS = 14              # 每行字符数（中文字符较宽，取小值）
_SAVE_PORT = 7760                 # ManualReviewServer 默认端口


def _strip_doc_suffix(doc_name: str) -> str:
    if doc_name.lower().endswith(".txt"):
        return doc_name[:-4]
    return doc_name


def _node_label(event_id: str, text: str, wrap: int = NODE_WRAP_CHARS) -> str:
    """生成方框节点标签，按 wrap 个字符换行（兼容中文无空格场景）。"""
    raw = f"{event_id}: {text.strip().replace(chr(10), ' ')}"
    if len(raw) <= wrap:
        return raw
    lines: list[str] = []
    while len(raw) > wrap:
        lines.append(raw[:wrap])
        raw = raw[wrap:]
    if raw:
        lines.append(raw)
    return "\n".join(lines)


def _edge_title(reason: str, manual_review: bool) -> str:
    text = (reason or "").strip()
    if manual_review:
        return f"{text} [人工审核]" if text else "[人工审核]"
    return text


_JS_LOGIC = r"""
var _edgeLookup = {};      // edgeId -> {type, from, to, reason, manualReview, isNew}
var _deletedEdges = {};    // edgeId -> true
var _manualEdits = {};     // edgeId -> {originalType, newType, reason}
var _simpActive = false;
var _simpOrig = {};
var _editMode = false;
var _evtMode = 'list';     // 'list' | 'doc'
var _addEdge = null;       // null | {role: 'source'|'target', anchorId: string}
var _newEdgePair = null;   // null | {source: string, target: string}
var _hasUnsaved = false;
var _pyManualReviewEdges = _pyManualReviewEdges || [];

function _esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function _edgeTitle(reason, manualReview) {
    var title = _esc(reason || '');
    if (manualReview) return title ? (title + ' [人工审核]') : '[人工审核]';
    return title;
}
function _edgeColor(edgeType, manualReview) {
    if (manualReview) return '#f5a623';
    return edgeType === 'causal' ? '#d62728' : '#1f77b4';
}
function _edgeWidth(edgeType) {
    return edgeType === 'causal' ? 3 : 1.6;
}
function _edgeDashes(edgeType, manualReview) {
    if (edgeType !== 'temporal') return false;
    return manualReview ? [10, 6] : true;
}
function _edgeStyle(edgeType, manualReview) {
    var color = _edgeColor(edgeType, manualReview);
    return {
        color: {color: color, hover: color, highlight: color},
        width: _edgeWidth(edgeType),
        dashes: _edgeDashes(edgeType, manualReview),
    };
}
function _wait(ms) {
    return new Promise(function(resolve) { setTimeout(resolve, ms); });
}

/* ── 图遍历辅助 ─────────────────────────────────────────────── */
function _liveEdges() {
    return network.body.data.edges.get().filter(function(e) { return !_deletedEdges[e.id]; });
}
function getDescendants(nodeId) {
    var visited = {}, queue = [String(nodeId)];
    while (queue.length) {
        var cur = queue.shift();
        if (visited[cur]) continue;
        visited[cur] = true;
        _liveEdges().forEach(function(e) {
            var f = String(e.from), t = String(e.to);
            if (f === cur && !visited[t]) queue.push(t);
        });
    }
    return visited;
}
function getAncestors(nodeId) {
    var visited = {}, queue = [String(nodeId)];
    while (queue.length) {
        var cur = queue.shift();
        if (visited[cur]) continue;
        visited[cur] = true;
        _liveEdges().forEach(function(e) {
            var f = String(e.from), t = String(e.to);
            if (t === cur && !visited[f]) queue.push(f);
        });
    }
    return visited;
}
function hasEdge(fromId, toId) {
    var f = String(fromId), t = String(toId);
    return _liveEdges().some(function(e) { return String(e.from) === f && String(e.to) === t; });
}

/* ── 模式切换 ──────────────────────────────────────────────── */
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
            var ev = (_pyEvents || []).find(function(e) { return e.id === id; });
            upd.push({id: id, label: String(id), title: ev ? (_esc(id) + ': ' + _esc(ev.text)) : ''});
        });
        if (btn) btn.textContent = '简洁模式';
    } else {
        ids.forEach(function(id) {
            var o = _simpOrig[id];
            if (o) upd.push({id: id, label: o.label, title: o.title});
        });
        if (btn) btn.textContent = '详细模式';
    }
    network.body.data.nodes.update(upd);
    network.redraw();
}

/* ── 编辑模式 ──────────────────────────────────────────────── */
function toggleEditMode() {
    var btn = document.getElementById('btn-edit');
    if (_editMode) {
        if (_hasUnsaved && !confirm('有未保存的修改，退出修改模式后将丢失，确定退出？')) return;
        _editMode = false;
        btn.textContent = '修改';
        btn.style.background = '#f5f5f5';
        btn.style.borderColor = '#aaa';
        closeAllMenus();
    } else {
        _editMode = true;
        btn.textContent = '退出修改';
        btn.style.background = '#fff3cd';
        btn.style.borderColor = '#f5a623';
    }
}
function closeAllMenus() {
    closeEdgeMenu();
    closeNodeMenu();
    closeCandidatePanel();
}

/* ── 边右键菜单 ─────────────────────────────────────────────── */
function showEdgeMenu(edgeId, x, y) {
    var menu = document.getElementById('edge-ctx-menu');
    menu.setAttribute('data-eid', edgeId);
    var e = _edgeLookup[edgeId] || {};
    var typeLabel = e.type === 'causal' ? '因果边' : '时序边';
    document.getElementById('ctx-type-label').textContent = '当前: ' + typeLabel;
    // 预计算「修改方向」是否可用（会产生环或反向边已存在则禁用）
    var reverseBtn = document.getElementById('ctx-reverse-btn');
    if (reverseBtn && e.from && e.to) {
        var fId = String(e.from), tId = String(e.to);
        _deletedEdges[edgeId] = true;
        var desc = getDescendants(tId);
        delete _deletedEdges[edgeId];
        var disabled = !!(desc[fId] || hasEdge(tId, fId));
        reverseBtn.style.color = disabled ? '#bbb' : '';
        reverseBtn.style.cursor = disabled ? 'not-allowed' : 'pointer';
        reverseBtn.setAttribute('data-disabled', disabled ? '1' : '0');
    }
    menu.style.left = (x + 6) + 'px';
    menu.style.top  = (y + 6) + 'px';
    menu.style.display = 'block';
}
function closeEdgeMenu() {
    var menu = document.getElementById('edge-ctx-menu');
    if (menu) menu.style.display = 'none';
}
function menuDelete() {
    var edgeId = document.getElementById('edge-ctx-menu').getAttribute('data-eid');
    if (!edgeId) return;
    network.body.data.edges.remove(edgeId);
    _deletedEdges[edgeId] = true;
    closeEdgeMenu();
    _setUnsaved();
}
function menuModify() {
    var edgeId = document.getElementById('edge-ctx-menu').getAttribute('data-eid');
    if (!edgeId) return;
    closeEdgeMenu();
    var e = _edgeLookup[edgeId] || {};
    var other = e.type === 'causal' ? '时序边' : '因果边';
    document.getElementById('dlg-title').textContent = '修改为' + other;
    document.getElementById('dlg-reason').value = e.reason || '';
    var dlg = document.getElementById('modify-dlg');
    dlg.setAttribute('data-eid', edgeId);
    dlg.style.display = 'flex';
    document.getElementById('dlg-reason').focus();
}

function menuReverse() {
    var reverseBtn = document.getElementById('ctx-reverse-btn');
    if (reverseBtn && reverseBtn.getAttribute('data-disabled') === '1') return;
    var edgeId = document.getElementById('edge-ctx-menu').getAttribute('data-eid');
    if (!edgeId) return;
    closeEdgeMenu();
    var e = _edgeLookup[edgeId];
    if (!e) return;
    var fromId = String(e.from), toId = String(e.to);
    // 再次验证（防止菜单开启期间图结构变化）
    _deletedEdges[edgeId] = true;
    var desc = getDescendants(toId);
    delete _deletedEdges[edgeId];
    if (desc[fromId]) { showToast('修改方向会产生环路，无法执行', 2500, '#c0392b'); return; }
    if (hasEdge(toId, fromId)) { showToast('反向边已存在', 2500, '#c0392b'); return; }
    // 删除原边，添加反向边
    network.body.data.edges.remove(edgeId);
    _deletedEdges[edgeId] = true;
    var isCausal = (e.type === 'causal');
    var newEid = 'rev_' + Date.now();
    var style = _edgeStyle(e.type, true);
    network.body.data.edges.add([{
        id: newEid, from: toId, to: fromId, arrows: 'to',
        color: style.color,
        width: style.width, dashes: style.dashes,
        title: _esc(e.reason || '') + ' [方向已反转]',
    }]);
    _edgeLookup[newEid] = {type: e.type, from: toId, to: fromId, reason: e.reason, manualReview: true, isNew: true};
    showToast('已反转方向: ' + toId + ' → ' + fromId, 2200, '#27ae60');
    _setUnsaved();
}
/* ── 节点点击菜单 ───────────────────────────────────────────── */
function showNodeMenu(nodeId, x, y) {
    var menu = document.getElementById('node-ctx-menu');
    menu.setAttribute('data-nid', String(nodeId));
    menu.style.left = (x + 6) + 'px';
    menu.style.top  = (y + 6) + 'px';
    menu.style.display = 'block';
}
function closeNodeMenu() {
    var menu = document.getElementById('node-ctx-menu');
    if (menu) menu.style.display = 'none';
}
function startAddEdge(role) {
    var nodeId = document.getElementById('node-ctx-menu').getAttribute('data-nid');
    closeNodeMenu();
    _addEdge = {role: role, anchorId: nodeId};
    _showCandidatePanel(nodeId, role);
}

/* ── 候选节点面板 ───────────────────────────────────────────── */
function _showCandidatePanel(anchorId, role) {
    var allIds = network.body.data.nodes.getIds().map(String);
    var excluded = {};
    excluded[String(anchorId)] = true;
    if (role === 'source') {
        // anchor 是起点，选终点 B：B 不能是 anchor 的祖先（防环）且不能有 anchor→B（防重复）
        var anc = getAncestors(anchorId);
        Object.keys(anc).forEach(function(k) { excluded[k] = true; });
        _liveEdges().forEach(function(e) {
            if (String(e.from) === String(anchorId)) excluded[String(e.to)] = true;
        });
    } else {
        // anchor 是终点，选起点 A：A 不能是 anchor 的后代（防环）且不能有 A→anchor（防重复）
        var desc = getDescendants(anchorId);
        Object.keys(desc).forEach(function(k) { excluded[k] = true; });
        _liveEdges().forEach(function(e) {
            if (String(e.to) === String(anchorId)) excluded[String(e.from)] = true;
        });
    }
    var candidates = allIds.filter(function(id) { return !excluded[id]; });
    if (candidates.length === 0) {
        showToast('没有可新增的候选节点', 2000, '#555');
        _addEdge = null;
        return;
    }
    var title = role === 'source' ? '以该节点为起点，选择终点：' : '以该节点为终点，选择起点：';
    document.getElementById('candidate-title').textContent = title;
    var html = '';
    candidates.forEach(function(id) {
        var ev = (_pyEvents || []).find(function(e) { return e.id === id; });
        var label = ev ? (id + ': ' + ev.text) : id;
        html += '<div onclick="selectCandidate(\'' + id + '\')" '
              + 'style="padding:6px 10px;cursor:pointer;border-radius:4px;margin-bottom:3px;'
              + 'font-size:12px;background:#f0f7ff;border:1px solid #c5d8f0;" '
              + 'onmouseover="this.style.background=\'#dbeeff\'" '
              + 'onmouseout="this.style.background=\'#f0f7ff\'">'
              + _esc(label) + '</div>';
    });
    document.getElementById('candidate-body').innerHTML = html;
    document.getElementById('candidate-panel').style.display = 'flex';
}
function closeCandidatePanel() {
    var p = document.getElementById('candidate-panel');
    if (p) p.style.display = 'none';
    _addEdge = null;
}
function selectCandidate(candidateId) {
    if (!_addEdge) { closeCandidatePanel(); return; }
    var src = _addEdge.role === 'source' ? _addEdge.anchorId : candidateId;
    var tgt = _addEdge.role === 'target' ? _addEdge.anchorId : candidateId;
    closeCandidatePanel();
    _newEdgePair = {source: src, target: tgt};
    _showNewEdgeDlg(src, tgt);
}

/* ── 新建边对话框 ───────────────────────────────────────────── */
function _showNewEdgeDlg(src, tgt) {
    document.getElementById('new-edge-subtitle').textContent = src + ' → ' + tgt;
    document.getElementById('new-edge-reason').value = '';
    document.getElementById('new-edge-type').value = 'causal';
    document.getElementById('new-edge-dlg').style.display = 'flex';
    document.getElementById('new-edge-reason').focus();
}
function closeNewEdgeDlg() {
    document.getElementById('new-edge-dlg').style.display = 'none';
    _newEdgePair = null;
}
function applyNewEdge() {
    if (!_newEdgePair) { closeNewEdgeDlg(); return; }
    var reason = document.getElementById('new-edge-reason').value.trim();
    if (!reason) { alert('请填写原因。'); return; }
    var edgeType = document.getElementById('new-edge-type').value;
    var src = _newEdgePair.source, tgt = _newEdgePair.target;
    var eid = 'new_' + Date.now();
    var style = _edgeStyle(edgeType, true);
    network.body.data.edges.add([{
        id: eid, from: src, to: tgt, arrows: 'to',
        color: style.color,
        width: style.width, dashes: style.dashes,
        title: _esc(reason) + ' [人工新增]',
    }]);
    _edgeLookup[eid] = {type: edgeType, from: src, to: tgt, reason: reason, manualReview: true, isNew: true};
    closeNewEdgeDlg();
    showToast('已新增边 ' + src + ' → ' + tgt, 2000, '#27ae60');
    _setUnsaved();
}

/* ── 修改对话框 ────────────────────────────────────────────── */
function closeModifyDlg() {
    document.getElementById('modify-dlg').style.display = 'none';
}
function applyModify() {
    var dlg = document.getElementById('modify-dlg');
    var edgeId = dlg.getAttribute('data-eid');
    var reason = document.getElementById('dlg-reason').value.trim();
    if (!reason) { alert('请填写修改原因。'); return; }
    var e = _edgeLookup[edgeId];
    if (!e) { closeModifyDlg(); return; }
    var newType = e.type === 'causal' ? 'temporal' : 'causal';
    var style = _edgeStyle(newType, true);
    network.body.data.edges.update([{
        id: edgeId,
        color: style.color,
        dashes: style.dashes, width: style.width,
        title: _esc(reason) + ' [人工审核]',
    }]);
    _manualEdits[edgeId] = {originalType: e.type, newType: newType, reason: reason};
    e.reason = reason; e.type = newType; e.manualReview = true;
    closeModifyDlg();
    _setUnsaved();
}

/* ── 保存/下载 ─────────────────────────────────────────────── */
function _setUnsaved() {
    _hasUnsaved = true;
    var el = document.getElementById('save-status');
    if (el) { el.textContent = '有未保存的修改'; el.style.opacity = '1'; }
}
function _updateGraphInfo(data) {
    var causalCount = document.getElementById('causal-count');
    var temporalCount = document.getElementById('temporal-count');
    var titleEl = document.getElementById('doc-title');
    if (causalCount) causalCount.textContent = String((data.causal_edges || []).length);
    if (temporalCount) temporalCount.textContent = String((data.temporal_edges || []).length);
    if (titleEl && data.doc_name) {
        titleEl.textContent = String(data.doc_name).replace(/\.txt$/i, '');
    }
}
function _loadLatestGraph() {
    if (!_graphJsonPath) return Promise.resolve(false);
    var url = 'http://localhost:' + _savePort + '/load_graph?path=' + encodeURIComponent(_graphJsonPath) + '&_ts=' + Date.now();
    return fetch(url, {cache: 'no-store'}).then(function(r) {
        var ok = r.ok;
        return r.json().then(function(j) { return {ok: ok, body: j}; });
    }).then(function(res) {
        if (!res.ok || !res.body || !res.body.data) return false;
        _applyLoadedData(res.body.data);
        return true;
    }).catch(function() {
        return false;
    });
}
function _saveRequest(data) {
    return fetch('http://localhost:' + _savePort + '/save_graph', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({path: _graphJsonPath, data: data}),
    }).then(function(r) {
        var ok = r.ok;
        return r.json().then(function(j) { return {ok: ok, body: j}; });
    });
}
function _waitForSaveServer(maxAttempts) {
    var attempts = 0;
    function probe() {
        attempts += 1;
        return _loadLatestGraph().then(function(ok) {
            if (ok || attempts >= maxAttempts) return ok;
            return _wait(500).then(probe);
        }).catch(function() {
            if (attempts >= maxAttempts) return false;
            return _wait(500).then(probe);
        });
    }
    return probe();
}
function _saveFallback(data, msg) {
    showToast(msg, 6000, '#c0392b');
    var el = document.getElementById('save-status');
    if (el) { el.textContent = '保存失败，修改尚未写入 graph.json'; el.style.opacity = '1'; }
}
function _handleSaveResponse(res, data) {
    if (res.ok && (!res.body || res.body.ok !== false)) {
        showToast('已保存并重载 ✓', 3000, '#27ae60');
        var el = document.getElementById('save-status');
        if (el) { el.textContent = ''; el.style.opacity = '0'; }
        _applyLoadedData(res.body.data || data);
        return true;
    }
    var err = (res.body && res.body.error) ? ('保存失败：' + res.body.error) : '保存失败';
    _saveFallback(data, err);
    return false;
}
function buildSaveData() {
    var causal = [], temporal = [];
    for (var eid in _edgeLookup) {
        if (_deletedEdges[eid]) continue;
        var e = _edgeLookup[eid];
        if (e.type === 'causal') {
            var d = {cause: e.from, effect: e.to, reason: e.reason};
            if (e.manualReview) d.manual_review = true;
            causal.push(d);
        } else {
            var d = {source: e.from, target: e.to, reason: e.reason};
            if (e.manualReview) d.manual_review = true;
            temporal.push(d);
        }
    }
    var evts = (_pyEvents || []).map(function(e) { return {event_id: e.id, text: e.text}; });
    return {
        doc_id: (_origGraphMeta || {}).doc_id || '',
        doc_name: (_origGraphMeta || {}).doc_name || '',
        doc_text: _pyDocText || '',
        events: evts, causal_edges: causal, temporal_edges: temporal,
        manual_review_edges: _pyManualReviewEdges || [],
    };
}
function saveGraph() {
    var data = buildSaveData();
    showToast('保存中…', 30000, '#555');
    _saveRequest(data).then(function(res) {
        _handleSaveResponse(res, data);
    }).catch(function() {
        showToast('本地保存服务暂未就绪，正在自动重试…', 4000, '#e67e22');
        _waitForSaveServer(6).then(function(ok) {
            if (!ok) {
                _saveFallback(data, '自动保存服务仍未就绪，未写入 graph.json');
                return;
            }
            showToast('本地保存服务已就绪，正在重试保存…', 4000, '#e67e22');
            _saveRequest(data).then(function(res) {
                _handleSaveResponse(res, data);
            }).catch(function() {
                _saveFallback(data, '自动重试保存失败，未写入 graph.json');
            });
        });
    });
}
function _applyLoadedData(data) {
    /* 用服务器返回的 graph.json 内容就地重建可视化，边恢复红/蓝色，状态全清零 */
    if (!data || typeof network === 'undefined') return;
    network.body.data.edges.clear();
    _edgeLookup = {};
    _deletedEdges = {};
    _manualEdits = {};
    _hasUnsaved = false;
    _pyCausal = (data.causal_edges || []).map(function(e) {
        return {from: e.cause, to: e.effect, reason: e.reason || '', manual_review: !!e.manual_review};
    });
    _pyTemporal = (data.temporal_edges || []).map(function(e) {
        return {from: e.source, to: e.target, reason: e.reason || '', manual_review: !!e.manual_review};
    });
    if (Array.isArray(data.events)) {
        _pyEvents = data.events.map(function(ev) { return {id: ev.event_id, text: ev.text || ''}; });
    }
    if (typeof data.doc_text === 'string') {
        _pyDocText = data.doc_text;
    }
    if (data.doc_id || data.doc_name) {
        _origGraphMeta = {
            doc_id: data.doc_id || ((_origGraphMeta || {}).doc_id || ''),
            doc_name: data.doc_name || ((_origGraphMeta || {}).doc_name || ''),
        };
    }
    _pyManualReviewEdges = Array.isArray(data.manual_review_edges) ? data.manual_review_edges : [];
    _pyCausal.forEach(function(e, i) {
        var eid = 'c' + i;
        var manualReview = !!e.manual_review;
        var style = _edgeStyle('causal', manualReview);
        network.body.data.edges.add([{
            id: eid, from: e.from, to: e.to, arrows: 'to',
            color: style.color,
            width: style.width, dashes: style.dashes, title: _edgeTitle(e.reason, manualReview),
        }]);
        _edgeLookup[eid] = {type:'causal', from:String(e.from), to:String(e.to), reason:e.reason, manualReview:manualReview};
    });
    _pyTemporal.forEach(function(e, i) {
        var eid = 't' + i;
        var manualReview = !!e.manual_review;
        var style = _edgeStyle('temporal', manualReview);
        network.body.data.edges.add([{
            id: eid, from: e.from, to: e.to, arrows: 'to',
            color: style.color,
            width: style.width, dashes: style.dashes, title: _edgeTitle(e.reason, manualReview),
        }]);
        _edgeLookup[eid] = {type:'temporal', from:String(e.from), to:String(e.to), reason:e.reason||'', manualReview:manualReview};
    });
    _updateGraphInfo(data);
    _renderEventPanel();
}
function downloadGraph(data) {
    var blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'graph.json';
    a.click();
}
function showToast(msg, duration, bg) {
    var toast = document.getElementById('save-toast');
    if (!toast) return;
    toast.textContent = msg;
    toast.style.background = bg || '#333';
    toast.style.display = 'block';
    toast.style.opacity = '1';
    if (toast._timer) clearTimeout(toast._timer);
    toast._timer = setTimeout(function() {
        toast.style.opacity = '0';
        setTimeout(function() { toast.style.display = 'none'; }, 400);
    }, duration || 2500);
}

/* ── 事件面板（列表/全文切换） ─────────────────────────────── */
function toggleEventList() {
    _evtMode = (_evtMode === 'list') ? 'doc' : 'list';
    _renderEventPanel();
}
function _renderEventPanel() {
    var body = document.getElementById('evt-list-body');
    var btn  = document.getElementById('btn-evtlist');
    if (!body) return;
    if (_evtMode === 'doc') {
        body.innerHTML = '<pre style="white-space:pre-wrap;margin:0;font-size:12px;font-family:sans-serif;'
            + 'color:#333;word-break:break-all;">' + _esc(_pyDocText || '（无全文）') + '</pre>';
        if (btn) btn.textContent = '正在显示全文 ▲（点击返回事件列表）';
    } else {
        var html = '';
        (_pyEvents || []).forEach(function(e) {
            html += '<div style="margin-bottom:6px;">'
                  + '<span style="font-weight:700;color:#1f77b4;">' + _esc(e.id) + '</span>'
                  + '<span style="color:#555;">: ' + _esc(e.text) + '</span></div>';
        });
        body.innerHTML = html;
        if (btn) btn.textContent = '事件列表 ▼（点击查看全文）';
    }
}
/* ── 候选面板拖动 ───────────────────────────────────────── */
var _candDrag = null;
document.addEventListener('mousedown', function(e) {
    var handle = document.getElementById('candidate-drag-handle');
    if (!handle || !handle.contains(e.target)) return;
    if (e.button !== 0) return;
    var panel = document.getElementById('candidate-panel');
    if (!panel) return;
    var rect = panel.getBoundingClientRect();
    // 如果还在用百分比定位，先转换为具体像素坐标
    panel.style.top  = rect.top  + 'px';
    panel.style.left = rect.left + 'px';
    panel.style.marginLeft = '0';
    _candDrag = { startX: e.clientX, startY: e.clientY,
                  initLeft: rect.left, initTop: rect.top };
    e.preventDefault();
});
document.addEventListener('mousemove', function(e) {
    if (!_candDrag) return;
    var panel = document.getElementById('candidate-panel');
    if (!panel) return;
    panel.style.left = (_candDrag.initLeft + e.clientX - _candDrag.startX) + 'px';
    panel.style.top  = (_candDrag.initTop  + e.clientY - _candDrag.startY) + 'px';
});
document.addEventListener('mouseup', function(e) {
    if (e.button === 0) _candDrag = null;
});
/* ── 初始化 ────────────────────────────────────────────────── */
window.addEventListener('load', function() {
    (_pyCausal || []).forEach(function(e, i) {
        _edgeLookup['c' + i] = {type:'causal', from:e.from, to:e.to, reason:e.reason, manualReview:!!e.manual_review};
    });
    (_pyTemporal || []).forEach(function(e, i) {
        _edgeLookup['t' + i] = {type:'temporal', from:e.from, to:e.to, reason:e.reason, manualReview:!!e.manual_review};
    });
    if (typeof network !== 'undefined') {
        network.on('click', function(params) {
            if (!_editMode) { closeAllMenus(); return; }
            closeEdgeMenu();
            closeNodeMenu();
            // 如果处于候选选择状态，任意点击画布空白处取消
            if (_addEdge && params.nodes.length === 0 && params.edges.length === 0) {
                closeCandidatePanel();
                return;
            }
            if (params.nodes.length > 0) {
                // 如果候选面板已展开，直接调 selectCandidate
                var panel = document.getElementById('candidate-panel');
                if (panel && panel.style.display !== 'none') {
                    selectCandidate(String(params.nodes[0]));
                } else {
                    showNodeMenu(params.nodes[0], params.pointer.DOM.x, params.pointer.DOM.y);
                }
            } else if (params.edges.length > 0) {
                if (document.getElementById('candidate-panel').style.display !== 'none') return;
                showEdgeMenu(params.edges[0], params.pointer.DOM.x, params.pointer.DOM.y);
            }
        });
    }
    _updateGraphInfo({
        doc_name: (_origGraphMeta || {}).doc_name || '',
        causal_edges: _pyCausal || [],
        temporal_edges: _pyTemporal || [],
    });
    _renderEventPanel();
    _loadLatestGraph();
});
"""


def _js_data_block(
    causal_edges: list[dict],
    temporal_edges: list[dict],
    events: list[dict],
    graph_json_path: str,
    doc_text: str = "",
    meta: dict | None = None,
    manual_review_edges: list[dict] | None = None,
    save_port: int = _SAVE_PORT,
) -> str:
    """生成含图数据的 <script> 标签（只含变量声明，逻辑在 _JS_LOGIC 中）。"""
    causal_js = json.dumps(
        [
            {
                "from": e["cause"],
                "to": e["effect"],
                "reason": e.get("reason", ""),
                "manual_review": bool(e.get("manual_review")),
            }
            for e in causal_edges
        ],
        ensure_ascii=False,
    )
    temporal_js = json.dumps(
        [
            {
                "from": e["source"],
                "to": e["target"],
                "reason": e.get("reason", ""),
                "manual_review": bool(e.get("manual_review")),
            }
            for e in temporal_edges
        ],
        ensure_ascii=False,
    )
    events_js = json.dumps(
        [{"id": ev["event_id"], "text": ev.get("text", "")} for ev in events],
        ensure_ascii=False,
    )
    meta_js = json.dumps(meta or {"doc_id": "", "doc_name": ""}, ensure_ascii=False)
    path_js = json.dumps(graph_json_path, ensure_ascii=False)
    doc_text_js = json.dumps(doc_text, ensure_ascii=False)
    manual_review_edges_js = json.dumps(manual_review_edges or [], ensure_ascii=False)
    return (
        f"<script>"
        f"var _pyCausal={causal_js};"
        f"var _pyTemporal={temporal_js};"
        f"var _pyEvents={events_js};"
        f"var _origGraphMeta={meta_js};"
        f"var _graphJsonPath={path_js};"
        f"var _savePort={save_port};"
        f"var _pyDocText={doc_text_js};"
        f"var _pyManualReviewEdges={manual_review_edges_js};"
        f"</script>"
    )


def _try_start_review_server(graph_path: Path) -> int:
    """尝试在后台启动 ManualReviewServer（端口已占用则静默跳过）。"""
    import socket
    import subprocess
    import sys as _sys
    import time
    import urllib.request

    def _is_running(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
                _s.settimeout(0.3)
                return _s.connect_ex(("localhost", port)) == 0
        except Exception:
            return False

    def _is_healthy(port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=0.5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return bool(payload.get("ok") and payload.get("service") == "manual-review-v2")
        except Exception:
            return False

    def _free_port() -> int:
        for port in range(_SAVE_PORT + 1, _SAVE_PORT + 80):
            if not _is_running(port):
                return port
        return _SAVE_PORT

    if _is_healthy(_SAVE_PORT):
        return _SAVE_PORT

    port = _SAVE_PORT if not _is_running(_SAVE_PORT) else _free_port()
    try:
        server_script = Path(__file__).parent / "manual_review.py"
        if not server_script.exists():
            return port
        kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if _sys.platform == "win32":
            kwargs["creationflags"] = 8 | 512  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(
            [
                _sys.executable,
                str(server_script),
                "--graph-path",
                str(graph_path),
                "--port",
                str(port),
                "--no-browser",
            ],
            **kwargs,
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if _is_healthy(port):
                return port
            time.sleep(0.2)
    except Exception:
        pass
    return port


def _collect_viz_docs(output_path: Path, current_graph: dict) -> list[dict]:
    """扫描 visualization 目录，收集所有已渲染文档信息，用于文档链接面板。"""
    viz_dir = output_path.parent.parent  # output/visualization/
    doc_results_dir = output_path.parent.parent.parent / "doc_results"
    current_doc_id = output_path.parent.name
    found: dict[str, Path] = {p.parent.name: p for p in sorted(viz_dir.glob("*/graph.html"))}
    found.setdefault(current_doc_id, output_path)
    docs: list[dict] = []
    for doc_id in sorted(found.keys()):
        if doc_id == current_doc_id:
            doc_name: str = current_graph.get("doc_name") or doc_id
        else:
            graph_json = doc_results_dir / doc_id / "graph.json"
            doc_name = doc_id
            if graph_json.exists():
                try:
                    g = read_json_utf8(graph_json)
                    doc_name = g.get("doc_name") or doc_id
                except Exception:
                    pass
        rel_path = "graph.html" if doc_id == current_doc_id else f"../{doc_id}/graph.html"
        docs.append({"doc_id": doc_id, "doc_name": doc_name, "rel_path": rel_path, "current": doc_id == current_doc_id})
    return docs


def _right_panels_html(events: list[dict], docs: list[dict]) -> str:
    """右侧弹性列容器：上为事件列表，下为文档网络链接。"""
    events_panel = (
        '<div id="events-panel" style="background:#ffffffee;border:1px solid #c0c0c0;'
        'border-radius:8px;padding:10px 14px;font-family:sans-serif;font-size:12px;'
        'line-height:1.5;box-shadow:0 2px 6px rgba(0,0,0,0.10);overflow-y:auto;'
        'flex:1 1 0;min-height:0;">'
        '<button id="btn-evtlist" onclick="toggleEventList()" '
        'style="width:100%;text-align:left;font-weight:700;font-size:13px;'
        'border:none;background:none;cursor:pointer;padding:0 0 6px 0;'
        'color:#333;border-bottom:1px solid #e0e0e0;margin-bottom:8px;">'
        '\u4e8b\u4ef6\u5217\u8868 \u25bc</button>'
        '<div id="evt-list-body"></div>'
        '</div>'
    )
    doc_items = ""
    for i, doc in enumerate(docs):
        title = _strip_doc_suffix(doc["doc_name"])
        if doc["current"]:
            doc_items += (
                f'<div style="padding:3px 0;font-size:12px;color:#1f77b4;font-weight:700;">'
                f'{i + 1}\u00a0{_html.escape(title)}</div>'
            )
        else:
            doc_items += (
                f'<a href="{doc["rel_path"]}" '
                f'style="display:block;padding:3px 0;font-size:12px;color:#555;'
                f'text-decoration:none;line-height:1.5;" '
                f'onmouseover="this.style.color=\'#1f77b4\'" '
                f'onmouseout="this.style.color=\'#555\'">'
                f'{i + 1}\u00a0{_html.escape(title)}</a>'
            )
    doc_links_panel = (
        '<div id="doc-links-panel" style="background:#ffffffee;border:1px solid #c0c0c0;'
        'border-radius:8px;padding:8px 14px;font-family:sans-serif;line-height:1.5;'
        'box-shadow:0 2px 6px rgba(0,0,0,0.10);overflow-y:auto;'
        'flex:0 0 auto;max-height:40%;min-height:36px;">'
        '<div style="font-weight:700;font-size:13px;color:#333;border-bottom:1px solid #e0e0e0;'
        'padding-bottom:4px;margin-bottom:5px;">\u6587\u6863\u7f51\u7edc</div>'
        + doc_items +
        '</div>'
    ) if docs else ""
    return (
        '<div style="position:fixed;top:14px;right:14px;bottom:14px;width:300px;'
        'z-index:9999;display:flex;flex-direction:column;gap:8px;">'
        + events_panel + doc_links_panel +
        '</div>'
    )


def _info_box_html(doc_title: str, causal_count: int, temporal_count: int) -> str:
    return (
        '<div style="position:fixed;top:14px;left:14px;z-index:9999;'
        'background:#ffffffcc;border:1px solid #c0c0c0;border-radius:8px;'
        'padding:10px 14px;font-family:sans-serif;font-size:13px;line-height:1.5;'
        'box-shadow:0 2px 6px rgba(0,0,0,0.08);max-width:340px;">'
        f'<div id="doc-title" style="font-weight:600;font-size:14px;margin-bottom:4px;">{_html.escape(doc_title)}</div>'
        f'<div>\u56e0\u679c\u8fb9\uff1a<b id="causal-count">{causal_count}</b> \u6761'
        f'&emsp;\u65f6\u5e8f\u8fb9\uff1a<b id="temporal-count">{temporal_count}</b> \u6761</div>'
        '<div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;">'
        '<button id="btn-mode" onclick="toggleSimplifiedMode()" '
        'style="cursor:pointer;padding:3px 10px;font-size:12px;border:1px solid #aaa;'
        'border-radius:4px;background:#f5f5f5;">\u8be6\u7ec6\u6a21\u5f0f</button>'
        '<button id="btn-edit" onclick="toggleEditMode()" '
        'style="cursor:pointer;padding:3px 10px;font-size:12px;border:1px solid #aaa;'
        'border-radius:4px;background:#f5f5f5;">\u4fee\u6539</button>'
        '<button id="btn-save" onclick="saveGraph()" '
        'style="cursor:pointer;padding:3px 10px;font-size:12px;border:1px solid #aaa;'
        'border-radius:4px;background:#f5f5f5;">\u4fdd\u5b58</button>'
        '</div>'
        '<div id="save-status" style="font-size:11px;color:#e67e22;margin-top:4px;'
        'min-height:14px;"></div>'
        '</div>'
    )


def _edit_overlay_html() -> str:
    """边/节点编辑的菜单和对话框 HTML（初始隐藏）。"""
    return (
        # 保存 toast（居中顶部弹窗）
        '<div id="save-toast" style="display:none;position:fixed;top:20px;left:50%;transform:translateX(-50%);'
        'z-index:10010;color:#fff;font-family:sans-serif;font-size:14px;font-weight:600;'
        'padding:10px 22px;border-radius:8px;box-shadow:0 3px 12px rgba(0,0,0,0.25);'
        'transition:opacity 0.4s;pointer-events:none;"></div>'
        # 边右键菜单
        '<div id="edge-ctx-menu" style="display:none;position:fixed;z-index:10000;'
        'background:#fff;border:1px solid #c0c0c0;border-radius:6px;'
        'padding:8px 0;font-family:sans-serif;font-size:13px;'
        'box-shadow:0 3px 8px rgba(0,0,0,0.15);min-width:140px;">'
        '<div id="ctx-type-label" style="padding:4px 14px;color:#666;font-size:11px;'
        'border-bottom:1px solid #eee;margin-bottom:4px;"></div>'
        '<div onclick="menuDelete()" '
        'style="padding:6px 14px;cursor:pointer;" '
        'onmouseover="this.style.background=\'#fee2e2\'" onmouseout="this.style.background=\'\'">删除</div>'
        '<div onclick="menuModify()" '
        'style="padding:6px 14px;cursor:pointer;" '
        'onmouseover="this.style.background=\'#fef3c7\'" onmouseout="this.style.background=\'\'">修改类型</div>'
        '<div id="ctx-reverse-btn" onclick="menuReverse()" '
        'style="padding:6px 14px;cursor:pointer;" '
        'onmouseover="if(this.getAttribute(\'data-disabled\')!==\'1\')this.style.background=\'#e8f5e9\'" '
        'onmouseout="this.style.background=\'\'">修改方向</div>'
        '<div onclick="closeEdgeMenu()" '
        'style="padding:6px 14px;cursor:pointer;color:#999;" '
        'onmouseover="this.style.background=\'#f5f5f5\'" onmouseout="this.style.background=\'\'">取消</div>'
        '</div>'
        # 节点点击菜单（新增边）
        '<div id="node-ctx-menu" style="display:none;position:fixed;z-index:10000;'
        'background:#fff;border:1px solid #c0c0c0;border-radius:6px;'
        'padding:8px 0;font-family:sans-serif;font-size:13px;'
        'box-shadow:0 3px 8px rgba(0,0,0,0.15);min-width:220px;">'
        '<div onclick="startAddEdge(\'target\')" '
        'style="padding:6px 14px;cursor:pointer;" '
        'onmouseover="this.style.background=\'#e8f5e9\'" onmouseout="this.style.background=\'\'">以该节点为终点，选择起点：</div>'
        '<div onclick="startAddEdge(\'source\')" '
        'style="padding:6px 14px;cursor:pointer;" '
        'onmouseover="this.style.background=\'#e8f5e9\'" onmouseout="this.style.background=\'\'">以该节点为起点，选择终点：</div>'
        '<div onclick="closeNodeMenu()" '
        'style="padding:6px 14px;cursor:pointer;color:#999;" '
        'onmouseover="this.style.background=\'#f5f5f5\'" onmouseout="this.style.background=\'\'">取消</div>'
        '</div>'
        # 候选节点面板（可拖动）
        '<div id="candidate-panel" style="display:none;position:fixed;top:20%;left:50%;'
        'margin-left:-150px;z-index:10001;background:#fff;'
        'border:1px solid #c0c0c0;border-radius:10px;'
        'font-family:sans-serif;min-width:300px;max-width:420px;'
        'max-height:65vh;overflow:hidden;flex-direction:column;'
        'box-shadow:0 4px 16px rgba(0,0,0,0.2);">'
        '<div id="candidate-drag-handle" '
        'style="display:flex;justify-content:space-between;align-items:center;'
        'padding:10px 14px 8px 14px;cursor:grab;background:#f5f5f5;'
        'border-bottom:1px solid #e0e0e0;border-radius:10px 10px 0 0;flex-shrink:0;">'
        '<div id="candidate-title" style="font-weight:700;font-size:14px;user-select:none;"></div>'
        '<button onclick="closeCandidatePanel()" '
        'style="border:none;background:none;cursor:pointer;font-size:16px;color:#999;padding:0;">✕</button>'
        '</div>'
        '<div id="candidate-body" style="padding:10px 14px;overflow-y:auto;max-height:55vh;"></div>'
        '</div>'
        # 新建边对话框
        '<div id="new-edge-dlg" style="display:none;position:fixed;inset:0;z-index:10002;'
        'background:rgba(0,0,0,0.4);align-items:center;justify-content:center;">'
        '<div style="background:#fff;border-radius:10px;padding:22px 26px;'
        'font-family:sans-serif;min-width:320px;box-shadow:0 4px 16px rgba(0,0,0,0.2);">'
        '<div style="font-weight:700;font-size:15px;margin-bottom:4px;">新增边</div>'
        '<div id="new-edge-subtitle" style="font-size:13px;color:#555;margin-bottom:14px;"></div>'
        '<label style="display:block;font-size:13px;margin-bottom:4px;">类型</label>'
        '<select id="new-edge-type" style="width:100%;box-sizing:border-box;font-size:13px;'
        'border:1px solid #ccc;border-radius:4px;padding:5px;margin-bottom:10px;">'
        '<option value="causal">因果边</option><option value="temporal">时序边</option>'
        '</select>'
        '<label style="display:block;font-size:13px;margin-bottom:6px;">原因（必填）</label>'
        '<textarea id="new-edge-reason" rows="3" style="width:100%;box-sizing:border-box;'
        'font-size:13px;border:1px solid #ccc;border-radius:4px;padding:6px;resize:vertical;"></textarea>'
        '<div style="margin-top:12px;display:flex;gap:8px;justify-content:flex-end;">'
        '<button onclick="closeNewEdgeDlg()" '
        'style="padding:5px 14px;border:1px solid #ccc;border-radius:4px;cursor:pointer;background:#f5f5f5;">取消</button>'
        '<button onclick="applyNewEdge()" '
        'style="padding:5px 14px;border:none;border-radius:4px;cursor:pointer;'
        'background:#27ae60;color:#fff;font-weight:600;">新增</button>'
        '</div></div></div>'
        # 修改对话框（遮罩居中）
        '<div id="modify-dlg" style="display:none;position:fixed;inset:0;z-index:10002;'
        'background:rgba(0,0,0,0.4);align-items:center;justify-content:center;">'
        '<div style="background:#fff;border-radius:10px;padding:22px 26px;'
        'font-family:sans-serif;min-width:320px;box-shadow:0 4px 16px rgba(0,0,0,0.2);">'
        '<div id="dlg-title" style="font-weight:700;font-size:15px;margin-bottom:14px;"></div>'
        '<label style="display:block;font-size:13px;margin-bottom:6px;">修改原因（必填）</label>'
        '<textarea id="dlg-reason" rows="3" style="width:100%;box-sizing:border-box;'
        'font-size:13px;border:1px solid #ccc;border-radius:4px;padding:6px;resize:vertical;"></textarea>'
        '<div style="margin-top:12px;display:flex;gap:8px;justify-content:flex-end;">'
        '<button onclick="closeModifyDlg()" '
        'style="padding:5px 14px;border:1px solid #ccc;border-radius:4px;cursor:pointer;background:#f5f5f5;">取消</button>'
        '<button onclick="applyModify()" '
        'style="padding:5px 14px;border:none;border-radius:4px;cursor:pointer;'
        'background:#f5a623;color:#fff;font-weight:600;">确认修改</button>'
        '</div></div></div>'
    )


# -- pyvis 渲染 ----------------------------------------------------------------

def _render_with_pyvis(graph: dict[str, Any], output_path: Path, graph_path: Path, save_port: int) -> bool:
    try:
        from pyvis.network import Network
    except Exception:
        return False

    events = sorted(graph.get("events", []), key=lambda ev: event_id_num(ev["event_id"]))
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
            "color": {
              "background": "#dbeeff", "border": "#3a7bd5",
              "highlight": { "background": "#b8d9f8", "border": "#1a4fa0" },
              "hover":     { "background": "#c9e8ff", "border": "#2a5ec0" }
            },
            "margin": 8
          }
        }
        """
    )

    for event in events:
        eid = event["event_id"]
        text = event.get("text", "")
        net.add_node(eid, label=_node_label(eid, text), shape="box")

    # 使用预设 ID（c0, c1, ..., t0, t1, ...）以便 JS 侧可靠追踪
    for i, edge in enumerate(causal_edges):
        manual_review = bool(edge.get("manual_review"))
        title = _edge_title(edge.get("reason", ""), manual_review)
        color = MANUAL_REVIEW_COLOR if manual_review else CAUSAL_COLOR
        net.add_edge(
            edge["cause"], edge["effect"],
            id=f"c{i}",
            color=color, width=3,
            title=_html.escape(title),
            arrows="to",
        )
    for i, edge in enumerate(temporal_edges):
        manual_review = bool(edge.get("manual_review"))
        title = _edge_title(edge.get("reason", ""), manual_review)
        color = MANUAL_REVIEW_COLOR if manual_review else TEMPORAL_COLOR
        net.add_edge(
            edge["source"], edge["target"],
            id=f"t{i}",
            color=color, width=1.6, dashes=True,
            title=_html.escape(title),
            arrows="to",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = net.generate_html(notebook=False)
    import re as _re
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

    # 注入数据变量 + 逻辑 JS + UI 元素
    data_js = _js_data_block(
        causal_edges, temporal_edges, events,
        str(graph_path.resolve()),
        doc_text=graph.get("doc_text", ""),
        meta={"doc_id": graph.get("doc_id", ""), "doc_name": graph.get("doc_name", "")},
        manual_review_edges=graph.get("manual_review_edges", []),
        save_port=save_port,
    )
    logic_js = f"<script>{_JS_LOGIC}</script>"
    info_html = _info_box_html(
        _strip_doc_suffix(graph.get("doc_name") or graph.get("doc_id", "")),
        len(causal_edges),
        len(temporal_edges),
    )
    docs = _collect_viz_docs(output_path, graph)
    events_panel = _right_panels_html(events, docs)
    overlays = _edit_overlay_html()
    insert = data_js + logic_js + info_html + events_panel + overlays
    if "</body>" in html:
        html = html.replace("</body>", insert + "</body>", 1)
    else:
        html += insert
    output_path.write_text(html, encoding="utf-8")
    return True


# -- SVG fallback -------------------------------------------------------------

def _render_with_svg(graph: dict[str, Any], output_path: Path) -> None:
    events = sorted(graph.get("events", []), key=lambda ev: event_id_num(ev["event_id"]))
    causal_edges = graph.get("causal_edges", [])
    temporal_edges = graph.get("temporal_edges", [])
    if not events:
        write_text_utf8(output_path, "<html><body>No events.</body></html>")
        return

    radius, gap = 32, 110
    width = max(720, len(events) * (radius * 2 + gap))
    height = 460
    cy = height // 2
    positions: dict[str, tuple[int, int]] = {}
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'style="font-family:sans-serif">',
        '<defs>'
        f'<marker id="arrC" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{CAUSAL_COLOR}"/></marker>'
        f'<marker id="arrT" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{TEMPORAL_COLOR}"/></marker>'
        f'<marker id="arrM" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{MANUAL_REVIEW_COLOR}"/></marker>'
        '</defs>',
    ]

    for idx, event in enumerate(events):
        cx = gap + idx * (radius * 2 + gap)
        positions[event["event_id"]] = (cx, cy)
        label = event["event_id"]
        parts.append(
            f'<rect x="{cx - radius}" y="{cy - 14}" width="{radius * 2}" height="28" '
            f'rx="4" fill="{NODE_BG_COLOR}" stroke="{NODE_BORDER_COLOR}" stroke-width="2"/>'
            f'<text x="{cx}" y="{cy + radius + 16}" text-anchor="middle" font-size="12">'
            f'{_html.escape(label)}</text>'
        )

    def _arc(a: str, b: str, *, color: str, dashed: bool, marker: str, offset: int, reason: str) -> str:
        if a not in positions or b not in positions:
            return ""
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        mid_x, mid_y = (x1 + x2) // 2, y1 + offset
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        return (
            f'<path d="M {x1} {y1} Q {mid_x} {mid_y} {x2} {y2}" fill="none" '
            f'stroke="{color}" stroke-width="2"{dash} marker-end="url(#{marker})">'
            f'<title>{_html.escape(reason)}</title></path>'
        )

    for edge in causal_edges:
        manual_review = bool(edge.get("manual_review"))
        parts.append(_arc(
            edge["cause"], edge["effect"],
            color=MANUAL_REVIEW_COLOR if manual_review else CAUSAL_COLOR,
            dashed=False,
            marker="arrM" if manual_review else "arrC",
            offset=-110,
            reason=_edge_title(edge.get("reason", ""), manual_review),
        ))
    for edge in temporal_edges:
        manual_review = bool(edge.get("manual_review"))
        parts.append(_arc(
            edge["source"], edge["target"],
            color=MANUAL_REVIEW_COLOR if manual_review else TEMPORAL_COLOR,
            dashed=True,
            marker="arrM" if manual_review else "arrT",
            offset=110,
            reason=_edge_title(edge.get("reason", ""), manual_review),
        ))

    title = _strip_doc_suffix(graph.get("doc_name") or graph.get("doc_id", ""))
    parts.append(
        f'<g transform="translate(14,14)">'
        f'<rect width="280" height="64" rx="8" ry="8" fill="#ffffffcc" stroke="#c0c0c0"/>'
        f'<text x="14" y="22" font-size="14" font-weight="600">{_html.escape(title)}</text>'
        f'<text x="14" y="42" font-size="12">\u56e0\u679c\u8fb9\uff1a{len(causal_edges)} \u6761</text>'
        f'<text x="14" y="58" font-size="12">\u65f6\u5e8f\u8fb9\uff1a{len(temporal_edges)} \u6761</text>'
        f'</g>'
    )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_utf8(output_path, "".join(parts))


# -- 入口 ---------------------------------------------------------------------

def render_graph(graph_path: Path, output_path: Path) -> Path:
    """读 graph.json 与默认 pyvis 渲染。失败时退到 SVG。"""
    payload = read_json_utf8(graph_path)
    if not isinstance(payload, dict):
        raise ValueError(f"graph.json \u683c\u5f0f\u975e\u6cd5: {graph_path}")
    save_port = _try_start_review_server(graph_path)
    if not _render_with_pyvis(payload, output_path, graph_path, save_port):
        _render_with_svg(payload, output_path)
    return output_path


__all__ = ["render_graph"]


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="重新渲染 graph.html")
    _p.add_argument("--graph-path", required=True)
    _p.add_argument("--output-path", required=True)
    _a = _p.parse_args()
    render_graph(Path(_a.graph_path), Path(_a.output_path))