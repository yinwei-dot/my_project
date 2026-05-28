from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt_number(value: Any, digits: int = 4) -> str:
    try:
        num = float(value)
    except Exception:
        return "N/A"
    if math.isnan(num) or math.isinf(num):
        return "N/A"
    return f"{num:.{digits}f}"


def _sparkline_svg(values: list[float], width: int = 220, height: int = 56, color: str = "#d62728") -> str:
    usable = [float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v))]
    if len(usable) < 2:
        return '<div style="color:#999;font-size:11px;">无历史曲线</div>'

    lo = min(usable)
    hi = max(usable)
    span = max(hi - lo, 1e-8)
    pts: list[str] = []
    for i, value in enumerate(usable):
        x = i * (width - 8) / max(1, len(usable) - 1) + 4
        y = height - 4 - (value - lo) * (height - 8) / span
        pts.append(f"{x:.1f},{y:.1f}")

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="0.5" y="0.5" width="{width-1}" height="{height-1}" '
        'rx="6" fill="#fcfcfd" stroke="#e5e7eb"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}"/>'
        f'<text x="6" y="14" font-size="10" fill="#666">min {lo:.3f}</text>'
        f'<text x="6" y="{height-6}" font-size="10" fill="#666">max {hi:.3f}</text>'
        '</svg>'
    )


def _render_run_card(run_name: str, run_dir: Path) -> str:
    history_path = run_dir / "history.json"
    best_meta_path = run_dir / "best_meta.json"
    latest_meta_path = run_dir / "latest_meta.json"

    history = _read_json(history_path) if history_path.exists() else []
    best_meta = _read_json(best_meta_path) if best_meta_path.exists() else {}
    latest_meta = _read_json(latest_meta_path) if latest_meta_path.exists() else {}

    history = history if isinstance(history, list) else []
    best_meta = best_meta if isinstance(best_meta, dict) else {}
    latest_meta = latest_meta if isinstance(latest_meta, dict) else {}

    train_curve = [row.get("train_loss") for row in history if isinstance(row, dict) and row.get("train_loss") is not None]
    val_curve = [row.get("val_loss") for row in history if isinstance(row, dict) and row.get("val_loss") is not None]
    last_row = history[-1] if history and isinstance(history[-1], dict) else {}

    detail_rows = [
        ("best epoch", best_meta.get("epoch", latest_meta.get("epoch", "N/A"))),
        ("best val", _fmt_number(best_meta.get("best_val_loss", best_meta.get("val_loss")))),
        ("latest train", _fmt_number(latest_meta.get("train_loss", last_row.get("train_loss")))),
        ("latest robust", _fmt_number(last_row.get("loss_robust"))),
        ("latest causal", _fmt_number(last_row.get("loss_causal"))),
        ("latest total", _fmt_number(last_row.get("loss_total"))),
    ]
    details_html = "".join(
        f'<div style="display:flex;justify-content:space-between;gap:12px;">'
        f'<span style="color:#666;">{html.escape(str(k))}</span>'
        f'<span style="font-weight:600;color:#222;">{html.escape(str(v))}</span></div>'
        for k, v in detail_rows
    )

    return (
        '<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;'
        'padding:14px;box-shadow:0 4px 14px rgba(0,0,0,0.04);">'
        f'<div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:8px;">{html.escape(run_name)}</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start;">'
        '<div>'
        '<div style="font-size:12px;color:#666;margin-bottom:4px;">train loss</div>'
        f'{_sparkline_svg([float(v) for v in train_curve if v is not None], color="#d62728")}'
        '<div style="font-size:12px;color:#666;margin:8px 0 4px 0;">val loss</div>'
        f'{_sparkline_svg([float(v) for v in val_curve if v is not None], color="#1f77b4")}'
        '</div>'
        '<div style="display:flex;flex-direction:column;gap:6px;font-size:12px;">'
        f'{details_html}'
        f'<div style="color:#888;margin-top:4px;">history records: {len(history)}</div>'
        '</div>'
        '</div>'
        '</div>'
    )


def generate_training_report(
    model_root: Path,
    output_path: Path,
    *,
    hetgat_cfg: Any | None = None,
    vicreg_cfg: Any | None = None,
    train_cfg: Any | None = None,
    scorer_cfg: Any | None = None,
) -> Path:
    """从已有训练产物生成 HTML 训练报告页。"""
    run_dirs = [p for p in sorted(model_root.iterdir()) if p.is_dir()]

    cfg_rows = [
        ("HetGAT hidden_dims", getattr(hetgat_cfg, "hidden_dims", None)),
        ("HetGAT num_heads", getattr(hetgat_cfg, "num_heads", None)),
        ("dropout", getattr(hetgat_cfg, "dropout", None)),
        ("beta", getattr(vicreg_cfg, "beta", None)),
        ("lambda / mu / nu", f'{getattr(vicreg_cfg, "lambda_", "N/A")} / {getattr(vicreg_cfg, "mu", "N/A")} / {getattr(vicreg_cfg, "nu", "N/A")}'),
        ("epochs", getattr(train_cfg, "epochs", None)),
        ("n_folds", getattr(train_cfg, "n_folds", None)),
        ("w_rich / w_rare", f'{getattr(scorer_cfg, "w_rich", "N/A")} / {getattr(scorer_cfg, "w_rare", "N/A")}'),
    ]
    cfg_html = "".join(
        f'<div style="display:flex;justify-content:space-between;gap:16px;padding:6px 0;border-bottom:1px dashed #eceff3;">'
        f'<span style="color:#666;">{html.escape(str(k))}</span>'
        f'<span style="font-weight:600;color:#111827;">{html.escape(str(v))}</span></div>'
        for k, v in cfg_rows
    )

    cards = "".join(_render_run_card(run_dir.name, run_dir) for run_dir in run_dirs)
    if not cards:
        cards = '<div style="color:#999;">未找到任何训练产物目录。</div>'

    html_text = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>semantic_model training report</title>'
        '<style>'
        'body{margin:0;padding:28px;font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;'
        'background:linear-gradient(180deg,#f6f8fb 0%,#eef3f9 100%);color:#111827;}'
        '.wrap{max-width:1280px;margin:0 auto;}.hero{margin-bottom:18px;}.hero h1{margin:0 0 8px 0;font-size:28px;}'
        '.hero p{margin:0;color:#4b5563;}.grid{display:grid;grid-template-columns:340px 1fr;gap:18px;align-items:start;}'
        '.panel{background:#ffffffcc;border:1px solid #e5e7eb;border-radius:14px;padding:16px;backdrop-filter:blur(8px);}'
        '.cards{display:grid;grid-template-columns:1fr;gap:14px;}'
        '@media (max-width:960px){body{padding:16px}.grid{grid-template-columns:1fr}}'
        '</style></head><body><div class="wrap">'
        '<div class="hero"><h1>semantic_model 训练报告</h1>'
        f'<p>模型目录：{html.escape(str(model_root))}</p></div>'
        '<div class="grid">'
        '<div class="panel"><div style="font-size:16px;font-weight:700;margin-bottom:10px;">配置摘要</div>'
        f'{cfg_html}</div>'
        '<div class="cards">'
        f'{cards}'
        '</div></div></div></body></html>'
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path