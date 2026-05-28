"""
Topo_model 完整流水线入口
========================
步骤：
  1. 真实标签生成 + 数据划分清单（CreateGroundTruth + DataSplit）
  2. 特征工程（FeatureEngineering，仅 labeled 图）
  3. 模型训练（TrainModel，仅 manifest.train）
  4. 图推理评分（ScoreGraph，≥3 节点全部图）
  5. 打印：test 集对比表 + >30 节点仅分数

用法示例：
  python main.py
  python main.py --skip-train
  python main.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from pathlib import Path

_BASE = Path(__file__).resolve().parent
_SRC  = _BASE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import TopoConfig, ModelConfig, TrainConfig
from FeatureEngineering import process_all as feat_process_all
from CreateGroundTruth import process_all as gt_process_all
from TrainModel import (
    run_kfold, run_final, eval_on_holdout_test,
    _setup_train_logging,
)
from Dataset import TopoDataset, _setup_logging as _setup_data_logging
from DataSplit import get_doc_ids, load_manifest, MANIFEST_NAME
from ScoreGraph import score_all

GRAPH_ROOT    = _BASE / "data"   / "output" / "doc_results"
FEAT_ROOT     = _BASE / "output" / "features"
LABEL_ROOT    = _BASE / "output" / "labels"
MODEL_ROOT    = _BASE / "output" / "models"
SCORE_ROOT    = _BASE / "output" / "topology_scores"
MANIFEST_PATH = _BASE / "output" / MANIFEST_NAME
LOG_ROOT      = _BASE / "log"


def _make_logger(name: str) -> logging.Logger:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fh = logging.FileHandler(LOG_ROOT / f"{name}.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _dw(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(s))


def _cell(s, width: int, align: str = "<") -> str:
    s = str(s)
    pad = max(0, width - _dw(s)) * " "
    return (s + pad) if align == "<" else (pad + s)


def _model_rank_no_tie(scores: dict) -> dict[str, int]:
    ranked = sorted(scores.items(), key=lambda x: (-x[1]["score"], x[0]))
    return {nid: i + 1 for i, (nid, _) in enumerate(ranked)}


def _best_set_hit(removal_sets: list[list[str]], model_rank: dict[str, int]) -> tuple[float, list[str]]:
    if not removal_sets:
        return 0.0, []
    sorted_nodes = sorted(model_rank, key=lambda x: model_rank[x])
    best_rate = 0.0
    best_set: list[str] = []
    for removal_set in removal_sets:
        if not removal_set:
            continue
        n = len(removal_set)
        top_n = set(sorted_nodes[:n])
        hit = len(top_n & set(removal_set))
        rate = hit / n
        if rate > best_rate:
            best_rate = rate
            best_set = removal_set
    return best_rate, best_set


def _print_comparison(
    score_root: Path,
    label_root: Path,
    test_doc_ids: set[str],
) -> None:
    """仅对 hold-out test 集打印真实排名 vs 模型推理排名。"""
    if not test_doc_ids:
        print("[ 无 test 集文档 ]")
        return

    print("\n" + "=" * 72)
    print("  模型评分 vs 真实标签 对比（hold-out test）")
    print("=" * 72)

    for doc_id in sorted(test_doc_ids):
        score_path = score_root / doc_id / "topology_scores.json"
        label_path = label_root / doc_id / "topo_labels.json"
        if not score_path.exists() or not label_path.exists():
            continue

        with open(score_path, encoding="utf-8") as f:
            score_data = json.load(f)
        scores: dict = score_data.get("scores", {})

        with open(label_path, encoding="utf-8") as f:
            lbl = json.load(f)
        rank_labels = lbl.get("rank_labels", {})
        removal_sets = [s for s in lbl.get("removal_sets", []) if s]

        model_rank = _model_rank_no_tie(scores)
        raw_scores = {nid: info["score"] for nid, info in scores.items()}
        best_rate, best_set = _best_set_hit(removal_sets, model_rank)
        hit_str = f"命中 {best_rate*100:.0f}%" if best_rate == 1.0 else f"部分命中 {best_rate*100:.0f}%"

        true_sorted = sorted(rank_labels.items(), key=lambda x: (-x[1], x[0]))
        true_rank = {nid: i + 1 for i, (nid, _) in enumerate(true_sorted)}
        sets_str = "[" + ", ".join("{" + ",".join(s) + "}" for s in removal_sets) + "]"
        sorted_nodes = sorted(model_rank, key=lambda x: model_rank[x])
        n_pred = max(len(best_set), 1)
        pred_set = sorted_nodes[:n_pred]
        pred_str = "{" + ",".join(pred_set) + "}"

        n_nodes = score_data.get("num_nodes", len(scores))
        print(f"\n── {doc_id}  ({n_nodes} 个节点) ──")
        print(f"  关键节点集合: {sets_str}")
        print(f"  预测 Top{n_pred}={pred_str}  [{hit_str}]")

        W = (6, 12, 10, 12, 10)
        header = ("  "
                  + _cell("节点", W[0]) + " "
                  + _cell("真实rank", W[1], ">") + " "
                  + _cell("真实分数", W[2], ">") + " "
                  + _cell("模型rank", W[3], ">") + " "
                  + _cell("模型分数", W[4], ">"))
        print(header)
        print("  " + "-" * (sum(W) + len(W) - 1))

        for nid, _ in true_sorted:
            ms = raw_scores.get(nid, 0.0)
            mr = model_rank.get(nid, "?")
            tr = true_rank.get(nid, "?")
            trl = rank_labels.get(nid, 0.0)
            row = ("  "
                   + _cell(nid, W[0]) + " "
                   + _cell(tr, W[1], ">") + " "
                   + _cell(f"{trl:.4f}", W[2], ">") + " "
                   + _cell(mr, W[3], ">") + " "
                   + _cell(f"{ms:.4f}", W[4], ">"))
            print(row)

    print("\n" + "=" * 72)


def _print_inference_only(
    score_root: Path,
    inference_doc_ids: set[str],
) -> None:
    """>30 节点图：无真实标签，仅打印模型分数排名。"""
    if not inference_doc_ids:
        return

    print("\n" + "=" * 72)
    print("  大图画推理分数（>30 节点，无真实标签）")
    print("=" * 72)

    W = (6, 12, 10)
    for doc_id in sorted(inference_doc_ids):
        score_path = score_root / doc_id / "topology_scores.json"
        if not score_path.exists():
            continue

        with open(score_path, encoding="utf-8") as f:
            score_data = json.load(f)
        scores: dict = score_data.get("scores", {})
        n_nodes = score_data.get("num_nodes", len(scores))

        print(f"\n── {doc_id}  ({n_nodes} 个节点) ──")
        header = ("  "
                  + _cell("节点", W[0]) + " "
                  + _cell("模型rank", W[1], ">") + " "
                  + _cell("模型分数", W[2], ">"))
        print(header)
        print("  " + "-" * (sum(W) + len(W) - 1))

        ranked = sorted(scores.items(), key=lambda x: (-x[1]["score"], x[0]))
        for rank, (nid, info) in enumerate(ranked, start=1):
            row = ("  "
                   + _cell(nid, W[0]) + " "
                   + _cell(rank, W[1], ">") + " "
                   + _cell(f"{info['score']:.4f}", W[2], ">"))
            print(row)

    print("\n" + "=" * 72)


def _print_metrics(
    score_root: Path,
    label_root: Path,
    test_doc_ids: set[str],
) -> None:
    """汇总 hold-out test 集评估指标。"""
    total = full_hit = partial_hit = full_miss = 0
    miss_docs: list[str] = []

    for doc_id in sorted(test_doc_ids):
        score_path = score_root / doc_id / "topology_scores.json"
        label_path = label_root / doc_id / "topo_labels.json"
        if not (score_path.exists() and label_path.exists()):
            continue

        sd = json.loads(score_path.read_text(encoding="utf-8"))
        ld = json.loads(label_path.read_text(encoding="utf-8"))
        removal_sets = [s for s in ld.get("removal_sets", []) if s]
        if not removal_sets:
            continue

        scores = sd.get("scores", {})
        model_rank = _model_rank_no_tie(scores)
        best_rate, _ = _best_set_hit(removal_sets, model_rank)

        total += 1
        if best_rate == 1.0:
            full_hit += 1
        elif best_rate > 0.0:
            partial_hit += 1
        else:
            full_miss += 1
            miss_docs.append(doc_id)

    if total == 0:
        print("\n[ 无 test 集可评估文档 ]")
        return

    W = 52
    print("\n" + "=" * W)
    print("  评估指标汇总（hold-out test）")
    print("=" * W)
    print(f"  文档总数     : {total}")
    print(f"  完全命中     : {full_hit:>3}/{total} = {full_hit/total*100:5.1f}%")
    print(f"  部分命中     : {partial_hit:>3}/{total} = {partial_hit/total*100:5.1f}%")
    print(f"  完全未命中   : {full_miss:>3}/{total} = {full_miss/total*100:5.1f}%")
    if miss_docs:
        print(f"  未命中文档   : {miss_docs}")
    print("=" * W)


def _print_metrics_all(
    score_root: Path,
    label_root: Path,
    all_doc_ids: set[str],
    label: str = "全量（train + test）",
) -> None:
    """对所有有标签文档（train+test）汇总评估指标，不打印文档名。"""
    total = full_hit = partial_hit = full_miss = 0

    for doc_id in sorted(all_doc_ids):
        score_path = score_root / doc_id / "topology_scores.json"
        label_path = label_root / doc_id / "topo_labels.json"
        if not (score_path.exists() and label_path.exists()):
            continue

        sd = json.loads(score_path.read_text(encoding="utf-8"))
        ld = json.loads(label_path.read_text(encoding="utf-8"))
        removal_sets = [s for s in ld.get("removal_sets", []) if s]
        if not removal_sets:
            continue

        scores = sd.get("scores", {})
        model_rank = _model_rank_no_tie(scores)
        best_rate, _ = _best_set_hit(removal_sets, model_rank)

        total += 1
        if best_rate == 1.0:
            full_hit += 1
        elif best_rate > 0.0:
            partial_hit += 1
        else:
            full_miss += 1

    if total == 0:
        print(f"\n[ 无可评估文档（{label}）]")
        return

    W = 52
    print("\n" + "=" * W)
    print(f"  评估指标汇总（{label}）")
    print("=" * W)
    print(f"  文档总数     : {total}")
    print(f"  完全命中     : {full_hit:>3}/{total} = {full_hit/total*100:5.1f}%")
    print(f"  部分命中     : {partial_hit:>3}/{total} = {partial_hit/total*100:5.1f}%")
    print(f"  完全未命中   : {full_miss:>3}/{total} = {full_miss/total*100:5.1f}%")
    print("=" * W)


def main() -> None:
    parser = argparse.ArgumentParser(description="Topo_model 完整流水线")
    parser.add_argument("--skip-train", action="store_true",
                        help="跳过训练，使用已有模型推理")
    parser.add_argument("--regenerate-labels", action="store_true",
                        help="强制重算 topo_labels（schema v2），忽略已有标签断点")
    parser.add_argument("--regenerate-features", action="store_true",
                        help="强制重算节点特征（features.npy），忽略已有特征缓存")
    parser.add_argument("--device", default="cpu", help="推理/训练设备")
    args = parser.parse_args()

    topo_cfg  = TopoConfig(num_workers=1)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    logger_feat  = _make_logger("feat")
    logger_gt    = _make_logger("gt")
    logger_data  = _setup_data_logging(LOG_ROOT, "dataset")
    logger_train = _setup_train_logging(LOG_ROOT)

    # ── [1/5] 真实标签 + 划分清单 ─────────────────────────────────────────────
    print("\n[1/5] 真实标签生成 + 数据划分...")
    gt_process_all(
        GRAPH_ROOT, LABEL_ROOT, topo_cfg, logger_gt,
        resume=not args.regenerate_labels,
        manifest_path=MANIFEST_PATH, train_cfg=train_cfg,
    )

    if not MANIFEST_PATH.exists():
        print(f"[!] 未找到划分清单 {MANIFEST_PATH}")
        sys.exit(1)
    manifest = load_manifest(MANIFEST_PATH)
    labeled_ids = set(get_doc_ids(manifest, split="labeled"))
    train_ids   = set(get_doc_ids(manifest, split="train"))
    test_ids    = set(get_doc_ids(manifest, split="test"))
    score_ids   = set(get_doc_ids(manifest, split="score"))
    inference_ids = set(get_doc_ids(manifest, split="inference_only"))

    # ── [2/5] 特征工程（仅 labeled）──────────────────────────────────────────
    print("\n[2/5] 特征工程（labeled 图）...")
    feat_process_all(GRAPH_ROOT, FEAT_ROOT, topo_cfg, logger_feat,
                     resume=not args.regenerate_features, doc_ids=labeled_ids)

    # ── [3/5] 训练（仅 train）──────────────────────────────────────────────────
    final_model_dir = MODEL_ROOT / "final"
    has_model = (final_model_dir / "best_weights.pth").exists() or \
                (final_model_dir / "latest_weights.pth").exists()

    trainer = None
    if args.skip_train:
        if not has_model:
            print("\n[!] --skip-train 但 output/models/final 无权重，请先完整运行。")
            sys.exit(1)
        print("\n[3/5] 跳过训练（使用已有模型）。")
    else:
        print("\n[3/5] 训练模型...")
        train_dataset = TopoDataset.build(
            FEAT_ROOT, LABEL_ROOT, GRAPH_ROOT, topo_cfg, logger_data,
            manifest_path=MANIFEST_PATH, split="train",
        )
        test_dataset = TopoDataset.build(
            FEAT_ROOT, LABEL_ROOT, GRAPH_ROOT, topo_cfg, logger_data,
            manifest_path=MANIFEST_PATH, split="test",
        )
        n_train, n_test = len(train_dataset), len(test_dataset)
        n_folds = train_cfg.n_folds
        fold_val = n_train // n_folds if n_folds else 0
        fold_train = n_train - fold_val
        print(
            f"  数据划分（manifest）: train={n_train}  hold-out test={n_test}  "
            f"（共 {n_train + n_test} 张有标签图）"
        )
        print(
            f"  K 折（仅在 train 的 {n_train} 张内）: 每折约 train={fold_train}  "
            f"val={fold_val}（{n_folds} 折，各折 val 子集不同）"
        )
        print(
            f"  最终模型: 用全部 train={n_train} 张训练；"
            f"hold-out test={n_test} 张仅用于最终评估"
        )
        run_kfold(train_dataset, model_cfg, train_cfg, MODEL_ROOT, logger_train,
                  label_root=LABEL_ROOT)
        trainer = run_final(train_dataset, model_cfg, train_cfg, MODEL_ROOT, logger_train)
        eval_on_holdout_test(trainer, test_dataset, LABEL_ROOT, logger_train)

    # ── [4/5] 推理（≥3 节点全部图）────────────────────────────────────────────
    print("\n[4/5] 推理评分...")
    score_all(
        graph_root=GRAPH_ROOT,
        checkpoint_path=final_model_dir,
        output_root=SCORE_ROOT,
        device=args.device,
        logger=_make_logger("score"),
        doc_ids=score_ids,
    )

    # ── [5/5] 打印结果 ─────────────────────────────────────────────────────────
    _print_comparison(SCORE_ROOT, LABEL_ROOT, test_ids)
    _print_inference_only(SCORE_ROOT, inference_ids)
    _print_metrics(SCORE_ROOT, LABEL_ROOT, test_ids)
    _print_metrics_all(SCORE_ROOT, LABEL_ROOT, labeled_ids)


if __name__ == "__main__":
    import re as _re
    _ANSI_RE = _re.compile(r"\x1b\[[0-9;]*[mGKHF]")

    _result_path = _BASE / "output" / "models" / "result.txt"
    _result_path.parent.mkdir(parents=True, exist_ok=True)

    class _Tee:
        def __init__(self, terminal, logfile):
            self._term = terminal
            self._file = logfile
        def write(self, data):
            self._term.write(data)
            self._file.write(_ANSI_RE.sub("", data))
        def flush(self):
            self._term.flush()
            self._file.flush()
        def isatty(self):
            return hasattr(self._term, "isatty") and self._term.isatty()

    with open(_result_path, "w", encoding="utf-8") as _f:
        _orig = sys.stdout
        sys.stdout = _Tee(_orig, _f)
        try:
            main()
        finally:
            sys.stdout = _orig
