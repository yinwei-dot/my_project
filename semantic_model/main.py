"""
semantic_model 完整流水线入口
=============================
步骤：
  1. 文本编码（TextEncoder）
  2. 构建图数据集（GraphDataset）
  3. K-Fold 交叉验证（TrainModel.run_kfold）
  4. 全量最终训练（TrainModel.run_final）
  5. 语义评分推理（SemanticScorer.score_all）
  6. 打印语义评分表

用法：
  # 全量运行
  python main.py

  # 已有训练好的模型，跳过训练直接推理
  python main.py --skip-train

  # 跳过 K-Fold，只做全量训练 + 推理
  python main.py --skip-kfold

  # 指定设备
  python main.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from pathlib import Path

# ── 把 src/ 加入 Python 路径 ─────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent          # semantic_model/
_SRC  = _BASE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (
    EncoderConfig, HetGATConfig, VICRegConfig,
    TrainConfig, AugmentorConfig, ScorerConfig,
)
from TextEncoder import TextEncoder
from GraphDataset import GraphDataset
from TrainModel import run_kfold, run_final, _setup_train_logging
from SemanticScorer import SemanticScorer

# ── 路径常量 ─────────────────────────────────────────────────────────────────
GRAPH_ROOT = _BASE / "data"   / "output" / "doc_results"
MODEL_ROOT = _BASE / "output" / "models"
SCORE_ROOT = _BASE / "output" / "semantic_scores"
LOG_ROOT   = _BASE / "log"


# ── 工具 ──────────────────────────────────────────────────────────────────────

def _make_logger(name: str) -> logging.Logger:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    fh = logging.FileHandler(LOG_ROOT / f"{name}.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    lg.addHandler(fh)
    lg.addHandler(ch)
    return lg


def _rw(s: str, width: int) -> str:
    """右对齐到 `width` 显示列（CJK 字符按 2 列计）。"""
    disp = sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)
    return " " * max(0, width - disp) + s


# ── 打印语义评分表 ────────────────────────────────────────────────────────────

def _print_scores(score_root: Path) -> None:
    """打印每个文档的语义评分结果（按评分降序）。"""
    score_dirs = sorted(p for p in score_root.iterdir() if p.is_dir())
    if not score_dirs:
        print("[ 未找到任何语义评分结果 ]")
        return

    sep = "=" * 48
    print("\n" + sep)
    print(_rw("语义评分结果", 24).center(48))
    print(sep)

    for score_dir in score_dirs:
        score_file = score_dir / "semantic_scores.json"
        if not score_file.exists():
            continue
        with open(score_file, encoding="utf-8") as f:
            data = json.load(f)

        doc_id     = data["doc_id"]
        sem_scores: dict[str, float] = data.get("scores", {})
        sem_rank   = data.get("ranking", [])

        print(f"\n── {doc_id}  ({data.get('num_nodes', len(sem_scores))} 个节点) ──")
        header = (
            f"  {_rw('节点', 6)}"
            f"  {_rw('语义分', 8)}"
            f"  {_rw('排名', 4)}"
        )
        print(header)
        print("  " + "-" * 24)

        for rank, nid in enumerate(sem_rank, start=1):
            score = sem_scores.get(nid, 0.0)
            print(f"  {nid:<6}  {score:>8.4f}  {rank:>4}")

    print("\n" + sep)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="semantic_model 完整流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-train",
                        action="store_true",
                        help="跳过训练，直接用已有模型推理")
    parser.add_argument("--skip-kfold",
                        action="store_true",
                        help="跳过 K-Fold，只做全量训练")
    parser.add_argument("--device",
                        default="cpu",
                        help="训练/推理设备（cpu 或 cuda，默认 cpu）")
    args = parser.parse_args()

    # ── 超参配置（修改这里调整超参）────────────────────────────────────────────
    enc_cfg    = EncoderConfig()
    hetgat_cfg = HetGATConfig()
    vicreg_cfg = VICRegConfig()
    train_cfg  = TrainConfig()
    aug_cfg    = AugmentorConfig()
    scorer_cfg = ScorerConfig()

    logger = _make_logger("main")

    # ── 步骤 1：文本编码 ──────────────────────────────────────────────────────
    print("\n[1/4] 文本编码（TextEncoder）...")
    encoder = TextEncoder(enc_cfg)
    embeddings = encoder.encode_all(GRAPH_ROOT)

    # ── 步骤 2：构建图数据集 ──────────────────────────────────────────────────
    print("\n[2/4] 构建图数据集（GraphDataset）...")
    dataset = GraphDataset(GRAPH_ROOT, embeddings)
    print(f"      共 {len(dataset)} 个文档图")

    # ── 步骤 3：训练 ──────────────────────────────────────────────────────────
    final_model_dir = MODEL_ROOT / "final"
    has_model = (final_model_dir / "best_weights.pth").exists()

    if args.skip_train:
        if not has_model:
            print(
                "\n[!] --skip-train 指定，但 output/models/final 下未找到 best_weights.pth，"
                "请先完整运行一次流水线。"
            )
            sys.exit(1)
        print("\n[3/4] 跳过训练（使用已有模型）。")
    else:
        if not args.skip_kfold:
            print(f"\n[3/4] K-Fold 交叉验证（{train_cfg.n_folds} 折）...")
            fold_losses = run_kfold(
                dataset,
                in_dim=768,
                hetgat_cfg=hetgat_cfg,
                vicreg_cfg=vicreg_cfg,
                train_cfg=train_cfg,
                aug_cfg=aug_cfg,
                scorer_cfg=scorer_cfg,
                output_root=MODEL_ROOT,
                log_root=LOG_ROOT,
            )
            logger.info("K-Fold 结果: %s", fold_losses)
        else:
            print("\n[3/4] 跳过 K-Fold，直接全量训练...")

        print("\n      全量最终训练...")
        run_final(
            dataset,
            in_dim=768,
            hetgat_cfg=hetgat_cfg,
            vicreg_cfg=vicreg_cfg,
            train_cfg=train_cfg,
            aug_cfg=aug_cfg,
            scorer_cfg=scorer_cfg,
            output_root=MODEL_ROOT,
            log_root=LOG_ROOT,
        )

    # ── 步骤 4：语义评分推理 ──────────────────────────────────────────────────
    print("\n[4/4] 语义评分推理（SemanticScorer）...")
    scorer = SemanticScorer.from_checkpoint(
        checkpoint_path=final_model_dir,
        in_dim=768,
        hetgat_cfg=hetgat_cfg,
        scorer_cfg=scorer_cfg,
    )
    scorer.score_all(dataset, SCORE_ROOT)
    print(f"      评分结果已写入 {SCORE_ROOT}")

    _print_scores(SCORE_ROOT)


if __name__ == "__main__":
    main()
