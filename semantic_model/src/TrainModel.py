from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm

from config import (
    AugmentorConfig, HetGATConfig, VICRegConfig, TrainConfig, ScorerConfig,
)
from Augmentor import Augmentor
from GCLModel import GCLModel

_PROJECT_ROOT = Path(__file__).parent.parent


# ── 工具 ───────────────────────────────────────────────────────────────────────

def _setup_train_logging(log_root: Path, name: str) -> logging.Logger:
    """创建同时写文件和控制台的 logger。"""
    log_root.mkdir(parents=True, exist_ok=True)
    _log = logging.getLogger(name)
    if _log.handlers:
        return _log
    _log.setLevel(logging.DEBUG)
    _log.propagate = False
    fh = logging.FileHandler(log_root / f"{name}.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    _log.addHandler(fh)
    _log.addHandler(ch)
    return _log


# ── Trainer ────────────────────────────────────────────────────────────────────

class Trainer:
    """
    单折训练器：负责 VICReg 双视角图对比学习的完整训练循环。

    保存策略
    --------
    latest_weights.pth  : 每 epoch 更新（断点续训用，含完整模型权重）
    best_weights.pth    : val_loss 改善时更新，仅保存 encoder 权重
                          （SemanticScorer 只需要 encoder）
    无验证集时（run_final）: 最后一个 checkpoint epoch 保存为 best。

    评估指标
    --------
    val_loss : 验证集 VICReg loss，模型选择主标准
    """

    def __init__(
        self,
        train_data:  list,
        val_data:    list,
        in_dim:      int,
        hetgat_cfg:  HetGATConfig | None  = None,
        vicreg_cfg:  VICRegConfig | None  = None,
        train_cfg:   TrainConfig | None   = None,
        aug_cfg:     AugmentorConfig | None = None,
        scorer_cfg:  ScorerConfig | None  = None,
        output_dir:  str | Path = _PROJECT_ROOT / "output" / "models",
        log_root:    Path = _PROJECT_ROOT / "log",
        fold_id:     str = "final",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_data  = train_data
        self.val_data    = val_data
        self.train_cfg   = train_cfg or TrainConfig()
        self.scorer_cfg  = scorer_cfg or ScorerConfig()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = _setup_train_logging(log_root, f"train_{fold_id}")
        self.logger.info("使用设备: %s", self.device)

        self.model = GCLModel(
            in_dim,
            hetgat_cfg or HetGATConfig(),
            vicreg_cfg or VICRegConfig(),
        ).to(self.device)

        self.augmentor = Augmentor(aug_cfg or AugmentorConfig())

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )

        self.best_val_loss = float("inf")

        # checkpoint 路径
        self.latest_weights = self.output_dir / "latest_weights.pth"
        self.latest_optim   = self.output_dir / "latest_optim.pth"
        self.latest_meta    = self.output_dir / "latest_meta.json"
        self.best_weights   = self.output_dir / "best_weights.pth"
        self.best_meta      = self.output_dir / "best_meta.json"

    # ── checkpoint ────────────────────────────────────────────────────────────

    def load_checkpoint(self) -> int:
        """加载最新 checkpoint，返回起始 epoch（0 = 从头训练）。"""
        if not (self.latest_weights.exists() and self.latest_meta.exists()):
            return 0
        self.model.load_state_dict(
            torch.load(self.latest_weights, map_location=self.device, weights_only=True)
        )
        if self.latest_optim.exists():
            self.optimizer.load_state_dict(
                torch.load(self.latest_optim, map_location=self.device, weights_only=True)
            )
        with open(self.latest_meta, encoding="utf-8") as f:
            meta = json.load(f)
        self.best_val_loss = float(meta.get("best_val_loss", float("inf")))
        start = int(meta.get("epoch", -1)) + 1
        self.logger.info("断点续训：从 epoch %d 恢复，best_val_loss=%.4f", start, self.best_val_loss)
        return start

    def _save_latest(self, epoch: int, train_loss: float) -> None:
        """每 epoch 保存最新权重（用于断点续训）。"""
        torch.save(self.model.state_dict(), self.latest_weights)
        torch.save(self.optimizer.state_dict(), self.latest_optim)
        meta = {
            "epoch":        epoch,
            "best_val_loss": self.best_val_loss,
            "train_loss":    train_loss,
        }
        with open(self.latest_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _save_best(self, epoch: int, val_loss: float) -> None:
        """保存 best checkpoint，仅含 encoder 权重（SemanticScorer 使用）。"""
        torch.save(
            {"encoder": self.model.encoder.state_dict()},
            self.best_weights,
        )
        meta = {
            "epoch":         epoch,
            "val_loss":      val_loss,
            "best_val_loss": self.best_val_loss,
        }
        with open(self.best_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        self.logger.info(
            "Best model 更新 (epoch %d  val_loss=%.4f)",
            epoch + 1, val_loss,
        )

    # ── 评估 ─────────────────────────────────────────────────────────────────

    def _evaluate(self) -> float:
        """在验证集上计算 val_loss。val_data 为空时返回 nan。"""
        if not self.val_data:
            return float("nan")

        self.model.eval()
        with torch.no_grad():
            val_loss_t, _ = self.model.compute_loss(self.val_data, self.augmentor)
        return val_loss_t.item()

    # ── 训练主循环 ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        完整训练循环，支持断点续训。

        Returns
        -------
        {"history": List[dict], "best_val_loss": float}
        """
        cfg   = self.train_cfg
        start = self.load_checkpoint()
        history: list[dict] = []

        with tqdm(
            range(start, cfg.epochs),
            desc="Epochs", unit="ep", dynamic_ncols=True,
        ) as pbar:
            for epoch in pbar:
                # ── 训练步 ────────────────────────────────────────────────────
                self.model.train()
                loss, metrics = self.model.compute_loss(self.train_data, self.augmentor)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss = loss.item()

                # 每 epoch 保存 latest（保障随时可恢复）
                self._save_latest(epoch, train_loss)

                # ── 评估 + best checkpoint ────────────────────────────────────
                is_checkpoint = (
                    (epoch + 1) % cfg.checkpoint_interval == 0
                    or epoch == cfg.epochs - 1
                )
                if is_checkpoint:
                    val_loss = self._evaluate()

                    # 无验证集：最后一个 checkpoint epoch 视为 best
                    if not self.val_data:
                        is_best = (epoch == cfg.epochs - 1)
                    else:
                        is_best = (
                            not np.isnan(val_loss)
                            and val_loss < self.best_val_loss
                        )
                        if is_best:
                            self.best_val_loss = val_loss

                    if is_best:
                        self._save_best(epoch, val_loss)

                    record = {
                        "epoch":      epoch + 1,
                        "train_loss": train_loss,
                        "val_loss":   val_loss,
                        **metrics,
                    }
                    history.append(record)

                    val_str = f"{val_loss:.4f}" if not np.isnan(val_loss) else "N/A"
                    self.logger.info(
                        "Epoch %4d  train=%.4f  val=%s",
                        epoch + 1, train_loss, val_str,
                    )
                    pbar.set_postfix(
                        train=f"{train_loss:.4f}",
                        val=val_str,
                    )

        # 保存训练历史
        with open(self.output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        return {"history": history, "best_val_loss": self.best_val_loss}


# ── K-Fold 交叉验证 ────────────────────────────────────────────────────────────

def run_kfold(
    dataset,
    in_dim:      int = 768,
    hetgat_cfg:  HetGATConfig | None  = None,
    vicreg_cfg:  VICRegConfig | None  = None,
    train_cfg:   TrainConfig | None   = None,
    aug_cfg:     AugmentorConfig | None = None,
    scorer_cfg:  ScorerConfig | None  = None,
    output_root: Path = _PROJECT_ROOT / "output" / "models",
    log_root:    Path = _PROJECT_ROOT / "log",
) -> List[float]:
    """
    K-Fold 交叉验证。

    Returns
    -------
    List[float]  各折的 best_val_loss
    """
    train_cfg = train_cfg or TrainConfig()
    kf = KFold(
        n_splits=train_cfg.n_folds,
        shuffle=True,
        random_state=train_cfg.random_seed,
    )
    data_list = list(dataset)
    fold_losses: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data_list)):
        print(f"\n── Fold {fold_idx + 1}/{train_cfg.n_folds} ──")
        train_data = [data_list[i] for i in train_idx]
        val_data   = [data_list[i] for i in val_idx]

        trainer = Trainer(
            train_data=train_data,
            val_data=val_data,
            in_dim=in_dim,
            hetgat_cfg=hetgat_cfg,
            vicreg_cfg=vicreg_cfg,
            train_cfg=train_cfg,
            aug_cfg=aug_cfg,
            scorer_cfg=scorer_cfg,
            output_dir=output_root / f"fold_{fold_idx}",
            log_root=log_root,
            fold_id=str(fold_idx),
        )
        result = trainer.run()
        fold_losses.append(result["best_val_loss"])
        print(f"Fold {fold_idx + 1}  best_val_loss = {result['best_val_loss']:.4f}")

    mean_loss = float(np.mean([v for v in fold_losses if not np.isnan(v)]))
    print(f"\nK-Fold 完成  mean_val_loss = {mean_loss:.4f}")
    return fold_losses


# ── 全量最终训练 ────────────────────────────────────────────────────────────────

def run_final(
    dataset,
    in_dim:      int = 768,
    hetgat_cfg:  HetGATConfig | None  = None,
    vicreg_cfg:  VICRegConfig | None  = None,
    train_cfg:   TrainConfig | None   = None,
    aug_cfg:     AugmentorConfig | None = None,
    scorer_cfg:  ScorerConfig | None  = None,
    output_root: Path = _PROJECT_ROOT / "output" / "models",
    log_root:    Path = _PROJECT_ROOT / "log",
) -> None:
    """在全量数据上训练最终模型（无验证集），供部署使用。"""
    print("\n── 全量最终训练 ──")
    trainer = Trainer(
        train_data=list(dataset),
        val_data=[],    # 无验证集，最后一个 checkpoint epoch 自动设为 best
        in_dim=in_dim,
        hetgat_cfg=hetgat_cfg,
        vicreg_cfg=vicreg_cfg,
        train_cfg=train_cfg,
        aug_cfg=aug_cfg,
        scorer_cfg=scorer_cfg,
        output_dir=output_root / "final",
        log_root=log_root,
        fold_id="final",
    )
    trainer.run()


# ── 独立测试入口 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from config import EncoderConfig, HetGATConfig, VICRegConfig, TrainConfig, AugmentorConfig
    from TextEncoder import TextEncoder
    from GraphDataset import GraphDataset

    _ROOT = Path(__file__).parent.parent
    graph_dir = _ROOT / "data" / "output" / "doc_results"

    print("[1] 加载数据...")
    enc = TextEncoder(EncoderConfig())
    embeddings = enc.encode_all(graph_dir)
    from GraphDataset import GraphDataset
    dataset = GraphDataset(graph_dir, embeddings)

    # 快速验证：1-fold, 2 epochs
    fast_cfg = TrainConfig(n_folds=2, epochs=2, checkpoint_interval=1)

    print("[2] 跑 2-fold × 2 epoch 验证训练循环...")
    fold_losses = run_kfold(
        dataset,
        in_dim=768,
        train_cfg=fast_cfg,
        output_root=_ROOT / "output" / "models_test",
        log_root=_ROOT / "log",
    )
    print(f"\nfold_losses = {fold_losses}")
    print("✓ TrainModel 测试通过")
