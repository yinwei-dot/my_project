from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold

try:
    import swanlab as _swanlab          # type: ignore[import]
    _HAS_SWANLAB = True
except ImportError:
    _HAS_SWANLAB = False
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ModelConfig, TopoConfig, TrainConfig
from Dataset import TopoDataset
from Dataset import _setup_logging as _setup_data_logging
from Model import TopoModel



# ── Kendall's τ (模块级，无 scipy 依赖) ──────────────────────────────────────

def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算两个排名向量的 Kendall's τ（无并列版本，O(n²)，n 通常 < 30）。
    返回值 ∈ [-1, 1]，越接近 1 表示模型排序与真实标签越一致。
    """
    n = len(x)
    if n < 2:
        return 0.0
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            p  = dx * dy
            if p > 0:
                c += 1
            elif p < 0:
                d += 1
    return (c - d) / (n * (n - 1) / 2)


def _graph_loss(rank_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """MSE 损失：直接回归台阶 rank_labels（输出无归一化，绝对值监督）。"""
    return F.mse_loss(rank_scores, labels)


def _fmt_cm(tp: int, fp: int, fn: int, tn: int) -> str:
    """格式化二分类混淆矩阵为可读字符串（D 节点 vs ND 节点）。"""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    w = max(len(str(v)) for v in (tp, fp, fn, tn))
    return (
        f"  混淆矩阵 (列=预测, 行=真实):\n"
        f"  真实D    TP={tp:{w}d}  FN={fn:{w}d}\n"
        f"  真实ND   FP={fp:{w}d}  TN={tn:{w}d}\n"
        f"  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}"
    )


def _load_model_weights(
    model: TopoModel,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[list[str], list[str]]:
    """兼容旧 checkpoint：允许缺少新增的双头参数。"""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    incompatible = model.load_state_dict(state_dict, strict=False)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def _best_set_hit_rate(
    removal_sets: list[list[str]],
    node_ids: list[str],
    pred_scores: np.ndarray,
) -> float:
    if not removal_sets:
        return 0.0
    scores = {nid: float(pred_scores[i]) for i, nid in enumerate(node_ids)}
    sorted_nids = sorted(scores, key=lambda x: -scores[x])
    return max(
        len(set(sorted_nids[: len(rs)]) & set(rs)) / len(rs)
        for rs in removal_sets
        if rs
    )


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    单折 / 全量训练器。

    每个图单独计算 loss（图大小不一，DataLoader batch_size=1）。
    支持断点续训：优先加载 output_dir 下的 latest_weights.pth + latest_meta.json。
    保存策略：以 val loss 下降为标准更新 best_weights.pth；
              val_dataset=None 时（全量训练）每隔 checkpoint_interval 轮保存一次。
    """

    def __init__(
        self,
        model: TopoModel,
        train_dataset: TopoDataset,
        val_dataset: TopoDataset | None,
        train_cfg: TrainConfig,
        output_dir: Path,
        logger: logging.Logger,
        eval_label_root: Path | None = None,
        feat_dim: int = 8,
        run_name: str = "train",
        swanlab_dir: Path | None = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("使用设备: %s", self.device)

        self.model     = model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=train_cfg.lr_decay_factor,
            patience=train_cfg.lr_patience,
        )
        self.cfg       = train_cfg
        self.logger    = logger
        self.output_dir = output_dir
        self.feat_dim   = feat_dim
        self.eval_label_root = eval_label_root
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DataLoader：图大小不一，collate_fn 返回列表，batch_size=1 逐图处理
        self.train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True,
            collate_fn=lambda b: b,
        )
        self.val_loader = (
            DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda b: b)
            if val_dataset is not None else None
        )

        self.best_val_loss        = float("inf")
        self.best_val_tau         = float("-inf")
        self.best_val_hit_rate    = 0.0
        self.monitor_no_improve_count = 0

        # checkpoint 路径（weights-only 友好：state_dict + JSON 元数据分开存）
        self.weights_path      = output_dir / "latest_weights.pth"
        self.optim_path        = output_dir / "latest_optim.pth"
        self.sched_path        = output_dir / "latest_sched.pth"
        self.meta_path         = output_dir / "latest_meta.json"
        self.best_weights_path = output_dir / "best_weights.pth"
        self.best_meta_path    = output_dir / "best_meta.json"

        # ── SwanLab 初始化 ────────────────────────────────────────────────────
        # 统一写入 swanlab_dir（默认 output_dir/swanlab），所有折可共享同一数据库
        _sl_dir = swanlab_dir if swanlab_dir is not None else output_dir / "swanlab"
        self._sl_dir = _sl_dir          # 供 run() 结束时打印正确的 watch 命令
        self._swanlab_on = train_cfg.use_swanlab and _HAS_SWANLAB
        if self._swanlab_on:
            # 清除上一折留下的 SWANLAB_MODE，避免 fold 2+ 出现 "will be overwritten" 警告
            os.environ.pop("SWANLAB_MODE", None)
            # 吞掉 swanlab.init() 自动打印的 banner
            # （路径过长时会在终端换行，且内容与 run() 末尾的 tqdm.write 完全重复）
            import sys as _sys_ref
            _saved_stdout = _sys_ref.stdout
            _saved_stderr = _sys_ref.stderr
            _sl_sink = io.StringIO()
            _sys_ref.stdout = _sl_sink
            _sys_ref.stderr = _sl_sink
            try:
                _swanlab.init(
                    project=train_cfg.swanlab_project,
                    experiment_name=run_name,
                    config={
                        "epochs":       train_cfg.epochs,
                        "lr":           train_cfg.lr,
                        "weight_decay": train_cfg.weight_decay,
                        "feat_dim":     feat_dim,
                        "feature_out_activation": model.feature_encoder.out_activation_name,
                        "feature_out_leaky_slope": model.feature_encoder.out_leaky_slope,
                        "scorer_output_norm": model.scorer.scorer_output_norm,
                        "loss_type":               "mse",
                        "member_threshold":         train_cfg.member_threshold,
                    },
                    logdir=str(_sl_dir),
                    mode="local",
                )
                import time as _time_sl
                _time_sl.sleep(0.3)  # 让后台线程打 banner 进 StringIO sink，避免污染 tqdm
            finally:
                _sys_ref.stdout = _saved_stdout
                _sys_ref.stderr = _saved_stderr
            logger.info("SwanLab 已初始化 (run: %s, logdir: %s)", run_name, _sl_dir)

    # ── checkpoint ────────────────────────────────────────────────────────────

    def load_checkpoint(self) -> int:
        """加载最新 checkpoint，返回起始 epoch（0 表示从头训练）。"""
        if not (self.weights_path.exists() and self.meta_path.exists()):
            return 0
        missing, unexpected = _load_model_weights(
            self.model, self.weights_path, self.device,
        )
        if missing or unexpected:
            self.logger.info(
                "checkpoint 与当前模型结构不完全兼容；按 warm start 处理，"
                "不恢复优化器/调度器/epoch。missing=%s unexpected=%s",
                missing, unexpected,
            )
            self.best_val_loss        = float("inf")
            self.best_val_tau         = float("-inf")
            self.best_val_hit_rate    = 0.0
            self.monitor_no_improve_count = 0
            return 0
        if self.optim_path.exists():
            self.optimizer.load_state_dict(
                torch.load(self.optim_path, map_location=self.device, weights_only=True)
            )
        if self.sched_path.exists():
            self.scheduler.load_state_dict(
                torch.load(self.sched_path, map_location=self.device, weights_only=True)
            )
        with open(self.meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.best_val_loss        = float(meta.get("best_loss", float("inf")))
        self.best_val_tau         = float(meta.get("best_tau", float("-inf")))
        self.best_val_hit_rate    = float(meta.get("best_hit_rate", 0.0))
        self.monitor_no_improve_count = int(
            meta.get("no_improve_count",
                meta.get("monitor_no_improve_count",
                    meta.get("tau_no_improve_count", 0)))
        )
        start = int(meta.get("epoch", -1)) + 1
        self.logger.info(
            "断点续训：从 epoch %d 恢复，best_loss=%.4f best_tau=%.4f best_hit=%.4f",
            start, self.best_val_loss, self.best_val_tau, self.best_val_hit_rate,
        )
        return start

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        *,
        val_metrics: dict | None = None,
        is_best: bool = False,
    ) -> None:
        """保存 checkpoint（state_dict + JSON 元数据，weights_only 友好）。"""
        torch.save(self.model.state_dict(), self.weights_path)
        torch.save(self.optimizer.state_dict(), self.optim_path)
        torch.save(self.scheduler.state_dict(), self.sched_path)
        val_tau      = None if val_metrics is None else val_metrics.get("mean_tau")
        val_hit_rate = None if val_metrics is None else val_metrics.get("hit_rate")
        meta = {
            "epoch":             int(epoch),
            "feat_dim":          self.feat_dim,
            "best_loss":         float(self.best_val_loss),
            "best_tau":          float(self.best_val_tau),
            "best_hit_rate":     float(self.best_val_hit_rate),
            "val_loss":          float(val_loss),
            "val_tau":           None if val_tau is None else float(val_tau),
            "val_hit_rate":      None if val_hit_rate is None else float(val_hit_rate),
            "no_improve_count":  int(self.monitor_no_improve_count),
            "loss_type":         "mse",
            "feature_out_activation":  self.model.feature_encoder.out_activation_name,
            "scorer_output_norm":      self.model.scorer.scorer_output_norm,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if is_best:
            torch.save(self.model.state_dict(), self.best_weights_path)
            with open(self.best_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            extra_parts: list[str] = []
            if val_tau is not None:
                extra_parts.append(f"tau={float(val_tau):.4f}")
            if val_hit_rate is not None:
                extra_parts.append(f"hit={float(val_hit_rate):.4f}")
            suffix = (" | " + " | ".join(extra_parts)) if extra_parts else ""
            self.logger.info(
                "Best model 更新 (epoch %d  loss=%.4f%s)",
                epoch + 1, val_loss, suffix,
            )

    def _evaluate_loader(
        self,
        loader: DataLoader,
        label_root: Path | None,
    ) -> dict:
        self.model.eval()
        total_loss = 0.0
        n_graphs = 0
        tau_list: list[float] = []
        n_total = n_hit = 0
        tp = fp = fn = tn = 0

        for batch in loader:
            item       = batch[0]
            features   = item["features"].to(self.device)
            edge_index = item["edge_index"].to(self.device)
            labels     = item["labels"].to(self.device)
            member     = item["member"].to(self.device)
            node_ids   = item["node_ids"]
            doc_id     = item["doc_id"]

            with torch.no_grad():
                rank_scores = self.model(features, edge_index)
                loss = _graph_loss(rank_scores, labels)

            pred_np   = rank_scores.detach().cpu().numpy()
            label_np  = labels.detach().cpu().numpy()
            member_np = member.detach().cpu().numpy()

            if len(pred_np) >= 2:
                tau_list.append(_kendall_tau(pred_np, label_np))

            # 二分类：预测分 > member_threshold → 预测为D节点
            pred_m = (pred_np > self.cfg.member_threshold).astype(int)
            true_m = (member_np > 0.5).astype(int)
            tp += int(((pred_m == 1) & (true_m == 1)).sum())
            fp += int(((pred_m == 1) & (true_m == 0)).sum())
            fn += int(((pred_m == 0) & (true_m == 1)).sum())
            tn += int(((pred_m == 0) & (true_m == 0)).sum())

            if label_root is not None:
                label_path = label_root / doc_id / "topo_labels.json"
                if label_path.exists():
                    with open(label_path, encoding="utf-8") as f:
                        lbl = json.load(f)
                    removal_sets = [s for s in lbl.get("removal_sets", []) if s]
                    if removal_sets:
                        best_rate = _best_set_hit_rate(removal_sets, node_ids, pred_np)
                        n_total += 1
                        if best_rate == 1.0:
                            n_hit += 1

            total_loss += loss.item()
            n_graphs += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        member_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "loss":             total_loss / n_graphs if n_graphs > 0 else 0.0,
            "mean_tau":         float(np.mean(tau_list)) if tau_list else 0.0,
            "n_hit":            n_hit,
            "n_total":          n_total,
            "hit_rate":         n_hit / n_total if n_total > 0 else 0.0,
            "member_tp":        tp,
            "member_fp":        fp,
            "member_fn":        fn,
            "member_tn":        tn,
            "member_precision": precision,
            "member_recall":    recall,
            "member_f1":        member_f1,
        }

    # ── val 集评估（命中率 + Kendall's τ，使用 best 权重）─────────────────────

    def eval_on_val(self, label_root: Path) -> dict:
        """
        用当前折的最优权重在验证集上计算：
          - Best-Set 命中率（top-n 预测是否覆盖某个最小移除集合）
          - 平均 Kendall's τ（模型分数排序与 rank_label 的一致性）

        Returns:
            {"mean_tau", "hit_rate", ...}
            val_loader 为 None 时返回 {}
        """
        if self.val_loader is None:
            return {}

        # 加载 best 权重（若存在），否则使用 latest
        best_w = (
            self.best_weights_path
            if self.best_weights_path.exists()
            else self.weights_path
        )
        if best_w.exists():
            _load_model_weights(self.model, best_w, self.device)
        return self._evaluate_loader(self.val_loader, label_root)

    # ── 单 epoch ──────────────────────────────────────────────────────────────

    def _run_epoch(
        self, loader: DataLoader, *, train: bool
    ) -> dict:
        """
        单 epoch 前向（+ 反向）传播。

        Returns:
            train=True  时返回 {"loss": avg_loss}
            train=False 时返回包含 loss / tau / hit / member_f1 的指标字典
        """
        if not train:
            return self._evaluate_loader(loader, self.eval_label_root)

        self.model.train(train)
        total_loss = 0.0
        n_graphs   = 0

        for batch in loader:
            item       = batch[0]                              # batch_size=1，取第 0 个
            features   = item["features"].to(self.device)     # [N, feat_dim]
            edge_index = item["edge_index"].to(self.device)   # [2, E]
            labels     = item["labels"].to(self.device)       # [N]
            member     = item["member"].to(self.device)       # [N]

            if train:
                self.optimizer.zero_grad()
                rank_scores = self.model(features, edge_index)
                loss = _graph_loss(rank_scores, labels)
                loss.backward()
                # 缓存梯度范数（仅训练步，供 SwanLab 读取最后一批均值）
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in self.model.parameters()
                    if p.grad is not None
                ) ** 0.5
                self.model._last_grad_norm = grad_norm
                self.optimizer.step()

            total_loss += loss.item()
            n_graphs += 1

        return {"loss": total_loss / n_graphs if n_graphs > 0 else 0.0}

    # ── 完整训练循环 ──────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        完整训练循环，支持断点续训。

        Returns:
            {
              "history"      : list[dict]  每 epoch 的 loss 记录
              "best_val_loss": float
              "best_val_tau":  float
            }
        """
        start_epoch = self.load_checkpoint()
        history: list[dict] = []

        with tqdm(
            range(start_epoch, self.cfg.epochs),
            desc="Epochs", unit="ep",
            dynamic_ncols=True,
        ) as pbar:
            for epoch in pbar:
                train_metrics = self._run_epoch(self.train_loader, train=True)
                train_loss = float(train_metrics["loss"])

                if self.val_loader is not None:
                    val_metrics = self._run_epoch(self.val_loader, train=False)
                else:
                    val_metrics = {"loss": train_loss}

                val_loss = float(val_metrics["loss"])
                val_tau  = val_metrics.get("mean_tau")
                val_hit  = val_metrics.get("hit_rate")
                val_f1   = val_metrics.get("member_f1")

                metric_parts = [
                    f"Epoch {epoch + 1}/{self.cfg.epochs}",
                    f"train loss={train_loss:.4f}",
                    f"val loss={val_loss:.4f}",
                ]
                if val_tau is not None:
                    metric_parts.append(f"val tau={float(val_tau):.4f}")
                if val_hit is not None and val_metrics.get("n_total", 0):
                    metric_parts.append(f"val hit={float(val_hit):.4f}")
                if val_f1 is not None:
                    metric_parts.append(f"val f1={float(val_f1):.4f}")
                self.logger.info(" | ".join(metric_parts))

                postfix: dict[str, str] = {
                    "tL": f"{train_loss:.4f}",
                    "vL": f"{val_loss:.4f}",
                }
                if val_tau is not None:
                    postfix["vτ"] = f"{float(val_tau):.3f}"
                if val_hit is not None and val_metrics.get("n_total", 0):
                    postfix["vHit"] = f"{float(val_hit):.3f}"
                if val_f1 is not None:
                    postfix["vF1"] = f"{float(val_f1):.3f}"
                pbar.set_postfix(**postfix)

                # best 模型与早停： val_loss 下降则更新
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss     = val_loss
                    self.best_val_tau      = float(val_tau) if val_tau is not None else self.best_val_tau
                    self.best_val_hit_rate = float(val_hit) if val_hit is not None else self.best_val_hit_rate
                    self.monitor_no_improve_count = 0
                else:
                    self.monitor_no_improve_count += 1

                # 学习率衰减仍用 val loss（更平滑）
                self.scheduler.step(val_loss)

                if is_best or (epoch + 1) % self.cfg.checkpoint_interval == 0:
                    self.save_checkpoint(
                        epoch, val_loss, val_metrics=val_metrics, is_best=is_best
                    )

                # ── SwanLab 记录 ───────────────────────────────────────────
                current_lr = self.optimizer.param_groups[0]["lr"]
                if self._swanlab_on:
                    log_dict: dict = {
                        "train/loss": train_loss,
                        "val/loss":   val_loss,
                        "train/lr":   current_lr,
                    }
                    if val_tau is not None:
                        log_dict["val/tau"] = float(val_tau)
                    if val_hit is not None and val_metrics.get("n_total", 0):
                        log_dict["val/hit_rate"] = float(val_hit)
                    if val_f1 is not None:
                        log_dict["val/member_f1"] = float(val_f1)
                    if hasattr(self.model, "_last_s_local_abs_mean"):
                        s_local  = self.model._last_s_local_abs_mean
                        s_global = self.model._last_s_global_abs_mean
                        log_dict["debug/s_local_abs_mean"]  = s_local
                        log_dict["debug/s_global_abs_mean"] = s_global
                        # 双分支比值：持续 >> 1 表示全局分退化，持续 << 1 表示局部分退化
                        log_dict["debug/s_ratio"] = s_local / (s_global + 1e-6)
                    if hasattr(self.model, "_last_feature_pre_mean"):
                        log_dict["debug/feature_pre_min"] = self.model._last_feature_pre_min
                        log_dict["debug/feature_pre_max"] = self.model._last_feature_pre_max
                        log_dict["debug/feature_pre_mean"] = self.model._last_feature_pre_mean
                        log_dict["debug/feature_pre_std"] = self.model._last_feature_pre_std
                        log_dict["debug/feature_post_min"] = self.model._last_feature_post_min
                        log_dict["debug/feature_post_max"] = self.model._last_feature_post_max
                        log_dict["debug/feature_post_mean"] = self.model._last_feature_post_mean
                        log_dict["debug/feature_post_std"] = self.model._last_feature_post_std
                        log_dict["debug/feature_zero_ratio"] = self.model._last_feature_zero_ratio
                        log_dict["debug/feature_lt_005_ratio"] = self.model._last_feature_lt_005_ratio
                        log_dict["debug/feature_gt_095_ratio"] = self.model._last_feature_gt_095_ratio
                    if hasattr(self.model, "_last_raw_mean"):
                        log_dict["debug/raw_min"] = self.model._last_raw_min
                        log_dict["debug/raw_max"] = self.model._last_raw_max
                        log_dict["debug/raw_mean"] = self.model._last_raw_mean
                        log_dict["debug/raw_std"] = self.model._last_raw_std
                        log_dict["debug/pred_min"] = self.model._last_pred_min
                        log_dict["debug/pred_max"] = self.model._last_pred_max
                        log_dict["debug/pred_mean"] = self.model._last_pred_mean
                        log_dict["debug/pred_std"] = self.model._last_pred_std
                        log_dict["debug/pred_range"] = self.model._last_pred_range
                        log_dict["debug/s_local_mean"] = self.model._last_s_local_mean
                        log_dict["debug/s_local_std"] = self.model._last_s_local_std
                        log_dict["debug/s_global_mean"] = self.model._last_s_global_mean
                        log_dict["debug/s_global_std"] = self.model._last_s_global_std
                        log_dict["debug/global_proj_weight_norm"] = self.model._last_global_proj_weight_norm
                        log_dict["debug/global_proj_bias"] = self.model._last_global_proj_bias
                    if hasattr(self.model, "_last_grad_norm"):
                        log_dict["debug/grad_norm"] = self.model._last_grad_norm
                    import sys as _sys_sl
                    _sl_old = _sys_sl.stdout
                    _sys_sl.stdout = io.StringIO()
                    try:
                        _swanlab.log(log_dict, step=epoch + 1)
                    finally:
                        _sys_sl.stdout = _sl_old

                # ── 早停（按 val τ；无验证集时按 val loss）────────────────
                if self.monitor_no_improve_count >= self.cfg.early_stop_patience:
                    self.logger.info(
                        "早停触发：连续 %d 个 epoch val loss 无改善（best_loss=%.4f），训练终止",
                        self.monitor_no_improve_count, self.best_val_loss,
                    )
                    break

                history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        **val_metrics,
                    }
                )

        # 训练结束强制保存最后一个 checkpoint
        if history:
            self.save_checkpoint(
                self.cfg.epochs - 1,
                history[-1]["val_loss"],
                val_metrics=history[-1],
            )

        if self._swanlab_on:
            import io as _io
            import sys as _sys
            _old_stdout = _sys.stdout
            _sys.stdout = _io.StringIO()  # 吞掉 swanlab.finish() 内部的 print
            try:
                _swanlab.finish()
            finally:
                _sys.stdout = _old_stdout

        return {
            "history":           history,
            "best_val_loss":     self.best_val_loss,
            "best_val_tau":      self.best_val_tau,
            "best_val_hit_rate": self.best_val_hit_rate,
        }


# ── K 折交叉验证 ──────────────────────────────────────────────────────────────

def run_kfold(
    full_dataset: TopoDataset,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    output_dir: Path,
    logger: logging.Logger,
    label_root: Path | None = None,
) -> list[dict]:
    """
    K 折交叉验证：每折独立初始化模型，训练完保存到 output_dir/fold_{k}/。
    汇总结果保存到 output_dir/kfold_summary.json。

    Args:
        label_root: 若提供，每折训练完后会加载 topo_labels.json 计算
                    Best-Set 命中率和 Kendall's τ，并打印汇总。

    Returns:
        fold_results : 每折结果列表（含 history / best_val_loss）
    """
    n  = len(full_dataset)
    kf = KFold(n_splits=train_cfg.n_folds, shuffle=True, random_state=train_cfg.random_seed)
    fold_results: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(n))):
        logger.info("=" * 60)
        logger.info(
            "Fold %d/%d | train=%d  val=%d",
            fold_idx + 1, train_cfg.n_folds, len(train_idx), len(val_idx),
        )
        tqdm.write(
            f"\n── Fold {fold_idx + 1}/{train_cfg.n_folds}"
            f"  (折内 train={len(train_idx)}  折内 val={len(val_idx)}) ──"
        )

        train_subset = TopoDataset([full_dataset.records[i] for i in train_idx])
        val_subset   = TopoDataset([full_dataset.records[i] for i in val_idx])

        model   = TopoModel(cfg=model_cfg)
        trainer = Trainer(
            model, train_subset, val_subset, train_cfg,
            output_dir / f"fold_{fold_idx + 1}", logger,
            eval_label_root=label_root,
            feat_dim=model_cfg.feat_dim,
            run_name=f"fold_{fold_idx + 1}",
            swanlab_dir=output_dir / "swanlab",   # 统一写入项目级目录
        )
        result = trainer.run()
        fold_results.append({"fold": fold_idx + 1, **result})

        # 每折评估汇总
        summary_parts = [f"best_loss={result['best_val_loss']:.4f}"]
        fold_eval: dict = {}
        if label_root is not None:
            fold_eval = trainer.eval_on_val(label_root)
            if fold_eval:
                summary_parts.append(
                    f"hit={fold_eval['n_hit']}/{fold_eval['n_total']}"
                    f"({fold_eval['hit_rate']*100:.0f}%)"
                )
                summary_parts.append(f"tau={fold_eval['mean_tau']:.3f}")
                if fold_eval.get("member_f1") is not None:
                    summary_parts.append(f"D-f1={fold_eval['member_f1']:.3f}")
        fold_results[-1]["eval"] = fold_eval
        tqdm.write(
            f"Fold {fold_idx + 1} 完成 | " + " | ".join(summary_parts)
        )
        if fold_eval and all(k in fold_eval for k in ("member_tp", "member_fp", "member_fn", "member_tn")):
            tqdm.write(_fmt_cm(
                fold_eval["member_tp"], fold_eval["member_fp"],
                fold_eval["member_fn"], fold_eval["member_tn"],
            ))

    # 汇总
    losses    = [r["best_val_loss"] for r in fold_results]
    mean_loss = float(np.mean(losses))
    std_loss  = float(np.std(losses))

    summary_line = f"\nK-Fold 完成 | mean_loss={mean_loss:.4f} ± {std_loss:.4f}"
    cm_agg: dict = {}
    if label_root is not None:
        evals = [r["eval"] for r in fold_results if r.get("eval")]
        if evals:
            mean_tau_all = float(np.mean([e["mean_tau"] for e in evals if "mean_tau" in e]))
            total_hit    = sum(e["n_hit"]   for e in evals)
            total_docs   = sum(e["n_total"] for e in evals)
            hit_rate_all = total_hit / total_docs if total_docs > 0 else 0.0
            f1_vals = [e["member_f1"] for e in evals if "member_f1" in e]
            mean_f1_all = float(np.mean(f1_vals)) if f1_vals else 0.0
            summary_line += (
                f" | hit={total_hit}/{total_docs}"
                f"({hit_rate_all*100:.0f}%) | mean_tau={mean_tau_all:.3f}"
                f" | mean_f1={mean_f1_all:.3f}"
            )
            # 汇总混淡矩阵
            agg_tp = sum(e.get("member_tp", 0) for e in evals)
            agg_fp = sum(e.get("member_fp", 0) for e in evals)
            agg_fn = sum(e.get("member_fn", 0) for e in evals)
            agg_tn = sum(e.get("member_tn", 0) for e in evals)
            cm_agg = {"tp": agg_tp, "fp": agg_fp, "fn": agg_fn, "tn": agg_tn}
    tqdm.write(summary_line)
    if cm_agg:
        tqdm.write(_fmt_cm(cm_agg["tp"], cm_agg["fp"], cm_agg["fn"], cm_agg["tn"]))

    logger.info(
        "K-Fold 完成 | loss 各折=%s | mean=%.4f ± %.4f",
        [f"{l:.4f}" for l in losses], mean_loss, std_loss,
    )

    summary_path = output_dir / "kfold_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data: dict = {
        "mean_loss": mean_loss,
        "std_loss":  std_loss,
        "folds":     fold_results,
    }
    if cm_agg:
        summary_data["cm_total"] = cm_agg
    if label_root is not None:
        evals_all = [r["eval"] for r in fold_results if r.get("eval")]
        f1_vals_all = [e["member_f1"] for e in evals_all if "member_f1" in e]
        if f1_vals_all:
            summary_data["mean_f1"] = float(np.mean(f1_vals_all))
            summary_data["std_f1"]  = float(np.std(f1_vals_all))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    logger.info("K-Fold 汇总已保存 → %s", summary_path)
    return fold_results


# ── 多 seed 稳定性验证 ────────────────────────────────────────────────────────

def run_multiseed_kfold(
    full_dataset: TopoDataset,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    output_dir: Path,
    logger: logging.Logger,
    label_root: Path | None = None,
) -> dict:
    """
    多 seed K-Fold：对 train_cfg.seed_list 中每个 seed 跑一次 K-Fold，
    汇总 hit_rate / mean_tau / member_f1 的均值 ± 标准差，评估指标稳定性。
    """
    import dataclasses

    _seed_pool = train_cfg.seed_list or [train_cfg.random_seed]
    seeds = _seed_pool[: train_cfg.n_seeds] if train_cfg.n_seeds > 0 else _seed_pool
    all_hit: list[float] = []
    all_tau: list[float] = []
    all_f1:  list[float] = []
    all_tp = all_fp = all_fn = all_tn = 0

    for i, seed in enumerate(seeds):
        seed_cfg = dataclasses.replace(train_cfg, random_seed=seed)
        seed_dir = output_dir / f"seed_{seed}"
        tqdm.write(f"\n{'='*60}\n多seed实验 {i+1}/{len(seeds)} | seed={seed}\n{'='*60}")
        fold_results = run_kfold(
            full_dataset, model_cfg, seed_cfg, seed_dir, logger, label_root=label_root,
        )
        evals = [r.get("eval", {}) for r in fold_results if r.get("eval")]
        if evals:
            total_hit  = sum(e.get("n_hit", 0) for e in evals)
            total_docs = sum(e.get("n_total", 0) for e in evals)
            tau_vals   = [e["mean_tau"] for e in evals if "mean_tau" in e]
            f1_vals    = [e["member_f1"] for e in evals if "member_f1" in e]
            hit_rate   = total_hit / total_docs if total_docs > 0 else 0.0
            mean_tau   = float(np.mean(tau_vals)) if tau_vals else 0.0
            mean_f1    = float(np.mean(f1_vals))  if f1_vals  else 0.0
            all_hit.append(hit_rate)
            all_tau.append(mean_tau)
            all_f1.append(mean_f1)
            all_tp += sum(e.get("member_tp", 0) for e in evals)
            all_fp += sum(e.get("member_fp", 0) for e in evals)
            all_fn += sum(e.get("member_fn", 0) for e in evals)
            all_tn += sum(e.get("member_tn", 0) for e in evals)

    summary: dict = {}
    if all_hit:
        summary = {
            "n_seeds":       len(seeds),
            "seeds":         seeds,
            "hit_rate_mean": float(np.mean(all_hit)),
            "hit_rate_std":  float(np.std(all_hit)),
            "tau_mean":      float(np.mean(all_tau)),
            "tau_std":       float(np.std(all_tau)),
            "f1_mean":       float(np.mean(all_f1)),
            "f1_std":        float(np.std(all_f1)),
            "cm_total":      {"tp": all_tp, "fp": all_fp, "fn": all_fn, "tn": all_tn},
        }
        if len(seeds) > 1:
            tqdm.write(
                f"\n{'='*60}\n多seed汇总 ({len(seeds)} seeds)\n{'='*60}\n"
                f"  hit_rate : {summary['hit_rate_mean']:.3f} ± {summary['hit_rate_std']:.3f}\n"
                f"  mean_tau : {summary['tau_mean']:.3f} ± {summary['tau_std']:.3f}\n"
                f"  D-f1     : {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f}"
            )
            tqdm.write(_fmt_cm(all_tp, all_fp, all_fn, all_tn))
        summary_path = output_dir / "multiseed_summary.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("多seed汇总已保存 → %s", summary_path)
    return summary


# ── 全量训练（最终模型）──────────────────────────────────────────────────────

def run_final(
    train_dataset: TopoDataset,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    output_dir: Path,
    logger: logging.Logger,
) -> Trainer:
    """
    用 train 子集训练最终模型（无验证集），供推理部署。
    最终权重保存到 output_dir/final/。
    """
    n = len(train_dataset)
    logger.info("开始训练最终模型，train 样本 %d 个。", n)
    tqdm.write(f"\n── 训练最终模型（train={n} 个样本）──")

    model   = TopoModel(cfg=model_cfg)
    trainer = Trainer(
        model, train_dataset, None, train_cfg, output_dir / "final", logger,
        feat_dim=model_cfg.feat_dim, run_name="final",
        swanlab_dir=output_dir / "swanlab",
    )
    trainer.run()
    tqdm.write("最终模型训练完成。权重 → output/models/final/latest_weights.pth")
    return trainer


def eval_on_holdout_test(
    trainer: Trainer,
    test_dataset: TopoDataset,
    label_root: Path,
    logger: logging.Logger,
) -> dict:
    """在 hold-out test 集上评估最终模型（命中率 + Kendall's τ）。"""
    if len(test_dataset) == 0:
        return {}

    trainer.val_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=lambda x: x,
    )
    result = trainer.eval_on_val(label_root)
    trainer.val_loader = None

    if result:
        f1_str = f" | D-f1={result['member_f1']:.3f}" if result.get("member_f1") is not None else ""
        tqdm.write(
            f"\nHold-out Test（manifest 预留的 {len(test_dataset)} 张，"
            f"未参与 K 折与最终训练）| "
            f"hit={result['n_hit']}/{result['n_total']}"
            f"({result['hit_rate']*100:.0f}%) | mean_tau={result['mean_tau']:.3f}{f1_str}"
        )
        if all(k in result for k in ("member_tp", "member_fp", "member_fn", "member_tn")):
            tqdm.write(_fmt_cm(
                result["member_tp"], result["member_fp"],
                result["member_fn"], result["member_tn"],
            ))
        logger.info(
            "Hold-out Test | hit=%d/%d (%.1f%%) | mean_tau=%.3f | member_f1=%.3f"
            " | CM: TP=%d FP=%d FN=%d TN=%d",
            result["n_hit"], result["n_total"],
            result["hit_rate"] * 100, result["mean_tau"],
            result.get("member_f1", 0.0),
            result.get("member_tp", 0), result.get("member_fp", 0),
            result.get("member_fn", 0), result.get("member_tn", 0),
        )
    return result


# ── 日志配置 ──────────────────────────────────────────────────────────────────

def _setup_train_logging(log_dir: Path) -> logging.Logger:
    """控制台输出 WARNING+，日志文件记录 INFO+。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"train_{timestamp}.log"

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    logger.addHandler(ch)

    return logger


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]

    topo_cfg  = TopoConfig(num_workers=1)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # 日志
    data_logger  = _setup_data_logging(base / "log", "dataset_train")
    train_logger = _setup_train_logging(base / "log")

    manifest_path = base / "output" / "split_manifest.json"
    label_root = base / "output" / "labels"

    train_dataset = TopoDataset.build(
        base / "output" / "features", label_root,
        base / "data" / "output" / "doc_results",
        topo_cfg, data_logger, manifest_path=manifest_path, split="train",
    )
    test_dataset = TopoDataset.build(
        base / "output" / "features", label_root,
        base / "data" / "output" / "doc_results",
        topo_cfg, data_logger, manifest_path=manifest_path, split="test",
    )
    print(f"train={len(train_dataset)}  test={len(test_dataset)}")

    output_dir = base / "output" / "models"
    run_kfold(train_dataset, model_cfg, train_cfg, output_dir, train_logger,
              label_root=label_root)
    trainer = run_final(train_dataset, model_cfg, train_cfg, output_dir, train_logger)
    eval_on_holdout_test(trainer, test_dataset, label_root, train_logger)
