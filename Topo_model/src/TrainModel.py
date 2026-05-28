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


# ── Pairwise / ListNet Loss ───────────────────────────────────────────────────

def _pairwise_rank_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.0,
    pair_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    RankNet 风格 hinge：对每个 label_i > label_j 的合法对，
    loss_ij = relu(margin - (pred_i - pred_j))，可按 pair_weights 加权平均。
    """
    n = preds.numel()
    if n < 2:
        return preds.sum() * 0.0

    diff_label = labels.unsqueeze(0) - labels.unsqueeze(1)
    mask = diff_label > 0
    if not mask.any():
        return preds.sum() * 0.0

    diff_pred = preds.unsqueeze(0) - preds.unsqueeze(1)
    hinge = F.relu(margin - diff_pred)
    if pair_weights is None:
        return hinge[mask].mean()
    w = pair_weights[mask]
    return (hinge[mask] * w).sum() / w.sum().clamp(min=1e-8)


def _cross_tier_pairwise_loss(
    preds: torch.Tensor,
    member: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """i∈D, j∈ND：要求 pred_i > pred_j + margin。"""
    n = preds.numel()
    if n < 2:
        return preds.sum() * 0.0
    mi = member.unsqueeze(0) > 0.5
    mj = member.unsqueeze(1) < 0.5
    mask = mi & mj
    if not mask.any():
        return preds.sum() * 0.0
    diff_pred = preds.unsqueeze(0) - preds.unsqueeze(1)
    return F.relu(margin - diff_pred)[mask].mean()


def _pairwise_pair_weights(
    member: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    """D 内=1，ND 内=within_nd_pair_weight，跨层由 cross_tier 单独处理。"""
    n = member.numel()
    mi = member.unsqueeze(0) > 0.5
    mj = member.unsqueeze(1) > 0.5
    both_d = mi & mj
    both_nd = (~mi) & (~mj)
    w = torch.ones(n, n, device=member.device, dtype=member.dtype)
    w = w.masked_fill(both_nd, cfg.within_nd_pair_weight)
    w = w.masked_fill(both_d, 1.0)
    cross = (mi & (~mj)) | ((~mi) & mj)
    w = w.masked_fill(cross, 0.0)
    return w


def _graph_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    member: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    """成员 BCE + 排序 Pairwise（D/ND 组内加权）+ 跨层 hinge。"""
    if cfg.listnet_temperature:
        rank_loss = _listnet_loss(preds, labels, cfg.listnet_temperature)
    elif cfg.label_weight > 0:
        w = 1.0 + cfg.label_weight * labels
        rank_loss = ((preds - labels).pow(2) * w).mean()
    elif cfg.loss_type == "pairwise":
        pair_w = _pairwise_pair_weights(member, cfg)
        rank_loss = _pairwise_rank_loss(
            preds, labels, cfg.pairwise_margin, pair_weights=pair_w,
        )
    else:
        rank_loss = F.mse_loss(preds, labels)

    mem_loss = F.binary_cross_entropy_with_logits(preds, member)
    cross_loss = _cross_tier_pairwise_loss(
        preds, member, cfg.cross_tier_margin,
    )
    return (
        cfg.member_loss_weight * mem_loss
        + rank_loss
        + cfg.cross_tier_weight * cross_loss
    )


def _listnet_loss(
    preds: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """
    ListNet Loss（单图）。

    将真实标签 softmax(labels / T) 视为目标概率分布，
    最小化其与 log_softmax(preds) 的交叉熵。

    temperature 越小，目标分布越集中于高分节点（T→0 ≈ Top-1 对比）；
    推荐 T=0.2：重点优化 Top 命中，对中间排名容忍。
    """
    p_true    = torch.softmax(labels / temperature, dim=0)  # [N] 目标概率
    log_p_pred = torch.log_softmax(preds, dim=0)             # [N] 预测 log 概率
    return -(p_true * log_p_pred).sum()


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
        self.tau_no_improve_count = 0   # val τ 无改善，用于早停与 best 保存

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
                        "loss_type": train_cfg.loss_type,
                        "pairwise_margin": train_cfg.pairwise_margin,
                        "label_weight": train_cfg.label_weight,
                        "listnet_temperature": train_cfg.listnet_temperature,
                        "member_loss_weight": train_cfg.member_loss_weight,
                        "cross_tier_margin": train_cfg.cross_tier_margin,
                        "cross_tier_weight": train_cfg.cross_tier_weight,
                        "within_nd_pair_weight": train_cfg.within_nd_pair_weight,
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
        self.model.load_state_dict(
            torch.load(self.weights_path, map_location=self.device, weights_only=True)
        )
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
        self.tau_no_improve_count = int(
            meta.get("tau_no_improve_count", meta.get("no_improve_count", 0))
        )
        start = int(meta.get("epoch", -1)) + 1
        self.logger.info(
            "断点续训：从 epoch %d 恢复，best_loss=%.4f best_tau=%.4f",
            start, self.best_val_loss, self.best_val_tau,
        )
        return start

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        *,
        val_tau: float | None = None,
        is_best: bool = False,
    ) -> None:
        """保存 checkpoint（state_dict + JSON 元数据，weights_only 友好）。"""
        torch.save(self.model.state_dict(), self.weights_path)
        torch.save(self.optimizer.state_dict(), self.optim_path)
        torch.save(self.scheduler.state_dict(), self.sched_path)
        meta = {
            "epoch":            int(epoch),
            "feat_dim":         self.feat_dim,
            "best_loss":            float(self.best_val_loss),
            "best_tau":             float(self.best_val_tau),
            "val_loss":             float(val_loss),
            "val_tau":              None if val_tau is None else float(val_tau),
            "tau_no_improve_count": int(self.tau_no_improve_count),
            "feature_out_activation": self.model.feature_encoder.out_activation_name,
            "feature_out_leaky_slope": self.model.feature_encoder.out_leaky_slope,
            "scorer_output_norm": self.model.scorer.scorer_output_norm,
            "loss_type": self.cfg.loss_type,
            "pairwise_margin": self.cfg.pairwise_margin,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if is_best:
            torch.save(self.model.state_dict(), self.best_weights_path)
            with open(self.best_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            tau_msg = f" tau={val_tau:.4f}" if val_tau is not None else ""
            self.logger.info(
                "Best model 更新 (epoch %d  loss=%.4f%s)",
                epoch + 1, val_loss, tau_msg,
            )

    # ── val 集评估（命中率 + Kendall's τ，使用 best 权重）─────────────────────

    def eval_on_val(self, label_root: Path) -> dict:
        """
        用当前折的最优权重在验证集上计算：
          - Best-Set 命中率（top-n 预测是否覆盖某个最小移除集合）
          - 平均 Kendall's τ（模型分数排序与 rank_label 的一致性）

        Returns:
            {"mean_tau", "hit_rate", "n_hit", "n_total"}
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
            self.model.load_state_dict(
                torch.load(best_w, map_location=self.device, weights_only=True)
            )

        self.model.eval()
        n_total = n_hit = 0
        tau_list: list[float] = []

        for batch in self.val_loader:
            item       = batch[0]
            features   = item["features"].to(self.device)
            edge_index = item["edge_index"].to(self.device)
            labels_np  = item["labels"].cpu().numpy()
            node_ids   = item["node_ids"]
            doc_id     = item["doc_id"]

            with torch.no_grad():
                preds = self.model(features, edge_index)
            pred_np = preds.cpu().numpy()

            # Kendall's τ（排序一致性）
            if len(pred_np) >= 2:
                tau_list.append(_kendall_tau(pred_np, labels_np))

            # Best-Set 命中率
            label_path = label_root / doc_id / "topo_labels.json"
            if label_path.exists():
                with open(label_path, encoding="utf-8") as f:
                    lbl = json.load(f)
                removal_sets = [s for s in lbl.get("removal_sets", []) if s]
                if removal_sets:
                    scores      = {nid: float(pred_np[i]) for i, nid in enumerate(node_ids)}
                    sorted_nids = sorted(scores, key=lambda x: -scores[x])
                    best_rate   = max(
                        len(set(sorted_nids[: len(rs)]) & set(rs)) / len(rs)
                        for rs in removal_sets
                    )
                    n_total += 1
                    if best_rate == 1.0:
                        n_hit += 1

        return {
            "mean_tau": float(np.mean(tau_list)) if tau_list else 0.0,
            "hit_rate": n_hit / n_total if n_total > 0 else 0.0,
            "n_hit":    n_hit,
            "n_total":  n_total,
        }

    # ── 单 epoch ──────────────────────────────────────────────────────────────

    def _run_epoch(
        self, loader: DataLoader, *, train: bool
    ) -> tuple[float, float | None]:
        """
        单 epoch 前向（+ 反向）传播。

        Returns:
            (avg_loss, mean_tau)
            train=True  时 mean_tau=None（不计算排序指标）
            train=False 时 mean_tau 为各图 Kendall's τ 的均值
        """
        self.model.train(train)
        total_loss = 0.0
        n_graphs   = 0
        tau_list: list[float] = []

        for batch in loader:
            item       = batch[0]                              # batch_size=1，取第 0 个
            features   = item["features"].to(self.device)     # [N, feat_dim]
            edge_index = item["edge_index"].to(self.device)   # [2, E]
            labels     = item["labels"].to(self.device)       # [N]
            member     = item["member"].to(self.device)       # [N]

            if train:
                self.optimizer.zero_grad()
                preds = self.model(features, edge_index)       # [N]
                loss = _graph_loss(preds, labels, member, self.cfg)
                loss.backward()
                # 缓存梯度范数（仅训练步，供 SwanLab 读取最后一批均值）
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in self.model.parameters()
                    if p.grad is not None
                ) ** 0.5
                self.model._last_grad_norm = grad_norm
                self.optimizer.step()
            else:
                with torch.no_grad():
                    preds = self.model(features, edge_index)
                    loss = _graph_loss(preds, labels, member, self.cfg)
                # 收集排序一致性
                pred_np  = preds.detach().cpu().numpy()
                label_np = labels.detach().cpu().numpy()
                if len(pred_np) >= 2:
                    tau_list.append(_kendall_tau(pred_np, label_np))

            total_loss += loss.item()
            n_graphs += 1

        avg_loss = total_loss / n_graphs if n_graphs > 0 else 0.0
        mean_tau = float(np.mean(tau_list)) if tau_list else None
        return avg_loss, (None if train else mean_tau)

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
        use_tau_metric = self.val_loader is not None

        with tqdm(
            range(start_epoch, self.cfg.epochs),
            desc="Epochs", unit="ep",
            dynamic_ncols=True,
        ) as pbar:
            for epoch in pbar:
                train_loss, _ = self._run_epoch(self.train_loader, train=True)

                if self.val_loader is not None:
                    val_loss, val_tau = self._run_epoch(self.val_loader, train=False)
                else:
                    val_loss = train_loss
                    val_tau  = None

                if val_tau is not None:
                    self.logger.info(
                        "Epoch %d/%d | train loss=%.4f | val loss=%.4f | val tau=%.4f",
                        epoch + 1, self.cfg.epochs, train_loss, val_loss, val_tau,
                    )
                else:
                    self.logger.info(
                        "Epoch %d/%d | train loss=%.4f | val loss=%.4f",
                        epoch + 1, self.cfg.epochs, train_loss, val_loss,
                    )

                postfix: dict[str, str] = {
                    "tL": f"{train_loss:.4f}",
                    "vL": f"{val_loss:.4f}",
                }
                if val_tau is not None:
                    postfix["vτ"] = f"{val_tau:.3f}"
                pbar.set_postfix(**postfix)

                # best 模型与早停：有验证集时看 τ，否则回退 val loss
                if use_tau_metric and val_tau is not None:
                    is_best = val_tau > self.best_val_tau
                    if is_best:
                        self.best_val_tau         = val_tau
                        self.tau_no_improve_count = 0
                    else:
                        self.tau_no_improve_count += 1
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                else:
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss        = val_loss
                        self.tau_no_improve_count = 0
                    else:
                        self.tau_no_improve_count += 1

                # 学习率衰减仍用 val loss（更平滑）
                self.scheduler.step(val_loss)

                if is_best or (epoch + 1) % self.cfg.checkpoint_interval == 0:
                    self.save_checkpoint(
                        epoch, val_loss, val_tau=val_tau, is_best=is_best
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
                        # Kendall's τ ∈ [-1,1]，衡量模型排序与真实排名的一致性
                        log_dict["val/tau"] = val_tau
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
                if self.tau_no_improve_count >= self.cfg.early_stop_patience:
                    if use_tau_metric:
                        self.logger.info(
                            "早停触发：连续 %d 个 epoch val τ 无改善（best_tau=%.4f），训练终止",
                            self.tau_no_improve_count, self.best_val_tau,
                        )
                    else:
                        self.logger.info(
                            "早停触发：连续 %d 个 epoch val loss 无改善（best_loss=%.4f），训练终止",
                            self.tau_no_improve_count, self.best_val_loss,
                        )
                    break

                history.append({
                    "epoch":      epoch + 1,
                    "train_loss": train_loss,
                    "val_loss":   val_loss,
                    "val_tau":    val_tau,
                })

        # 训练结束强制保存最后一个 checkpoint
        if history:
            self.save_checkpoint(
                self.cfg.epochs - 1,
                history[-1]["val_loss"],
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
            tqdm.write(f"swanlab: \U0001f31f Run `swanlab watch {self._sl_dir}` to view SwanLab Experiment Dashboard")

        return {
            "history":        history,
            "best_val_loss":  self.best_val_loss,
            "best_val_tau":   self.best_val_tau,
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
            feat_dim=model_cfg.feat_dim,
            run_name=f"fold_{fold_idx + 1}",
            swanlab_dir=output_dir / "swanlab",   # 统一写入项目级目录
        )
        result = trainer.run()
        fold_results.append({"fold": fold_idx + 1, **result})

        # 每折评估汇总
        summary_parts = [f"best_loss={result['best_val_loss']:.4f}"]
        if result.get("best_val_tau", float("-inf")) > float("-inf"):
            summary_parts.append(f"best_tau={result['best_val_tau']:.3f}")
        fold_eval: dict = {}
        if label_root is not None:
            fold_eval = trainer.eval_on_val(label_root)
            if fold_eval:
                summary_parts.append(
                    f"hit={fold_eval['n_hit']}/{fold_eval['n_total']}"
                    f"({fold_eval['hit_rate']*100:.0f}%)"
                )
                summary_parts.append(f"tau={fold_eval['mean_tau']:.3f}")
        fold_results[-1]["eval"] = fold_eval
        tqdm.write(
            f"Fold {fold_idx + 1} 完成 | " + " | ".join(summary_parts)
        )

    # 汇总
    losses    = [r["best_val_loss"] for r in fold_results]
    mean_loss = float(np.mean(losses))
    std_loss  = float(np.std(losses))

    summary_line = f"\nK-Fold 完成 | mean_loss={mean_loss:.4f} ± {std_loss:.4f}"
    if label_root is not None:
        evals = [r["eval"] for r in fold_results if r.get("eval")]
        if evals:
            mean_tau_all = float(np.mean([e["mean_tau"] for e in evals]))
            total_hit    = sum(e["n_hit"]   for e in evals)
            total_docs   = sum(e["n_total"] for e in evals)
            hit_rate_all = total_hit / total_docs if total_docs > 0 else 0.0
            summary_line += (
                f" | hit={total_hit}/{total_docs}"
                f"({hit_rate_all*100:.0f}%) | mean_tau={mean_tau_all:.3f}"
            )
    tqdm.write(summary_line)

    logger.info(
        "K-Fold 完成 | loss 各折=%s | mean=%.4f ± %.4f",
        [f"{l:.4f}" for l in losses], mean_loss, std_loss,
    )

    summary_path = output_dir / "kfold_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean_loss": mean_loss,
                "std_loss":  std_loss,
                "folds":     fold_results,
            },
            f, indent=2, ensure_ascii=False,
        )
    logger.info("K-Fold 汇总已保存 → %s", summary_path)
    return fold_results


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
        tqdm.write(
            f"\nHold-out Test（manifest 预留的 {len(test_dataset)} 张，"
            f"未参与 K 折与最终训练）| "
            f"hit={result['n_hit']}/{result['n_total']}"
            f"({result['hit_rate']*100:.0f}%) | mean_tau={result['mean_tau']:.3f}"
        )
        logger.info(
            "Hold-out Test | hit=%d/%d (%.1f%%) | mean_tau=%.3f",
            result["n_hit"], result["n_total"],
            result["hit_rate"] * 100, result["mean_tau"],
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
