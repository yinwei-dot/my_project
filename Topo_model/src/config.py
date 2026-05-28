from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class TopoConfig:
    # 可达对拆解阈值：移除节点后可达对数降到原来 theta 以下即满足条件
    theta: float = 0.5

    # 节点数 < min_nodes 的图全程跳过（不标签、不训练、不测试）
    min_nodes: int = 3

    # 仅对 [min_nodes, labeled_max_nodes] 生成真实标签；超出则只推理
    labeled_max_nodes: int = 30

    # 并行进程数，0 = 自动（CPU 核心数）；1 = 串行
    num_workers: int = 0

    def resolved_workers(self) -> int:
        """0 自动取 CPU 核数，最小为 1。"""
        return max(1, self.num_workers or os.cpu_count() or 1)


@dataclass
class ModelConfig:
    # ── LocalFeatureEncoder（MLP）参数 ──────────────────────────────────────
    # 输入特征维度（与 FeatureEngineering.py 的 FEATURE_NAMES 数量对应）
    feat_dim: int = 11
    # MLP 维度列表：最后一个元素为输出维度（GAT 输入维度），其余为隐藏层
    # [1]：无隐藏层，feat_dim → 1，与 NIRM 的 lin1 = Linear(5, 1) 一致
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [16, 1])
    # MLP Dropout 比率（无隐藏层时不使用）
    mlp_dropout: float = 0.1
    # LocalFeatureEncoder 输出激活：'none' | 'relu' | 'leaky_relu' | 'sigmoid'
    # 'none'：仅 Linear，无激活（与 NIRM init_score = lin1(x) 一致）
    feature_out_activation: str = "leaky_relu"
    # LeakyReLU 负斜率（仅 feature_out_activation='leaky_relu' 时生效）
    feature_out_leaky_slope: float = 0.1

    # ── DirectedGATEncoder 参数 ─────────────────────────────────────────────
    # GAT 层数
    gat_num_layers: int = 3
    # 每层每头的输出维度（长度须等于 gat_num_layers）
    # [4,4,4] × [8,4,2] 头 → 实际维度：1→32→16→8，与论文 NIRM 完全对齐
    gat_hidden_dims: list[int] = field(default_factory=lambda: [4, 4, 4])
    # 每层的注意力头数（长度须等于 gat_num_layers）
    gat_num_heads: list[int] = field(default_factory=lambda: [8, 4, 2])
    # GAT Dropout 比率
    gat_dropout: float = 0.2
    # LeakyReLU 负斜率
    gat_alpha: float = 0.2

    # ── NodeScorer 参数 ──────────────────────────────────────────────────────
    # 度归一化方式：'max'|'log'|'sqrt'|'none'
    deg_norm: str = "max"
    # 融合分后处理：None|'zscore'|'minmax'（仅 scorer_output_norm='sigmoid' 时生效）
    s_dis_norm: str | None = None
    # NodeScorer 最终输出归一化：'sigmoid' | 'minmax' | 'none'
    # 'none'：local + global 原始分，与 NIRM ranking_scores 一致
    scorer_output_norm: str = "minmax"


@dataclass
class TrainConfig:
    # 有标签图（3–30 节点）中划入测试集的比例
    test_ratio: float = 0.2
    # K 折交叉验证折数
    n_folds: int = 4
    # 随机种子（保证 train/test 与 K-Fold 划分可复现）
    random_seed: int = 42
    # 每折最大训练轮数（早停会提前结束）
    epochs: int = 200
    # Adam 学习率（与论文 NIRM 一致）
    lr: float = 1e-3
    # L2 正则化系数（weight_decay）
    weight_decay: float = 1e-4
    # 定期 checkpoint 间隔（epoch 数）
    checkpoint_interval: int = 10

    # ── 学习率衰减（ReduceLROnPlateau）──────────────────────────────────────
    # val loss 连续 lr_patience 个 epoch 无改善时，lr × lr_decay_factor
    lr_decay_factor: float = 0.4
    lr_patience: int = 8

    # ── 早停 / 选模 ─────────────────────────────────────────────────────────
    # 按 val Kendall τ 选 best_weights.pth，并据此早停（τ 无改善的连续 epoch 数）
    # 建议 ≥ lr_patience 的 2–3 倍（τ 比 loss 波动大）
    early_stop_patience: int = 24

    # ── 实验追踪（SwanLab）──────────────────────────────────────────────────
    # 是否启用 SwanLab（需安装：pip install swanlab）
    use_swanlab: bool = True
    # SwanLab 项目名
    swanlab_project: str = "topo_model"

    # ── 损失函数 ─────────────────────────────────────────────────────────────
    # 主损失：'pairwise' | 'mse'（listnet_temperature>0 时优先 ListNet）
    loss_type: str = "mse"
    # Pairwise hinge：对所有 label_i > label_j，最小化 mean(relu(margin - (pred_i - pred_j)))
    pairwise_margin: float = 0.0

    # 加权 MSE（仅 loss_type='mse' 且 label_weight>0 时）
    label_weight: float = 0

    # ListNet 温度；>0 时覆盖 loss_type
    listnet_temperature: float = 0

    # ── 双阶段监督（成员 BCE + 排序 Pairwise）────────────────────────────────
    # Stage1：拆解节点二分类 BCE 权重（纯 MSE 模式下设为 0）
    member_loss_weight: float = 0.0
    # 跨层 hinge：i∈D, j∈ND 时要求 pred_i - pred_j ≥ cross_tier_margin
    cross_tier_margin: float = 0.3
    cross_tier_weight: float = 0.0
    # ND 组内 Pairwise 权重（弱监督）
    within_nd_pair_weight: float = 0.25
