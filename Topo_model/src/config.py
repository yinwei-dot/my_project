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
    scorer_output_norm: str = "none"


@dataclass
class TrainConfig:
    # 有标签图（3–30 节点）中划入测试集的比例
    test_ratio: float = 0.2
    # K 折交叉验证折数
    n_folds: int = 2
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

    # ── 早停（val_loss 连续无改善则停训）────────────────────────────────────
    # 连续 early_stop_patience 个 epoch val_loss 不下降时终止训练
    early_stop_patience: int = 24

    # ── 实验追踪（SwanLab）──────────────────────────────────────────────────
    use_swanlab: bool = True
    swanlab_project: str = "topo_model"

    # ── 多 seed 稳定性验证 ───────────────────────────────────────────────────
    # 跑 n_seeds 组 K-Fold，每组用不同随机种子划分，汇总均值±标准差
    n_seeds: int = 1
    seed_list: list[int] = field(default_factory=lambda: [42, 123, 2024])

    # ── 二分类阈值 ────────────────────────────────────────────────────────────
    # 预测分 > member_threshold 视为 D 节点（拆解集成员）
    member_threshold: float = 0.7
