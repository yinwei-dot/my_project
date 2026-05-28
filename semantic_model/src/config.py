from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class EncoderConfig:
    # 文本编码模型；切换模型只改此行
    # model_name: str = "hfl/chinese-roberta-wwm-ext"
    model_name: str = "BAAI/bge-base-zh-v1.5"

    # tokenizer 截断长度
    max_length: int = 256

    # GPU forward 分批大小
    batch_size: int = 16

    # 并发线程数，0 = 自动取 CPU 核数
    num_workers: int = 0

    def resolved_workers(self) -> int:
        return max(1, self.num_workers or os.cpu_count() or 1)


@dataclass
class HetGATConfig:
    # ── 层结构（两个等长列表隐式定义层数）─────────────────────────────────────
    # hidden_dims[i]：第 i 层每个注意力头的输出维度
    #   - 中间层（concat=True）：实际输出 = hidden_dims[i] × num_heads[i]
    #   - 最后一层（concat=False，mean）：实际输出 = hidden_dims[-1]
    # 新增一层只需在两个列表末尾各追加一个值，len(hidden_dims) 即层数
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])

    # num_heads[i]：第 i 层的注意力头数，必须与 hidden_dims 等长
    num_heads: List[int] = field(default_factory=lambda: [4, 4])

    # ── 正则化 ────────────────────────────────────────────────────────────────
    # 注意力权重的 dropout 概率（训练时生效，eval 时自动关闭）
    dropout: float = 0.3

    # ── 语义注意力开关 ────────────────────────────────────────────────────────
    # True：为每个节点学习 4 种关系通道各自的重要性权重（semantic attention）
    # False：4 种关系通道等权相加（更快，但不可解释）
    use_semantic_att: bool = True

    # ── ProjectionHead 输出维度 ────────────────────────────────────────────────
    # VICReg 投影空间维度；训练后投影头丢弃，不影响节点嵌入维度
    proj_dim: int = 128


@dataclass
class AugmentorConfig:
    # ── View 1（语义鲁棒性）增强参数 ──────────────────────────────────────────
    # 特征随机掩码率（置零的维度比例）
    feat_mask_rate_v1: float = 0.15
    # 时序边 dropout 概率
    temporal_drop_v1: float = 0.20
    # 因果边 dropout 概率
    causal_drop_v1: float = 0.20

    # ── View 2（因果偏置）增强参数 ────────────────────────────────────────────
    # 不掩码特征；时序边大比例丢弃，因果边轻度保留
    temporal_drop_v2: float = 0.60
    causal_drop_v2: float = 0.10


@dataclass
class VICRegConfig:
    # ── VICReg 三项权重（原论文推荐起点）────────────────────────────────────
    lambda_: float = 25.0   # 不变性项（Invariance）权重
    mu:      float = 25.0   # 方差项（Variance）权重
    nu:      float = 1.0    # 协方差项（Covariance）权重

    # 方差项目标阈值：每维标准差不低于 gamma
    gamma: float = 1.0

    # L_causal 相对 L_robust 的比例系数
    beta: float = 0.5


@dataclass
class TrainConfig:
    # K 折交叉验证折数（与 Topo_model 对齐）
    n_folds: int = 4
    # 随机种子（保证 KFold 划分可复现）
    random_seed: int = 42
    # 每折训练轮数
    epochs: int = 20
    # Adam 学习率
    lr: float = 1e-3
    # L2 正则化系数
    weight_decay: float = 1e-4
    # checkpoint 保存间隔（epoch 数）
    checkpoint_interval: int = 10


@dataclass
class ScorerConfig:
    # 丰富性（L2 范数）权重
    w_rich: float = 1.0
    # 稀缺性（平均余弦距离）权重
    w_rare: float = 0.3
