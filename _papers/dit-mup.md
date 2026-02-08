---
title: "Scaling Diffusion Transformers Efficiently via μP"
authors: "Chenyu Zheng, Xinyu Zhang, Rongzhen Wang, Wei Huang, Zhi Tian, Weilin Huang, Jun Zhu, Chongxuan Li"
year: 2025
venue: "NeurIPS 2025"
tags: [图像生成, Transformer, 训练优化]
rating: 4
paper_url: "https://arxiv.org/abs/2505.15270"
code_url: ""
summary: "将 μP（最大更新参数化）从标准 Transformer 扩展到扩散 Transformer，实现超参数从小模型到大模型的零成本迁移，DiT-XL 收敛加速 2.9×"
---

## 核心思想

训练大型扩散 Transformer（DiT, PixArt-α, MMDiT）时，超参数调优成本极高。本文证明**主流扩散 Transformer 架构的 μP 与标准 Transformer 的 μP 完全一致**，从而可以在小模型上搜索最优超参数，直接迁移到大模型使用。

**核心定理**：DiT、PixArt-α、MMDiT 等架构的前向传播可以用 Ne⊗or⊤ Program 表示，因此它们的 μP 匹配标准 μP。

## 方法详解

### 1. abc-参数化框架

对每个权重矩阵 $W$，定义三元组 $(a_W, b_W, c_W)$ 控制宽度 $n$ 相关的缩放：

$$W = \phi_W \cdot n^{-a_W} \cdot \tilde{W} \tag{权重分解}$$

$$\tilde{W}_{ij} \sim \mathcal{N}(0, \sigma_W^2 \cdot n^{-2b_W}) \tag{初始化方差}$$

$$\eta_W = \eta_W^{\text{base}} \cdot n^{-c_W} \tag{学习率缩放}$$

其中 $\phi_W, \sigma_W, \eta_W^{\text{base}}$ 为**宽度无关**的基础超参数。

### 2. μP vs 标准参数化

| 权重类型 | μP $(a_W, b_W, c_W)$ | 标准 $(a_W, b_W, c_W)$ |
|---------|----------------------|------------------------|
| 输入层 | (0, 0, 1) | (0, 0, 0) |
| 隐藏层 | (0, 1/2, 1) | (0, 1/2, 1/2) |
| 输出层 | (1, 0, 0) | (0, 1/2, 1/2) |

**关键差异**：μP 中输出层学习率不随宽度缩放（$c_W=0$），而隐藏层学习率随 $1/n$ 缩放（$c_W=1$）。这保证了每一层的参数更新量在训练过程中保持**宽度无关**。

### 3. 迁移成本公式

$$\text{成本比} = \frac{R \times S_\text{proxy} \times B_\text{proxy} \times T_\text{proxy}}{S_\text{target} \times B_\text{target} \times T_\text{target}} \tag{1}$$

其中 $R$ 为搜索次数，$S$ 为参数量，$B$ 为 batch size，$T$ 为训练步数。

### 4. 扩展到扩散 Transformer

#### 4.1 处理 adaLN（DiT）

DiT 使用 adaptive layer normalization 将时间步信息注入网络。本文证明 adaLN 中的缩放和平移参数可以在 Ne⊗or⊤ 框架中表示，因此 μP 直接适用。

#### 4.2 处理 cross-attention（PixArt-α）

文本特征通过 cross-attention 注入。文本 embedding 维度固定，仅图像侧的宽度变化——标准 μP 的隐藏层规则直接适用。

#### 4.3 处理多模态注意力（MMDiT）

图像和文本有独立参数集但共享注意力。本文证明每个模态的参数集独立满足 μP 条件。

### 5. 缩放策略

- 固定注意力头维度（如 DiT/PixArt-α 为 72）
- 通过增加**头数**（而非头维度）来扩大模型宽度
- 基础宽度 $n_\text{base}$ 设为固定参考点（如 DiT 288 = 4 heads × 72）

### 6. 三阶段流程

1. **实现 abc-参数化**：按表 1 对各层权重应用宽度相关的初始化和学习率缩放
2. **验证基础超参数可迁移性**：在不同宽度/batch size/步数下确认最优基础超参数一致
3. **μTransfer**：在小代理模型上搜索最优超参数，直接应用于目标大模型

## 实验结果

### DiT-XL-2 on ImageNet

- 最优基础学习率：$2^{-10}$，在宽度 144-1152、batch size 64-512、步数 50K-200K 下均保持一致
- DiT-XL-2-μP 在 **2.4M 步**达到目标 FID，原始 DiT 需要 7M 步
- **2.9× 收敛加速**

### PixArt-α 缩放

| 配置 | 参数量 | 调优成本 |
|------|--------|---------|
| 代理模型 | 0.04B | 5 epochs, 39K 步 |
| 目标模型 | 0.61B | 30 epochs, 59K 步 |
| **成本比** | — | **5.5%** |

| Epoch 30 | PixArt-α | PixArt-α-μP |
|----------|----------|-------------|
| GenEval | 0.15 | **0.26** |
| FID-30K (MJHQ) | 42.71 | **29.96** |
| FID-30K (COCO) | 37.61 | **25.84** |

### MMDiT 缩放到 18B

- 代理模型：0.18B（目标的 1%），30K 步
- 目标模型：18B，200K 步
- 搜索空间：80 trials × 4 超参数
- **调优成本：仅为人工专家调参的 3%**

| 指标 | MMDiT-18B | MMDiT-μP-18B |
|------|-----------|-------------|
| GenEval | 0.8154 | **0.8218** |
| 对齐准确率 | 0.703 | **0.715** |

### 关键发现

1. **梯度裁剪**：标准做法 ~0.1，μP 最优值为 1.0——过度裁剪破坏了最大更新特性
2. **泛化增强**：PixArt-α-μP 在 epoch 20 后继续改善，基线反而退化
3. **最优学习率一致性**：多 epoch 训练偏好接近稳定性极限的学习率（$2^{-10}$）

## 个人思考

1. **理论证明的优雅**：不是启发式地调整 μP，而是证明扩散 Transformer 的前向传播可以用 Ne⊗or⊤ Program 表示 → μP 自然成立。
2. **实用价值极高**：18B 模型的超参数调优从不可行（需全量训练）变为仅 3% 成本——对工业界意义重大。
3. **梯度裁剪的反直觉发现**：标准的 0.1 裁剪值会破坏 μP 的最大更新保证，最优值 1.0 意味着几乎不裁剪。
4. **泛化改善**是意外收获：μP 不仅加速收敛，还改善了模型泛化——可能因为最优超参数处于过拟合和欠拟合的平衡点。
5. **代理模型选择**是开放问题：多小的代理模型仍然有效？这关系到成本下限。
