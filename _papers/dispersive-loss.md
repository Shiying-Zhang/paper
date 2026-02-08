---
title: "Diffuse and Disperse: Image Generation with Representation Regularization"
authors: "Runqian Wang, Kaiming He"
year: 2025
venue: "arXiv 2025"
tags: [图像生成, 自监督学习]
rating: 4
paper_url: "https://arxiv.org/abs/2506.09027"
code_url: ""
summary: "提出 Dispersive Loss——无需正样本对的对比学习正则化，无需预训练/额外参数/外部数据，一致提升扩散模型质量"
---

## 核心思想

扩散模型的中间表示是否可以通过正则化来改善生成质量？本文提出 **Dispersive Loss**：一种「**无正样本对的对比学习**」正则化器，鼓励模型内部表示在隐空间中**分散**（disperse）。

**与对比学习的关系**：对比学习 = 正样本对齐 + 负样本排斥。Dispersive Loss 只保留**负样本排斥**项，因为扩散模型的回归损失已经自然提供了对齐功能。

**优势**：无需预训练、无需额外参数、无需外部数据（对比 REPA 需要 1.1B 额外参数 + 142M 外部图像）。

![Dispersive Loss 应用于扩散模型中间层](https://arxiv.org/html/2506.09027v2/extracted_media/x1.png)

## 方法详解

### 1. 训练目标

$$\mathcal{L}(\mathbf{X}) = \mathbb{E}[\mathcal{L}_\text{Diff}(\mathbf{x}_i)] + \lambda \mathcal{L}_\text{Disp}(\mathbf{X}) \tag{1}$$

其中 $\mathcal{L}_\text{Diff}$ 为标准扩散损失，$\mathcal{L}_\text{Disp}$ 为 Dispersive Loss，$\lambda$ 为权重，$\mathbf{X}$ 为整个 batch。

### 2. 从对比学习到 Dispersive Loss 的推导

#### 2.1 标准 InfoNCE

$$\mathcal{L}_\text{Contrast} = -\log \frac{\exp(-\mathcal{D}(\mathbf{z}_i, \mathbf{z}_i^+)/\tau)}{\sum_j \exp(-\mathcal{D}(\mathbf{z}_i, \mathbf{z}_j)/\tau)} \tag{2}$$

#### 2.2 分解为两项

$$\mathcal{L}_\text{Contrast} = \underbrace{\mathcal{D}(\mathbf{z}_i, \mathbf{z}_i^+)/\tau}_{\text{正样本对齐}} + \underbrace{\log \sum_j \exp(-\mathcal{D}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}_{\text{负样本排斥}} \tag{3}$$

**关键观察**：扩散损失 $\mathcal{L}_\text{Diff}$ 已经通过回归目标隐式地提供了表示对齐（第一项）。因此只需要第二项。

#### 2.3 Dispersive Loss（去掉正样本对齐项）

$$\mathcal{L}_\text{Disp} = \log \sum_j \exp(-\mathcal{D}(\mathbf{z}_i, \mathbf{z}_j)/\tau) \tag{4}$$

进一步写成 batch 级期望形式：

$$\mathcal{L}_\text{Disp} = \log \mathbb{E}_j[\exp(-\mathcal{D}(\mathbf{z}_i, \mathbf{z}_j)/\tau)] \tag{5}$$

最终的 batch 对称形式：

$$\boxed{\mathcal{L}_\text{Disp} = \log \mathbb{E}_{i,j}[\exp(-\mathcal{D}(\mathbf{z}_i, \mathbf{z}_j)/\tau)]} \tag{6}$$

**公式链条**：InfoNCE (2) → 分解 (3) → 去掉正样本项 → Dispersive Loss (4-6)

### 3. 距离函数选择

- **余弦距离**：$\mathcal{D}(\mathbf{z}_i, \mathbf{z}_j) = -\mathbf{z}_i^\top \mathbf{z}_j / (\|\mathbf{z}_i\|\|\mathbf{z}_j\|)$
- **L2 距离**：$\mathcal{D}(\mathbf{z}_i, \mathbf{z}_j) = \|\mathbf{z}_i - \mathbf{z}_j\|_2^2$

实验表明 InfoNCE + L2 效果最好。

### 4. 三种变体

| 变体 | 公式 | FID ↓ |
|------|------|-------|
| **InfoNCE (L2)** | 公式 (6) + L2 距离 | **32.35** |
| InfoNCE (cosine) | 公式 (6) + 余弦距离 | 34.33 |
| Hinge | 铰链损失变体 | 33.93 |
| Covariance | 最小化协方差矩阵非对角元素 | 35.82 |
| Baseline | 无正则化 | 36.49 |

### 5. 应用位置

Dispersive Loss 应用于 Transformer 的中间 block 的隐表示 $\mathbf{z}_i$：

| 应用层 | FID ↓ |
|--------|-------|
| 不应用 | 36.49 |
| Block 0 | 33.97 |
| Block 6 | 33.79 |
| Block 11 | 33.86 |
| **所有 blocks** | **32.05** |

应用于所有层效果最好。有趣的是，即使只在某一层应用，**其他层的表示范数也会增加**——正则化效果会传播。

![表示范数变化：Dispersive Loss 增加所有层的范数](https://arxiv.org/html/2506.09027v2/extracted_media/x3.png)

### 6. 超参数鲁棒性

所有 $\lambda \in \{0.25, 0.5, 1.0\}$ 和 $\tau \in \{0.25, 0.5, 1.0, 2.0\}$ 的组合都大幅优于 baseline（36.49），方法对超参数非常鲁棒。

## 实验结果

### 多模型/多尺度一致性

Dispersive Loss 在 DiT 和 SiT 的 S/2、B/2、L/2、XL/2 四个尺度上均带来一致改进。

![跨模型尺度的一致改进](https://arxiv.org/html/2506.09027v2/extracted_media/x4.png)

### SiT-XL/2 长训练

| 训练轮数 | Baseline FID | + Dispersive FID | 相对改进 |
|---------|-------------|-----------------|---------|
| 80 epochs | 18.46 | 15.95 | −13.6% |
| 800 epochs | 8.99 | 8.08 | −10.1% |

### 单步生成

MeanFlow-XL/2: FID 从 3.43 降至 **3.21**（−6.4%）。

### 与 REPA 对比

| 方法 | FID ↓ | 额外参数 | 外部数据 | 预训练 |
|------|-------|---------|---------|--------|
| REPA | 1.80 | 1.1B | 142M 图像 | 1500 epochs |
| **Dispersive** | **1.97** | **0** | **0** | **0** |

Dispersive Loss 以零额外成本达到接近 REPA 的效果。

## 个人思考

1. **「对比学习去掉正样本」**的 formulation 非常优雅：扩散损失自带对齐，只需排斥即可。
2. **正则化传播效应**有趣：只在一层应用，所有层的表示范数都增加，说明 Transformer 各层之间的耦合很强。
3. **零成本改进**：不需要预训练编码器、不增加参数、不需要外部数据，是真正的 plug-and-play。
4. **与 Drifting Model 的联系**：Drifting 的排斥项和 Dispersive Loss 的设计动机一致——防止表示/样本坍缩。
5. **局限性**：与 REPA（使用预训练 DINOv2）相比仍有差距，说明外部知识的注入在某些情况下确实有帮助。
