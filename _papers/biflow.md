---
title: "Bidirectional Normalizing Flow: From Data to Noise and Back"
authors: "Yiyang Lu, Qiao Sun, Xianbang Wang, Zhicheng Jiang, Hanhong Zhao, Kaiming He"
year: 2025
venue: "arXiv 2025"
tags: [图像生成, Transformer]
rating: 4
paper_url: "https://arxiv.org/abs/2512.10953"
code_url: ""
summary: "提出双向归一化流 BiFlow，用可学习反向模型替代精确解析逆，相比因果解码加速 100 倍，NF 方法中 SOTA"
---

## 核心思想

传统归一化流（NF）要求前向和反向过程互为**精确解析逆函数**，这严重限制了架构设计（必须用因果注意力，逐块自回归解码）。本文提出 **BiFlow（双向归一化流）**：用一个**可学习的反向模型**近似逆映射，彻底移除精确逆约束。

结果：架构解锁双向注意力，生成质量提升，采样速度加速 **100 倍**。

![标准 NF（精确逆函数）vs BiFlow（学习逆函数）的概念对比](https://arxiv.org/html/2512.10953v1/extracted_media/x1.png)

## 方法详解

### 1. 预备知识：标准归一化流

#### 1.1 对数似然目标

$$\log p(\mathbf{x}) = \log p_0(\mathbf{z}) + \sum_i \log \left|\det \frac{\partial f_i(\mathbf{x}^i)}{\partial \mathbf{x}^i}\right| \tag{1}$$

其中 $\mathbf{x}^0 = \mathbf{x}$，$\mathbf{x}^{i+1} = f_i(\mathbf{x}^i)$，$\mathbf{z} = \mathcal{F}(\mathbf{x})$。

**含义**：通过变量变换公式，数据的对数似然由**先验分布项** + **所有变换的雅可比行列式项**组成。

**限制**：为保证可逆性和行列式可计算，前向模型 $\mathcal{F}$ 通常受限于因果注意力架构（如 TARFlow），导致自回归解码速度极慢。

### 2. BiFlow：学习反向过程

BiFlow 的核心是不再要求 $\mathcal{G} = \mathcal{F}^{-1}$（精确逆），而是学习一个参数化的反向模型 $\mathcal{G}_\phi$ 来近似逆映射。

#### 2.1 三种学习方法对比

**朴素蒸馏**：直接让反向模型拟合输入-输出对

$$\mathcal{L}_\text{naive} = \mathcal{D}(\mathbf{x}, \mathcal{G}_\phi(\mathbf{z}))$$

问题：忽略了中间块的轨迹信息。

**隐蒸馏**：对齐每个块的中间输出

$$\mathcal{L}_\text{hidden} = \sum_i \mathcal{D}(\mathbf{x}^i, \mathcal{G}_\phi^{-i}(\mathbf{z}))$$

问题：需要反复投影回输入空间，信息损失。

**隐对齐（本文采用）**：用可学习投影头将反向模型隐状态与前向轨迹对齐：

$$\boxed{\mathcal{L}_\text{align}(\mathbf{x}) = \sum_i \mathcal{D}(\mathbf{x}^i, \varphi_i(\mathbf{h}^i))} \tag{2}$$

其中 $\mathbf{h}^0 = \mathbf{x}'$（反向模型输出），$\varphi_i$ 为可学习投影头。

**为什么隐对齐最优？**
- 利用**完整前向轨迹** $\{\mathbf{x}^0, \mathbf{x}^1, \ldots, \mathbf{x}^N\}$ 进行逐层监督
- 投影头 $\varphi_i$ 允许反向模型的中间表示空间**独立于输入空间**
- 避免隐蒸馏的重复投影信息损失

![三种反向学习方法对比：(a) 朴素蒸馏 (b) 隐蒸馏 (c) 隐对齐](https://arxiv.org/html/2512.10953v1/extracted_media/x5.png)

| 方法 | 注意力类型 | FID ↓ |
|------|----------|-------|
| 精确逆函数 | 因果 | 44.46 |
| 朴素蒸馏 | 双向 | 43.41 |
| 隐蒸馏 | 双向 | 55.00 |
| **隐对齐** | **双向** | **36.93** |

### 3. 学习去噪

#### 3.1 原始 TARFlow 的显式去噪

TARFlow 在生成后需要额外的基于分数的去噪步骤：

$$\mathbf{x} \leftarrow \tilde{\mathbf{x}} + \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log p(\tilde{\mathbf{x}}) \tag{3}$$

问题：需要额外的前向-后向传递，几乎加倍推理成本。

#### 3.2 BiFlow 的学习去噪

BiFlow 将去噪集成为反向模型的一个额外块，联合训练。无需额外计算，去噪在单次前向传播中完成。

![TARFlow 显式去噪 vs BiFlow 学习去噪](https://arxiv.org/html/2512.10953v1/extracted_media/x8.png)

### 4. 训练时 CFG

#### 4.1 标准 CFG

$$\mathbf{h}^{i+1} = (1 + w_i)\mathcal{G}_\phi^i(\mathbf{h}^i | \mathbf{c}) - w_i \mathcal{G}_\phi^i(\mathbf{h}^i) \tag{4}$$

其中 $\mathbf{c}$ 是类条件，$w_i$ 是引导尺度。问题：推理时需要 2-NFE（条件 + 无条件各一次）。

#### 4.2 训练时 CFG 重参数化

$$\mathbf{h}^{i+1} = \frac{\mathcal{G}_\phi^{i, \text{cfg}}(\mathbf{h}^i | \mathbf{c}) + w_i \mathcal{G}_\phi^{i, \text{cfg}}(\mathbf{h}^i)}{1 + w_i} \tag{5}$$

**关键**：在训练时直接优化 CFG 版本的输出，推理时保持 **1-NFE**。

### 5. 自适应加权损失

$$\mathcal{D}_p = \text{sg}(w_p) \cdot \mathcal{D}(\mathbf{x}, \mathbf{y}), \quad w_p = (\mathcal{D}(\mathbf{x}, \mathbf{y}) + c)^{-p} \tag{7}$$

根据损失大小自适应调整权重，平衡不同距离尺度的梯度。

### 6. 范数控制

通过裁剪前向模型参数和归一化轨迹，确保各块之间的**平衡监督**。消融实验表明 FID 从 45.54 降至 31.88。

### 7. 噪声空间修复（零样本应用）

$$\mathbf{z}' = \mathcal{M} \odot \mathbf{z}_\text{mask} + (1 - \mathcal{M}) \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \tag{6}$$

保留掩码区域的噪声，重新采样非掩码区域，**训练无关**的修复能力。

## 实验结果

### BiFlow vs 改进 TARFlow

| 模型 | 参数量 | FID ↓ | 推理时间 | 相对速度 |
|------|--------|-------|---------|---------|
| **BiFlow-B/2** | 133M | **2.39** | 1.6ms | 基准 |
| iTARFlow-B/2 | 120M | 6.83 | 66ms | 42× 慢 |
| iTARFlow-XL/2 | 690M | 4.54 | 203ms | 128× 慢 |

BiFlow 以 **1/5 参数量** 达到 **更优 FID**，同时快 **42-128 倍**。

### 与其他方法对比（ImageNet 256×256）

| 方法 | 类型 | NFE | FID ↓ |
|------|------|-----|-------|
| **BiFlow-B/2** | **NF** | **1** | **2.39** |
| STARFlow-XL/1 | NF | 4 | 2.40 |
| iMF-XL/2 | Flow | 1 | 1.72 |
| DiT-XL/2 | Diffusion | 250 | 2.27 |
| SiT-XL/2 | Diffusion | 250 | 2.06 |

BiFlow 在 NF 方法中 **SOTA**，参数量仅为 STARFlow 的 1/10，速度快 400 倍。

### 距离度量消融

| 距离度量 | FID ↓ |
|---------|-------|
| MSE | 31.88 |
| + LPIPS | 14.15 |
| + ConvNeXt | **2.46** |

感知损失贡献约 **30 FID 点**的提升。

## 个人思考

1. **移除精确逆约束**是一个非常大胆的设计：传统 NF 的数学优雅性（精确似然）被放弃，换来了架构自由度和速度提升。
2. **隐对齐 > 隐蒸馏**的结论有启发性：中间表示不必强制对齐到输入空间，给予反向模型独立的表示自由度更好。
3. **训练时 CFG** 是一个巧妙的工程技巧：将 2-NFE 变为 1-NFE，实际部署中速度翻倍。
4. **与 Drifting / iMF 的定位差异**：BiFlow 来自 NF 传统，保留了前向模型的似然训练；而 Drifting/iMF 来自 Flow Matching 传统。
5. **噪声空间修复**是 NF 架构的独有优势——双射映射使得噪声空间操作可以直接映射回图像空间。
