---
title: "Generative Modeling via Drifting"
authors: "Mingyang Deng, He Li, Tianhong Li, Yilun Du, Kaiming He"
year: 2025
venue: "arXiv 2025"
tags: [图像生成, 自监督学习]
rating: 5
paper_url: "https://arxiv.org/abs/2602.04770"
code_url: ""
summary: "提出 Drifting Models 新范式，在训练时演化 pushforward 分布，实现单步生成，ImageNet 256×256 达到 FID 1.54"
---

## 核心思想

传统扩散/流模型在**推理时**迭代演化分布（需要多步 NFE），而本文提出了全新范式 **Drifting Models**：在**训练时**演化 pushforward 分布，推理时只需**单步前向传播**。

核心直觉：定义一个 **drifting field** $V$ 来控制生成样本的移动方向，当生成分布 $q$ 与数据分布 $p$ 匹配时，drifting field 达到平衡（$V=0$）。

![Teaser: 网络 f 将先验分布映射为数据分布，训练过程中 loss 逐渐降低](https://arxiv.org/html/2602.04770v1/extracted_media/figures/teaser_main.png)

## 方法详解

### 1. Pushforward 分布：问题建模

生成模型的核心是学习一个映射 $f$，使其 **pushforward 分布**匹配数据分布：

$$q = f_\# \, p_\varepsilon \tag{1}$$

其中 $p_\varepsilon$ 为先验分布（如标准高斯），$f_\#$ 表示 $f$ 诱导的 pushforward 操作。训练过程产生参数序列 $\{\theta_i\}$，对应生成分布序列 $\{q_i\}$，其中 $q_i = [f_{\theta_i}]_\# \, p_\varepsilon$。

**与扩散模型的区别**：扩散模型在推理时通过多步迭代演化分布（$q_0 \to q_1 \to \cdots \to q_T$），而 Drifting Models 将这种演化放到训练过程中，推理时只需单步 $x = f_\theta(\varepsilon)$。

### 2. Drifting Field：核心定义

训练迭代中，生成样本 $x_i = f_{\theta_i}(\varepsilon)$ 按以下规则漂移：

$$x_{i+1} = x_i + V_{p, q_i}(x_i) \tag{2}$$

其中 $V_{p,q}: \mathbb{R}^d \to \mathbb{R}^d$ 为 **drifting field**，下标 $p, q$ 表示它同时依赖数据分布 $p$ 和当前生成分布 $q$。

![Drifting 示意图：生成样本 x（黑色）被蓝色正样本吸引、被橙色负样本排斥](https://arxiv.org/html/2602.04770v1/extracted_media/figures/drift_field.png)

#### 反对称性（Proposition 3.1）

Drifting field 必须满足**反对称性**：

$$V_{p,q}(x) = -V_{q,p}(x), \quad \forall x \tag{3}$$

**为什么需要反对称性？** 当 $q = p$（生成分布匹配数据分布）时：

$$V_{p,p}(x) = -V_{p,p}(x) \implies V_{p,p}(x) = 0$$

即达到**平衡状态**，样本不再漂移。这保证了训练收敛。

### 3. Drifting Field 的具体构造

#### 3.1 Kernel 形式

Drifting field 通过 kernel 函数构造：

$$V_{p,q}(x) = \mathbb{E}_{y^+ \sim p} \mathbb{E}_{y^- \sim q} [\mathcal{K}(x, y^+, y^-)] \tag{7}$$

其中 $y^+$ 是数据样本（正样本），$y^-$ 是生成样本（负样本）。

#### 3.2 吸引项与排斥项

将 $V$ 分解为两项——**正样本吸引**和**负样本排斥**：

$$V_{p,q}(x) = V_p^+(x) - V_q^-(x) \tag{10}$$

其中吸引项和排斥项分别定义为：

$$V_p^+(x) := \frac{1}{Z_p(x)} \mathbb{E}_{y^+ \sim p}\big[k(x, y^+)(y^+ - x)\big] \tag{8a}$$

$$V_q^-(x) := \frac{1}{Z_q(x)} \mathbb{E}_{y^- \sim q}\big[k(x, y^-)(y^- - x)\big] \tag{8b}$$

归一化因子为：

$$Z_p(x) := \mathbb{E}_{y^+ \sim p}[k(x, y^+)], \quad Z_q(x) := \mathbb{E}_{y^- \sim q}[k(x, y^-)] \tag{9}$$

**直觉理解**：
- $V_p^+(x)$：将 $x$ 拉向附近的真实数据点 $y^+$，权重由 kernel $k$ 决定（越近权重越大）
- $V_q^-(x)$：将 $x$ 推离附近的生成样本 $y^-$，防止模式崩塌
- 两项相减 → 生成样本既向数据聚拢，又彼此散开

#### 3.3 展开形式

将公式 (8a)(8b) 代入 (10)，得到完整展开：

$$V_{p,q}(x) = \frac{1}{Z_p(x) \cdot Z_q(x)} \mathbb{E}_{y^+ \sim p, y^- \sim q}\big[k(x, y^+) \cdot k(x, y^-) \cdot (y^+ - y^-)\big] \tag{11}$$

**反对称性验证**：交换 $p \leftrightarrow q$（即 $y^+ \leftrightarrow y^-$），$(y^+ - y^-)$ 变号 → $V_{p,q} = -V_{q,p}$ ✓

#### 3.4 Kernel 函数

$$k(x, y) = \exp\!\Big(-\frac{1}{\tau}\|x - y\|\Big) \tag{12}$$

其中 $\tau$ 为温度参数。归一化通过 softmax 实现：$\tilde{k}(x, y) = \text{softmax}(-\frac{1}{\tau}\|x - y\|)$。

### 4. 训练目标推导

#### 4.1 固定点条件

当训练收敛（$q_{\hat{\theta}} \approx p$）时，$V = 0$，得到固定点关系：

$$f_{\hat{\theta}}(\varepsilon) = f_{\hat{\theta}}(\varepsilon) + V_{p, q_{\hat{\theta}}}(f_{\hat{\theta}}(\varepsilon)) \tag{4}$$

#### 4.2 迭代更新

根据公式 (2)，训练时第 $i$ 步到第 $i+1$ 步的更新：

$$f_{\theta_{i+1}}(\varepsilon) \leftarrow f_{\theta_i}(\varepsilon) + V_{p, q_{\theta_i}}(f_{\theta_i}(\varepsilon)) \tag{5}$$

即网络的目标输出是「当前输出 + 漂移量」。

#### 4.3 Loss 函数

将公式 (5) 转化为回归损失，并使用 **stop gradient** 冻结目标端：

$$\boxed{\mathcal{L} = \mathbb{E}_\varepsilon\Big[\big\|f_\theta(\varepsilon) - \text{stopgrad}\big(f_\theta(\varepsilon) + V_{p,q_\theta}(f_\theta(\varepsilon))\big)\big\|^2\Big]} \tag{6}$$

**公式之间的联系**：
- 公式 (6) 的最优解满足公式 (4) 的固定点条件
- stopgrad 使得右侧 $f_\theta(\varepsilon) + V$ 被视为常量目标，梯度只通过左侧 $f_\theta$ 反传
- 展开后等价于 $\mathcal{L} = \mathbb{E}_\varepsilon[\|V(f_\theta(\varepsilon))\|^2]$，即最小化漂移量 → 趋向平衡
- SGD 天然地实现了公式 (5) 的迭代演化，无需显式定义 ODE/SDE

### 5. 特征空间扩展

直接在像素空间计算 $V$ 效果有限，扩展到预训练编码器的**特征空间**：

#### 5.1 单特征空间

$$\mathcal{L}_\text{feat} = \mathbb{E}_\varepsilon\Big[\big\|\varphi(x) - \text{stopgrad}\big(\varphi(x) + V(\varphi(x))\big)\big\|^2\Big] \tag{13}$$

其中 $\varphi$ 为预训练编码器（MAE / MoCo / SimCLR），$x = f_\theta(\varepsilon)$。$V$ 中的 $k(x, y)$、$(y^+ - x)$ 等都在特征空间中计算。

#### 5.2 多尺度特征

从编码器的不同层提取特征，计算各层漂移损失并求和：

$$\mathcal{L}_\text{multi} = \sum_j \mathbb{E}_\varepsilon\Big[\big\|\varphi_j(x) - \text{stopgrad}\big(\varphi_j(x) + V(\varphi_j(x))\big)\big\|^2\Big] \tag{14}$$

其中 $\varphi_j$ 为第 $j$ 层/尺度的特征。这使得模型同时优化低层纹理和高层语义。

### 6. Classifier-Free Guidance (CFG)

对于类条件生成，通过修改负样本分布实现 CFG：

$$\tilde{q}(\cdot|c) := (1 - \gamma) \, q_\theta(\cdot|c) + \gamma \, p_\text{data}(\cdot|\varnothing) \tag{15}$$

其中 $\gamma \in [0, 1)$ 为混合率，$q_\theta(\cdot|c)$ 为类条件生成分布，$p_\text{data}(\cdot|\varnothing)$ 为无条件数据分布。

等价地，优化目标变为让 $q_\theta$ 匹配一个"增强"分布：

$$q_\theta(\cdot|c) = \alpha \, p_\text{data}(\cdot|c) - (\alpha - 1) \, p_\text{data}(\cdot|\varnothing), \quad \alpha = \frac{1}{1 - \gamma} \geq 1 \tag{16}$$

$\alpha$ 控制条件引导强度，$\alpha = 1$ 退化为无引导，$\alpha > 1$ 增强类别条件信号。

### 7. 训练算法伪代码

```
输入：生成器 f_θ，预训练编码器 φ，数据集 D
每个训练迭代：
  1. 采样噪声 ε ~ p_ε
  2. 生成样本 x = f_θ(ε)
  3. 采样正样本 y⁺ ~ p_data（N_pos 个数据点）
  4. 负样本 y⁻ = x（重用生成样本）
  5. 提取特征 φ(x), φ(y⁺), φ(y⁻)
  6. 计算 kernel: k(φ(x), φ(y⁺)), k(φ(x), φ(y⁻))     # 公式 (12)
  7. 计算吸引项 V⁺ 和排斥项 V⁻                           # 公式 (8a)(8b)
  8. 合成 V = V⁺ - V⁻                                    # 公式 (10)
  9. 冻结目标 x_target = stopgrad(φ(x) + V)
  10. 损失 L = ||φ(x) - x_target||²                      # 公式 (6/13)
  11. 反向传播更新 θ

推理（1-NFE）：
  ε ~ p_ε → x = f_θ(ε) → 输出 x
```

## 实验结果

### ImageNet 256×256 主要结果

| 方法 | 空间 | NFE | FID ↓ |
|------|------|-----|-------|
| **Drifting (本文)** | **Latent** | **1** | **1.54** |
| **Drifting (本文)** | **Pixel** | **1** | **1.61** |
| DMD2 | Latent | 1 | 1.28 |
| SiD-LSG | Latent | 1 | 1.38 |
| REPA | Latent | 250 | 1.42 |
| DiT-XL/2 | Latent | 250 | 2.27 |

- 单步生成中 **pixel space SOTA**（FID 1.61），大幅超越此前像素空间方法
- Latent space FID 1.54，与蒸馏方法（DMD2, SiD-LSG）接近，但**无需预训练教师模型**

### 消融实验关键发现

- **反对称性至关重要**：去掉排斥项 FID 飙升到 41~177
- **特征编码器质量**影响很大：latent-MAE width=640 达到 FID 3.36
- 正/负样本数越多效果越好

### 机器人操作任务

在机器人操控任务上，单步 Drifting 匹配甚至超越 100 步 Diffusion Policy。

![训练过程中生成分布（橙色）向双峰目标分布（蓝色）的演化过程](https://arxiv.org/html/2602.04770v1/extracted_media/figures/dynamics_comparison.png)

## 个人思考

1. **范式创新**：不同于扩散模型"推理迭代"的思路，将迭代放到训练时，推理只需一步。这是一个非常优雅的 reformulation。
2. **与对比学习的联系**：drifting field 的吸引-排斥机制与对比学习有异曲同工之妙，正样本吸引、负样本排斥。
3. **无需教师模型**：相比蒸馏类单步方法（DMD2 等），本方法从头训练，不依赖预训练的多步扩散模型作为教师。
4. **实用价值**：单步生成对实时应用（如机器人控制）极有价值，实验也验证了这一点。
5. **潜在局限**：训练时需要维护生成样本集合并计算 drifting field，可能带来额外的训练开销和内存需求。
