---
title: "SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation"
authors: "Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Dongyun Zou, Han Cai, Song Han, Enze Xie"
year: 2025
venue: "arXiv 2025"
tags: [图像生成, 蒸馏]
rating: 5
paper_url: "https://arxiv.org/abs/2503.09641"
code_url: ""
summary: "免训练 TrigFlow 变换 + sCM + LADD 混合蒸馏，实现 0.1s 生成 1024×1024 图像，FID 7.59，比 FLUX-schnell 快 10×"
---

## 核心思想

如何将多步扩散模型蒸馏为单步/少步模型？本文提出 **SANA-Sprint**，三大创新：
1. **免训练 TrigFlow 变换**：将 Flow Matching 模型无损转换为 TrigFlow 参数化
2. **连续时间一致性模型 (sCM)**：保持多样性的蒸馏损失
3. **LADD 对抗蒸馏**：增强保真度的对抗损失

结果：1024×1024 图像 **0.1s** 延迟（H100），FID 7.59，GenEval 0.74。

## 方法详解

### 1. 免训练 TrigFlow 变换

**动机**：sCM 需要 TrigFlow 参数化，但预训练模型使用 Flow Matching。重新训练代价巨大。本文证明可以**无损转换**。

#### 1.1 时间映射

$$t_\text{FM} = \frac{\sin(t_\text{Trig})}{\sin(t_\text{Trig}) + \cos(t_\text{Trig})} \tag{6}$$

将 TrigFlow 时间步映射到 Flow Matching 时间步。

#### 1.2 输入缩放

$$\mathbf{x}_{t,\text{FM}} = \frac{\mathbf{x}_{t,\text{Trig}}}{\sigma_d} \cdot \sqrt{t_\text{FM}^2 + (1-t_\text{FM})^2} \tag{7}$$

#### 1.3 输出变换

$$\hat{F}_\theta(\mathbf{x}_{t,\text{Trig}}/\sigma_d, t_\text{Trig}, y) = \frac{1}{\sqrt{t_\text{FM}^2 + (1-t_\text{FM})^2}} \left[(1-2t_\text{FM})\mathbf{x}_{t,\text{FM}} + (1-2t_\text{FM}+2t_\text{FM}^2)\mathbf{v}_\theta(\cdots)\right] \tag{8}$$

**无损性证明**：通过等价 SNR 匹配和条件期望的线性性。

**验证**：Flow Euler 50 步 FID 5.81 → TrigFlow Euler 50 步 FID 5.73（完全等价）。

### 2. 连续时间一致性模型 (sCM) 损失

$$\mathcal{L}_\text{sCM}(\theta, \varphi) = \mathbb{E}_{\mathbf{x}_t, t}\left[\frac{e^{w_\varphi(t)}}{D} \left\|\hat{F}_\theta(\cdot) - \hat{F}_{\theta^-}(\cdot) - \cos(t) \frac{d\hat{f}_{\theta^-}}{dt}\right\|_2^2 - w_\varphi(t)\right] \tag{9}$$

其中时间导数在 TrigFlow 下展开为：

$$\frac{d\hat{f}_{\theta^-}}{dt} = -\cos(t)\left(\sigma_d \hat{F}_{\theta^-}(\cdot) - \frac{d\mathbf{x}_t}{dt}\right) - \sin(t)\left(\mathbf{x}_t + \sigma_d \frac{d\hat{F}_{\theta^-}}{dt}\right) \tag{5}$$

通过 **JVP (Jacobian-Vector Product)** 高效计算。

**公式链条**：TrigFlow 变换 (6-8) → 使 sCM 损失 (9) 可以直接应用于 FM 预训练模型 → JVP 计算时间导数 (5)。

### 3. 稳定化机制

#### 3.1 Dense Time Embedding

将噪声系数 $c_\text{noise}(t)$ 从 $1000t$ 降为 $t$，控制梯度放大：

$$\partial_t F_\theta = \frac{\partial F_\theta}{\partial \text{emb}(c_\text{noise})} \cdot \frac{\partial \text{emb}(c_\text{noise})}{\partial c_\text{noise}} \cdot \frac{\partial c_\text{noise}(t)}{\partial t}$$

减少 $c_\text{noise}$ 的范围 → 降低 JVP 中的梯度放大 1000 倍。

#### 3.2 QK-Normalization

对 self/cross-attention 的 Query/Key 应用 RMSNorm，稳定大模型的 JVP 计算。

### 4. LADD 对抗蒸馏

#### 4.1 判别器损失

$$\mathcal{L}_\text{adv}^D(\psi) = \mathbb{E}\left[\sum_k \text{ReLU}(1 - D_{\psi,k}(\cdot))\right] + \mathbb{E}\left[\sum_k \text{ReLU}(1 + D_{\psi,k}(\cdot))\right] \tag{12}$$

#### 4.2 生成器损失

$$\mathcal{L}_\text{adv}^G(\theta) = -\mathbb{E}\left[\sum_k D_{\psi,k}(F_{\theta^\text{pre},k}(\hat{\mathbf{x}}_s^{f_\theta}, s, y))\right] \tag{11}$$

#### 4.3 组合损失

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{sCM} + \lambda \cdot \mathcal{L}_\text{adv}, \quad \lambda = 0.5$$

**sCM 保持多样性，LADD 增强保真度**——两者互补。

### 5. Max-Time 加权策略

以概率 $p$ 采样 $t = \pi/2$（最大噪声级别），强调早期时间步的保真度。$p=50\%$ 为单步质量的最优配置。

### 6. 两阶段训练

| 阶段 | 迭代 | 学习率 | 内容 |
|------|------|--------|------|
| Phase 1：教师微调 | 5,000 | 2e-5 | Dense embedding + QK-norm |
| Phase 2：学生蒸馏 | 20,000 | 2e-6 | sCM + LADD 联合训练 |

### 7. 步自适应统一模型

单个 checkpoint 支持 1-4 步推理，无需重训练。

| 步数 | 时间步 |
|------|--------|
| 1 步 | $[\pi/2, 0.0]$ |
| 2 步 | $[\arctan(200/0.5), 1.3, 0.0]$ |
| 4 步 | $[\arctan(200/0.5), 1.3, 1.1, 0.6, 0.0]$ |

## 实验结果

### 性能对比

| 模型 | 步数 | 吞吐量 | 延迟 | FID↓ | GenEval↑ |
|------|------|--------|------|------|----------|
| SANA-Sprint 0.6B | 1 | 7.22 | 0.21s | 7.04 | 0.72 |
| SANA-Sprint 1.6B | 1 | 6.71 | 0.21s | 7.69 | 0.76 |
| FLUX-schnell | 1 | 1.58 | 0.68s | 7.26 | 0.69 |
| SANA-Sprint 1.6B | 4 | 5.20 | 0.31s | 6.54 | 0.77 |
| FLUX-schnell | 4 | 0.50 | 2.10s | 7.94 | 0.71 |

4 步时比 FLUX-schnell 快 **64.7×**。

### 消融实验

| 配置 | FID↓ | CLIP Score↑ |
|------|------|------------|
| sCM only | 8.93 | 27.51 |
| LADD only | 12.20 | 27.00 |
| **sCM + LADD** | **8.11** | **28.02** |

| Max-Time 比例 | FID↓ |
|--------------|------|
| 0% | 9.44 |
| **50%** | **8.32** |
| 70% | 8.11 |

## 个人思考

1. **免训练 TrigFlow 变换**是最优雅的贡献：省去 4.8B 模型的预训练成本，且数学上无损。
2. **sCM + LADD 互补**的发现：sCM 保多样性（避免模式崩塌），LADD 增保真度（避免模糊），两者相加 FID 改善 0.6。
3. **稳定化设计**看似简单但至关重要：$c_\text{noise}$ 从 1000 降到 1、QK-norm 都是让 JVP 训练稳定的必要条件。
4. **0.1s 延迟**使得实时交互式图像生成成为可能——ControlNet 版本仅 0.25s。
5. **步自适应**很实用：用户可以根据延迟/质量需求选择 1-4 步，单模型搞定。
