---
title: "Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers"
authors: "Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, Song Han"
year: 2024
venue: "arXiv 2024"
tags: [图像生成, Transformer]
rating: 5
paper_url: "https://arxiv.org/abs/2410.10629"
code_url: ""
summary: "全栈高效 T2I 系统：32× 压缩 AE + 线性注意力 DiT + Decoder-only LLM 文本编码器 + Flow-DPM-Solver，4K 生成比 FLUX 快 100×"
---

## 核心思想

Sana 是一个**全栈优化**的文本到图像生成系统，在每个组件上都追求效率：
1. **Deep Compression AE** (f32c32)：32× 压缩，16× token 减少
2. **Linear DiT**：$O(N)$ 线性注意力 + Mix-FFN + 无位置编码
3. **Decoder-only LLM 文本编码器**（Gemma-2）：比 T5-XXL 快 6×
4. **Flow-DPM-Solver**：14-20 步高质量采样

结果：0.6B 参数模型在 1024×1024 上比 FLUX-dev 快 **39.5×**。

## 方法详解

### 1. Deep Compression Autoencoder (AE-F32C32P1)

**核心思想**：将压缩交给 AE，让扩散模型专注于去噪。

| 配置 | 压缩率 | Token 数 (1024px) | rFID |
|------|--------|------------------|------|
| AE-F8C4P2 (传统) | 低 | 4096 | 0.31 |
| AE-F8C16P4 | 中 | 1024 | 0.23 |
| **AE-F32C32P1** | **高** | **1024** | **0.34** |

**关键发现**：AE-F8C16P4 重建最好（rFID 0.23），但生成 FID 反而最差！原因：让 AE 做更多压缩 + 让 DiT 做更多生成 = 更好的分工。

### 2. Linear Diffusion Transformer

#### 2.1 线性注意力

标准注意力 $O(N^2)$ → 线性注意力 $O(N)$：

$$O_i = \frac{\text{ReLU}(Q_i)\left(\sum_{j=1}^{N}\text{ReLU}(K_j)^T V_j\right)}{\text{ReLU}(Q_i)\left(\sum_{j=1}^{N}\text{ReLU}(K_j)^T\right)} \tag{1}$$

预计算 $\sum \text{ReLU}(K_j)^T V_j$ 后，每个 token 仅需 $O(D^2)$ 计算。

#### 2.2 Mix-FFN

替代标准 MLP-FFN：反转残差块 → 3×3 深度卷积 → GLU（Gated Linear Unit）

**作用**：补偿线性注意力较弱的局部特征聚合能力。

#### 2.3 无位置编码 (NoPE)

**惊人发现**：去掉位置编码后性能不降！原因：3×3 卷积 + zero padding **隐式编码**了位置信息。

#### 2.4 性能对比 (1024×1024)

| 配置 | 延迟 |
|------|------|
| Full attention + FFN (F8C4P2) | 2250ms |
| Linear attention | 1931ms |
| + Mix-FFN | 2425ms |
| **+ Kernel Fusion (F32C32P1)** | **748ms** |

### 3. 文本编码器：Gemma-2

#### 3.1 从 T5 切换到 Decoder-only LLM

**动机**：现代 decoder-only LLM 有更好的指令理解和推理能力。

**稳定性问题**：Decoder-only LLM 的文本 embedding 方差比 T5 **大几个数量级** → 训练 NaN。

**解决方案**：
1. 对 Gemma-2 输出应用 RMSNorm（归一化方差到 1.0）
2. 可学习缩放因子（初始化 ~0.01）乘以文本嵌入

**效果**：Gemma-2-2B 推理比 T5-XXL 快 **6×**，质量相当。

### 4. Complex Human Instruction (CHI)

利用 Gemma 的上下文学习和 CoT 能力增强短提示：

| 训练配置 | GenEval↑ |
|---------|----------|
| 仅用户提示 | 45.5 |
| **CHI + 用户提示** | **47.7** (+2.2) |

CHI 防止了短提示（如 "a cat"）产生的伪影。

### 5. Flow-DPM-Solver

Flow Matching 使用速度预测：$\mathbf{v}_\theta(\mathbf{x}_t, t) = \boldsymbol{\varepsilon} - \mathbf{x}_0$

**理论优势**：在 $t \approx T$ 时（噪声端），数据预测近似为常量：

$$\mathbf{x}_\theta(\mathbf{x}_t, t) \approx \mathbb{E}_{q_0(\mathbf{x}_0)}[\mathbf{x}_0] \tag{4}$$

常量预测 → 减少离散化误差和累积采样误差。

Flow-DPM-Solver 改编自 DPM-Solver++：
- 缩放因子替换：$\alpha_t \to 1-\sigma_t$
- 时间步重定义：$[1,1000] \to [0,1]$

**效果**：从 28-50 步（Euler）降至 **14-20 步**，FID 和 CLIP Score 更好。

### 6. 数据流水线

#### 6.1 多 Caption 自动标注

每张图像由 4 个 VLM 标注（VILA-3B/13B, InternVL2-8B/26B），增加多样性。

#### 6.2 CLIP-Score 采样器

$$P(c_i) = \frac{\exp(c_i / \tau)}{\sum_j \exp(c_j / \tau)}$$

高 CLIP Score 的 caption 以更高概率被采样，改善文图对齐。

### 7. 端侧部署

- **量化策略**：W8A8（INT8 权重 + 激活）
- **保持全精度**：normalization 层、线性注意力、cross-attention KV
- **CUDA 优化**：融合 ReLU(K)^T V 与 QKV 投影、GLU 与量化

**RTX 4090**：1024×1024 从 0.88s 降至 **0.37s** (2.4× 加速)。

## 实验结果

### 1024×1024 性能对比

| 模型 | 参数 | 延迟 | FID↓ | CLIP↑ | GenEval↑ |
|------|------|------|------|-------|----------|
| FLUX-dev | 12B | 23s | 10.15 | 27.47 | 0.67 |
| **Sana-0.6B** | **0.6B** | **0.9s** | **5.81** | **28.36** | 0.64 |
| Sana-1.6B | 1.6B | — | — | — | — |

**39.5× 加速**，参数量仅为 FLUX 的 1/20。

### 4K 生成

| 模型 | 延迟 |
|------|------|
| FLUX-dev | 469s |
| **Sana-1.6B** | **9.6s** (106× 加速) |

### 消融：NoPE 的惊人效果

去掉所有位置编码（APE, RoPE），仅靠 Mix-FFN 中的 3×3 卷积，性能不降反升——这是首个**无位置编码的 DiT**。

## 个人思考

1. **全栈优化**的协同效应巨大：AE 32× 压缩 + 线性注意力 + Gemma-2 文本编码器 + Flow-DPM-Solver，每个组件的提升叠加产生 39-106× 加速。
2. **让 AE 做更多压缩**的哲学反直觉但正确——AE 擅长压缩（这是它的本职工作），DiT 擅长生成。
3. **NoPE 的发现**很惊人：位置编码在 DiT 中可能是多余的，3×3 卷积足以隐式编码位置。
4. **Gemma-2 稳定性修复**（RMSNorm + 缩放因子）是关键工程贡献——没有这个修复就无法使用 decoder-only LLM。
5. **端侧部署**（RTX 4090 上 0.37s）展示了实际应用潜力——真正的实时图像生成。
