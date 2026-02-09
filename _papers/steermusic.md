---
title: "SteerMusic: Enhanced Musical Consistency for Zero-shot Text-Guided and Personalized Music Editing"
authors: "Xinlei Niu, Kin Wai Cheuk, Jing Zhang, Naoki Murata, Chieh-Hsin Lai, Michele Mancusi, Woosung Choi, Giorgio Fabbro, Wei-Hsiang Liao, Charles Patrick Martin, Yuki Mitsufuji"
year: 2025
venue: "arXiv 2025"
tags: [音乐编辑, 扩散模型, 个性化生成]
rating: 4
paper_url: "https://arxiv.org/abs/2504.10826"
code_url: ""
summary: "零样本文本引导音乐编辑：Delta 去噪分数在数据空间优化（避免反演误差）+ 个性化概念 token + 分布偏移正则化 + 对比损失保持旋律，FAD/CLAP/MOS 全面最优"
---

## 核心思想

音乐编辑（如"把这段钢琴曲的乐器换成吉他"）需要**改变目标属性的同时保持旋律一致性**。

现有方法的问题：扩散模型的 DDIM 反演（inversion）会引入**累积误差** → 编辑后的音乐失去原始结构。

**SteerMusic** 提出：
1. **SteerMusic**（粗粒度编辑）：用 Delta Denoising Score（DDS）在数据空间直接优化，**绕过反演**
2. **SteerMusic+**（细粒度个性化编辑）：学习用户定义的音乐概念 token → 实现纯文本无法描述的编辑

## 背景知识

### 音乐编辑 vs 图像编辑

| 方面 | 图像编辑 | 音乐编辑 |
|------|---------|---------|
| 结构保持 | 保持布局/姿态 | **保持旋律/节奏** |
| 可用工具 | DDIM inversion 成熟 | **反演误差更大（音频更连续）** |
| 感知距离 | LPIPS | **LPAPS（不成熟）** |
| 编辑粒度 | 局部编辑成熟 | **全局风格转换为主** |

### 什么是 Delta Denoising Score

不在潜在空间做反演+编辑，而是**在数据空间直接优化**：

$$\nabla_\theta \mathcal{L}_{\text{DDS}} = \mathbb{E}\left[w(t) \left(\epsilon_\phi(x_t, y^{\text{tgt}}, t) - \epsilon_\phi(x_t^{\text{src}}, y^{\text{src}}, t)\right) \frac{\partial x}{\partial \theta}\right]$$

核心思想：不是"反演→编辑→生成"，而是直接优化生成结果使其**既像目标又像源**。

## 方法详解

### 1. SteerMusic：零样本编辑

#### DDS 梯度

计算目标和源的去噪预测差：

$$\nabla_\theta \mathcal{L}_{\text{DDS}} = \mathbb{E}\left[w(t) \left(\epsilon_\phi(x_t, y^{\text{tgt}}, t) - \epsilon_\phi(x_t^{\text{src}}, y^{\text{src}}, t)\right) \frac{\partial x}{\partial \theta}\right]$$

- $y^{\text{tgt}}$：目标提示词（如"guitar version"）
- $y^{\text{src}}$：源提示词（如"piano piece"）
- $x$：优化的音频参数

**优势**：源和目标的**差值**消除了共享的反演误差。

### 2. SteerMusic+：个性化编辑

#### 2.1 个性化扩散模型（PDM）

在用户提供的参考音频 $\mathcal{D}^{\text{ref}} = \{(x^{\text{ref}}, y^{\text{ref}})_n\}$ 上微调，学习新的概念 token。

#### 2.2 Personalized Delta Score（PDS）

$$\nabla_\theta \mathcal{L}_{\text{PDS}} = \mathbb{E}\left[w(t) \left(\epsilon_{\phi'}(x_t, y^{\text{tgt}}, t) - \epsilon_\phi(x_t^{\text{src}}, y^{\text{src}}, t)\right) \frac{\partial x}{\partial \theta}\right]$$

用**个性化模型** $\epsilon_{\phi'}$ 提供目标方向，用**标准模型** $\epsilon_\phi$ 提供源方向。

#### 2.3 分布偏移正则化

个性化模型的分布与标准模型不同 → 会引入偏移。正则化项：

$$\mathcal{L}_{\text{shift}} = \mathbb{E}\left[w(t) \|\epsilon_{\phi'}(x_t, y^{\text{tgt}}, t) - \epsilon_\phi(x_t, y^{\text{tgt}}, t)\|_2^2\right]$$

组合损失：

$$\nabla_\theta \mathcal{L}_{\text{PDS-O}} = \nabla_\theta \mathcal{L}_{\text{PDS}} + \lambda \nabla_\theta \mathcal{L}_{\text{shift}}$$

#### 2.4 个性化对比（PCon）损失

强制源和目标之间的**时间特征对齐**：

$$\mathcal{L}_{\text{PCon}} = \mathbb{E}_h \left[\sum_l \sum_{t'} \ell(h_l^{t'}, h_l^{\text{src},t'}, h_l^{\text{src},T_l \setminus t'})\right]$$

- $h_l^{t'}$：第 $l$ 层在时间 $t'$ 的特征
- $\ell$：softmax 对比学习损失（温度 $\tau$）
- 鼓励编辑后的音频在时间结构上与源对齐

### 3. 基座模型

**AudioLDM2**：在 Mel 频谱图上操作的潜在扩散模型。

## 实验结果

### 数据集

- **ZoME-Bench**：1000 个样本（10 秒），包含乐器变换（131）、风格变换（134）、情绪变换（100）、背景修改（95）
- **个性化概念**：8 种风格（4 乐器 + 4 风格）

### 零样本文本引导编辑

| 方法 | FAD_CLAP↓ | FAD_Vggish↓ | CQT-1 PCC↑ | LPAPS↓ | CLAP↑ | MOS-P↑ | MOS-T↑ |
|------|----------|-----------|-----------|--------|-------|--------|--------|
| DDIM | 0.477 | 1.022 | 0.330 | 5.377 | 0.264 | 1.37 | 1.91 |
| SDEdit | 0.638 | 1.274 | 0.169 | 6.208 | 0.218 | 0.92 | 1.68 |
| MusicMagus | 0.593 | 1.698 | 0.338 | 5.243 | 0.238 | 2.11 | 1.57 |
| ZETA | 0.509 | 1.021 | 0.293 | 5.458 | 0.252 | 1.22 | 1.60 |
| **SteerMusic** | **0.278** | **0.381** | **0.480** | **3.772** | 0.259 | **2.92** | **2.50** |

- FAD 降低 **45-70%** → 音频质量大幅提升
- CQT-1 PCC 提升 **42%** → 旋律保持显著改善
- MOS-P（保持度）**2.92** 远超基线

### 个性化音乐编辑

| 方法 | FAD_CLAP↓ | CQT-1 PCC↑ | CDPAM↓ | MOS-P↑ | MOS-T↑ |
|------|----------|-----------|--------|--------|--------|
| Textual Inv. | 0.789 | 0.216 | 0.713 | 1.64 | 1.63 |
| DreamSound | 0.902 | 0.292 | 0.609 | 1.42 | 1.81 |
| **SteerMusic+** | **0.362** | **0.399** | **0.593** | **3.07** | **2.47** |

### 用户偏好

- SteerMusic vs ZETA/MusicMagus：**70-80%** 偏好 SteerMusic
- SteerMusic+ vs DreamSound/Textual Inv.：**75-85%** 偏好 SteerMusic+

## 个人思考

1. **绕过反演**是核心洞察：DDIM 反演在音频上的误差比图像更大（连续信号的离散化损失），DDS 直接在数据空间优化彻底避免了这个问题。
2. **CQT-1 PCC 作为旋律保持指标**非常合适：Constant-Q 变换的低频成分直接对应旋律信息 → 比通用的感知距离更能反映音乐编辑的核心需求。
3. **分布偏移正则化**解决了个性化模型的关键问题：微调后的模型分布与原始模型不同 → 不正则化会导致编辑方向偏移。
4. **"保持旋律 vs 贴近风格"的权衡**是音乐编辑的根本矛盾 → $\lambda$ 参数给用户提供了控制这个权衡的能力。
5. **MOS-P 2.92/3.07 vs 基线 1-2**：人类评价的巨大差距说明现有方法在旋律保持上**完全不可用**，SteerMusic 是第一个实用的解决方案。
