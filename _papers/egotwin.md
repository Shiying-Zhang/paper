---
title: "EgoTwin: Dreaming Body and View in First Person"
authors: "Jingqiao Xiu, Fangzhou Hong, Yicong Li, Mengze Li, Wentao Wang, Sirui Han, Liang Pan, Ziwei Liu"
year: 2025
venue: "arXiv 2025"
tags: [视频生成, 人体运动]
rating: 4
paper_url: "https://arxiv.org/abs/2508.13013"
code_url: ""
summary: "联合第一人称视频和人体运动生成：头部中心表示 + 前向/逆向动力学注意力掩码 + 异步扩散，视角-运动对齐误差降 70%"
---

## 核心思想

同时生成**第一人称视频**和**全身运动**，满足两个约束：
1. **视角对齐**：相机轨迹必须与头部轨迹一致
2. **因果交互**：运动影响后续视频帧，视频帧反过来约束运动

**EgoTwin** 提出三个创新：头部中心运动表示、控制论启发的注意力掩码、异步扩散训练。

## 方法详解

### 1. 模态 Token 化

#### 1.1 文本和视频

- 文本：T5-XXL，$\mathbf{c} \in \mathbb{R}^{L_t \times D_t}$（$L_t = 226, D_t = 3072$）
- 视频：3D 因果 VAE（压缩比 4×8×8），$\mathbf{z}_v \in \mathbb{R}^{(N_v/4+1) \times H/8 \times W/8 \times C_v}$

#### 1.2 运动表示：从根中心到头部中心

**传统根中心表示**包含 7 组特征：根角速度 $\dot{r}^a$、根线速度 $\dot{r}^{xz}$、根高度 $r^y$、局部关节位置 $j^p$、局部关节速度 $j^v$、关节旋转 $j^r$、脚地接触 $c^f$。

**头部中心表示**（本文提出）：

$$(h^r, \dot{h}^r, h^p, \dot{h}^p, j^p, j^v, j^r)$$

- $h^r \in \mathbb{R}^6$：头部绝对旋转（6D 表示）
- $\dot{h}^r \in \mathbb{R}^6$：头部相对旋转
- $h^p \in \mathbb{R}^3$：头部绝对位置
- $\dot{h}^p \in \mathbb{R}^3$：头部相对位置
- 关节位置/速度转到头部坐标系

**为什么头部中心更好**：第一人称相机固定在头部，头部中心表示使**相机轨迹直接可读取**——无需从根关节逆运动学推导。

#### 1.3 运动 VAE

$$\mathcal{L}_\text{VAE} = \frac{1}{4} \sum_{c} \left[\mathcal{L}_\text{rec}^{(c)} + \lambda_\text{KL} \mathcal{L}_\text{KL}^{(c)}\right] \tag{1}$$

$c \in \{\text{head\_3D}, \text{head\_6D}, \text{joint\_3D}, \text{joint\_6D}\}$，对 4 种特征组分别计算重建和 KL 损失。1D 因果卷积 + ResNet 块，2 级 2× 下采样，运动 latent $\mathbf{Z}_m \in \mathbb{R}^{(N_m/4+1) \times C_m}$。

### 2. 扩散 Transformer

#### 2.1 三分支架构

- **文本 + 视频分支**：42 层，从 CogVideoX 初始化，共享权重（~5B 参数）
- **运动分支**：21 层（下半部分），缩小通道维度（~300M 参数）
- 统一 embedding 维度 $D = 3072$

#### 2.2 控制论启发的注意力掩码

设运动 token 数 $N_m = 2N_v$。基于前向/逆向动力学原则：

- **前向动力学**：$\{O^i, A^i\} \to O^{i+1}$（观察+动作→下一观察）
- **逆向动力学**：$\{O^i, O^{i+1}\} \to A^i$（两帧观察→推断动作）

注意力规则：
- 视频 token $O^i$ 关注运动 token $A^{i-1}$（前向：过去动作决定当前观察）
- 运动 token $A^i$ 关注视频 token $O^i$ 和 $O^{i+1}$（逆向：相邻观察约束当前动作）
- 初始姿态 $P^0$ 与初始帧 $I^0$ 双向关注
- 模态内和文本相关的注意力保持不变

#### 2.3 异步扩散

$$\mathcal{L}_\text{DiT} = \mathbb{E}\left[\|\boldsymbol{\varepsilon}_v - \boldsymbol{\varepsilon}_\theta^v(\mathbf{z}_v^{(t_v)}, \mathbf{z}_m^{(t_m)}, \mathbf{c}, t_v, t_m)\|_2^2 + \|\boldsymbol{\varepsilon}_m - \boldsymbol{\varepsilon}_\theta^m(\mathbf{z}_m^{(t_m)}, \mathbf{z}_v^{(t_v)}, \mathbf{c}, t_m, t_v)\|_2^2\right] \tag{2}$$

$t_v$ 和 $t_m$ **独立采样**——视频和运动的去噪时间步不同。

**为什么异步**：在推理时，一个模态可能已经很"干净"而另一个还很"噪"，异步训练使模型适应这种不对称性。

### 3. 条件生成采样

#### 3.1 TM2V（文本+运动→视频）

$$\hat{\boldsymbol{\varepsilon}}_\theta^v = \boldsymbol{\varepsilon}_\theta^v(\mathbf{z}_v^t, \mathbf{z}_m^T, \phi, t, T) + w_t[\cdots] + w_m[\boldsymbol{\varepsilon}_\theta^v(\mathbf{z}_v^t, \mathbf{z}_m^0, \mathbf{c}, t, 0) - \boldsymbol{\varepsilon}_\theta^v(\mathbf{z}_v^t, \mathbf{z}_m^T, \mathbf{c}, t, T)] \tag{3}$$

$w_t$：文本引导尺度，$w_m$：运动引导尺度。$\mathbf{z}_m^0$ 为干净运动 latent。

#### 3.2 TV2M（文本+视频→运动）

$$\hat{\boldsymbol{\varepsilon}}_\theta^m = \boldsymbol{\varepsilon}_\theta^m(\mathbf{z}_m^t, \mathbf{z}_v^T, \phi, t, T) + w_t[\cdots] + w_v[\boldsymbol{\varepsilon}_\theta^m(\mathbf{z}_m^t, \mathbf{z}_v^t, \mathbf{c}, t, t) - \boldsymbol{\varepsilon}_\theta^m(\mathbf{z}_m^t, \mathbf{z}_v^T, \mathbf{c}, t, T)] \tag{4}$$

### 4. 三阶段训练

1. **运动 VAE 训练**：公式 (1)
2. **文本→运动预训练**：冻结文本分支，10% 文本 dropout
3. **联合文本-视频-运动训练**：全模态，10% 文本 dropout

## 实验结果

### 主要结果

| 指标 | VidMLD | EgoTwin | 提升 |
|------|--------|---------|------|
| I-FID↓ | 157.86 | **98.17** | 37.8% |
| FVD↓ | 1547.28 | **1033.52** | 33.2% |
| CLIP-SIM↑ | 25.58 | **27.34** | +1.76 |
| M-FID↓ | 45.09 | **41.80** | 7.3% |
| TransErr↓ | 1.28 | **0.67** | **47.7%** |
| RotErr↓ | 1.53 | **0.46** | **69.9%** |
| HandScore↑ | 0.36 | **0.81** | +125% |

视角对齐误差（TransErr, RotErr）大幅降低。

### 消融实验

| 配置 | I-FID↓ | RotErr↓ | HandScore↑ |
|------|--------|---------|-----------|
| 根中心表示 | 134.27 | 1.22 | 0.44 |
| 无注意力掩码 | 117.54 | 0.89 | 0.57 |
| 同步扩散 | 109.73 | 0.62 | 0.73 |
| **完整 EgoTwin** | **98.17** | **0.46** | **0.81** |

三个组件均有显著贡献，头部中心表示对视角对齐最关键。

## 个人思考

1. **头部中心表示**直接暴露相机信息是核心设计——从根中心推导头部位姿会引入累积误差。
2. **前向/逆向动力学注意力掩码**将控制论原则引入 Transformer 设计——因果关系不是随意的，而是有物理结构的。
3. **异步扩散**是多模态生成的通用技巧——不同模态的"难度"不同，不应强制同步去噪。
4. **HandScore 从 0.36 到 0.81**（+125%）说明手部可见性对齐极大改善——这对 AR/VR 应用至关重要。
5. **5B 参数**的模型较大，但通过 CogVideoX 初始化避免了从零训练的成本。
