---
title: "OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models"
authors: "Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, Ping Luo"
year: 2024
venue: "arXiv 2024"
tags: [模型量化, LLM推理]
rating: 5
paper_url: "https://arxiv.org/abs/2308.13137"
code_url: "https://github.com/OpenGVLab/OmniQuant"
summary: "以 PTQ 效率达到 QAT 性能：可学习权重裁剪（LWC）+ 可学习等价变换（LET），单 A100 完成校准，W2A16 PPL 13.21（GPTQ 5500+）"
---

## 核心思想

LLM 量化有两类方法：
- **PTQ（后训练量化）**：快速但精度差（如 GPTQ、AWQ）
- **QAT（量化感知训练）**：精度好但需大量 GPU 时间（如 LLM-QAT）

**OmniQuant** 的目标是**以 PTQ 的效率达到 QAT 的性能**。核心技术：
1. **LWC（可学习权重裁剪）**：自动学习量化范围的最优裁剪比例
2. **LET（可学习等价变换）**：自动学习激活-权重之间的最优缩放和平移

两者都**冻结原始权重**，仅优化少量量化参数 → 训练极其高效。

## 背景知识：量化基础

### 什么是量化

将浮点权重/激活映射到低比特整数：

$$W_q = \text{clamp}\left(\left\lfloor \frac{W}{h} \right\rceil + z, \; 0, \; 2^N - 1\right)$$

其中：
- $N$：目标比特数（如 4-bit → $2^4 - 1 = 15$ 个量化级别）
- $h$：步长（step size），决定两个相邻量化值之间的间距
- $z$：零点（zero point），用于非对称量化
- $\lfloor \cdot \rceil$：四舍五入

### 标准 MinMax 量化

$$h = \frac{\max(W) - \min(W)}{2^N - 1}, \quad z = -\left\lfloor \frac{\min(W)}{h} \right\rceil$$

**问题**：少数异常大值（outlier）会拉大 $\max(W) - \min(W)$ → 步长 $h$ 变大 → 大多数正常值的量化精度变差。

### 为什么 LLM 量化特别难

1. **激活 outlier**：某些通道的激活值比其他通道大 100 倍以上 → 直接量化激活误差巨大
2. **极低比特**：W2（2-bit 权重）只有 4 个量化级别 → 几乎没有容错空间

## 方法详解

### 1. 总体优化框架：逐 block 校准

对每个 Transformer block，最小化量化前后输出的差异：

$$\min_{\Theta_1, \Theta_2} \|\mathcal{F}(W, X) - \mathcal{F}(Q_w(W; \Theta_1, \Theta_2), Q_a(X; \Theta_2))\|$$

- $\mathcal{F}$：Transformer block 的前向计算
- $Q_w, Q_a$：权重和激活的量化函数
- $\Theta_1$：LWC 参数（裁剪比例）
- $\Theta_2$：LET 参数（缩放和平移因子）

**为什么逐 block 而非逐层或全模型**：
- 逐层太贪心，误差会累积
- 全模型搜索空间太大
- 逐 block 是折中：block 内各层协同优化，block 间顺序传播

### 2. LWC：可学习权重裁剪

#### 2.1 核心思想

MinMax 量化的范围 $[\min(W), \max(W)]$ 可能被 outlier 拉大。LWC 学习最优的**裁剪比例** $\gamma, \beta$：

$$h = \frac{\gamma \cdot \max(W) - \beta \cdot \min(W)}{2^N - 1}$$

$$z = -\left\lfloor \frac{\beta \cdot \min(W)}{h} \right\rceil$$

- $\gamma \in [0, 1]$：上界裁剪强度。$\gamma < 1$ 表示裁剪掉上边的 outlier
- $\beta \in [0, 1]$：下界裁剪强度。$\beta < 1$ 表示裁剪掉下边的 outlier
- $\gamma = \beta = 1$：退化为标准 MinMax（不裁剪）

#### 2.2 参数化

用 sigmoid 保证 $\gamma, \beta \in [0, 1]$：

$$\gamma = \sigma(\gamma_\text{raw}), \quad \beta = \sigma(\beta_\text{raw})$$

$\gamma_\text{raw}, \beta_\text{raw}$ 初始化为 0 → $\sigma(0) = 0.5$ → 初始裁剪 50%（后续通过梯度调整到最优值）。

#### 2.3 参数量

每个线性层仅需 **2 个额外参数**（$\gamma, \beta$）→ 增加的参数量可忽略。

### 3. LET：可学习等价变换

#### 3.1 问题：激活 outlier

某些通道的激活值极大（如通道 $j$ 的激活是其他通道的 100 倍）→ 直接量化激活时，这个通道占据了大部分量化范围 → 其他通道精度极差。

#### 3.2 核心思想：等价变换

对线性层 $Y = XW + B$，进行等价变换：

$$Y = XW + B = \underbrace{[(X - \delta) \div s]}_{\tilde{X}} \cdot \underbrace{[s \odot W]}_{\tilde{W}} + \underbrace{[B + \delta W]}_{\tilde{B}}$$

- $s \in \mathbb{R}^{1 \times C_\text{in}}$：**逐通道缩放**因子
- $\delta \in \mathbb{R}^{1 \times C_\text{in}}$：**逐通道平移**因子
- $\odot$：逐通道乘法，$\div$：逐通道除法

**变换的等价性**：$\tilde{X} \cdot \tilde{W} + \tilde{B} = XW + B$（精确相等，无近似）。

**量化后**：

$$Y \approx Q_a(\tilde{X}) \cdot Q_w(\tilde{W}) + \tilde{B}$$

**为什么有效**：
- 激活大的通道：$s$ 取较大值 → $\tilde{X} = X/s$ 变小 → 激活更易量化
- 代价：$\tilde{W} = s \cdot W$ 变大 → 权重量化稍难
- 但权重比激活**更容易量化**（分布更规则）→ 整体误差减小

**与 SmoothQuant 的区别**：SmoothQuant 只有缩放 $s$（手动设定），OmniQuant 有缩放 $s$ + 平移 $\delta$（通过梯度自动学习）。

#### 3.3 注意力矩阵的变换

对自注意力的亲和矩阵 $P = \text{Softmax}(QK^T)$：

$$P = \text{Softmax}\left(\underbrace{(Q \div s_a)}_{\tilde{Q}} \cdot \underbrace{(s_a \odot K^T)}_{\tilde{K}^T}\right)$$

$s_a \in \mathbb{R}^{1 \times C_\text{out}}$ 平衡 Q 和 K 的量化难度。

#### 3.4 零额外推理开销

关键优势：$s, \delta, s_a$ 在校准完成后**融入权重和偏置**：

$$\tilde{W} = s \odot W, \quad \tilde{B} = B + \delta W$$

推理时直接使用 $\tilde{W}, \tilde{B}$ → **零额外计算**。

### 4. 训练算法

```
输入：预训练模型，128 条校准文本（2048 token/条）
for 每个 Transformer block:
    初始化 Θ₁ = {γ, β}（LWC），Θ₂ = {s, δ, s_a}（LET）
    for epoch = 1 to 20:
        1. 对 block 输入 X 应用 LET 变换 → X̃
        2. 对权重 W 应用 LET 变换 + LWC 量化 → Q_w(W̃)
        3. 对激活应用量化 → Q_a(X̃)
        4. 计算 block 输出误差 → ∇ loss
        5. AdamW 更新 Θ₁, Θ₂
    将 s, δ 融入权重/偏置
    量化并固定 block 权重
输出：全量化模型
```

**训练超参数**：
- LWC 学习率：$5 \times 10^{-3}$，LET 学习率：$10^{-2}$
- Epochs：20（标准），40（W2A16 极低比特）
- 硬件：**单张 A100-40G**
- 时间：1-16 小时（7B-70B 模型）

## 实验结果

### 权重量化（WikiText-2 PPL↓）

| 配置 | GPTQ | AWQ | **OmniQuant** |
|------|------|-----|-------------|
| LLaMA-13B W2A16 | 5500+ | 2.8×10⁵ | **13.21** |
| LLaMA-13B W3A16 | 6.76 | 7.45 | **5.68** |
| LLaMA-13B W4A16 | 5.40 | 5.34 | **5.21** |

W2（2-bit）时 GPTQ 完全崩溃（PPL 5500+），OmniQuant 仅 13.21——**巨大优势**。

### 权重-激活量化（LLaMA-7B 零样本准确率）

| 方法 | W6A6 | W4A4 |
|------|------|------|
| SmoothQuant | 62.81% | 38.41% |
| OS+ | 61.13% | 48.43% |
| LLM-QAT | — | 41.27% |
| **OmniQuant** | **63.17%** | **52.65%** |

W4A4 比前最优 PTQ 高 **4.22%**，甚至超过 QAT 方法 LLM-QAT。

### 实际部署速度（LLaMA-7B，A100-80G）

| 配置 | 显存 | 速度 |
|------|------|------|
| FP16 | 12.6GB | 69.2 tok/s |
| W4A16g128 | 3.8GB | **134.2 tok/s** |
| W2A16g128 | 2.2GB | 83.9 tok/s |

W4A16 实现 **2× 加速**和 **3.3× 显存节省**。

### 消融：LWC + LET 的协同效应

| 配置 | PPL (W4A4) |
|------|-----------|
| 仅 LET | 48.83% |
| LET + 网格搜索裁剪 | 49.59% |
| **LET + LWC（联合优化）** | **52.65%** |

联合优化远优于分步优化——LWC 和 LET 相互适应。

### 数据效率

- 仅需 **16 条**校准样本即可收敛
- 跨数据集（WikiText2/C4/Pile）方差 < 0.17 PPL

## 个人思考

1. **"冻结权重 + 学习量化参数"** 是精妙的设计：避免了 QAT 的大量计算，又比 PTQ 的手工设定好得多。
2. **LWC 的极致简洁**：每层仅 2 个额外参数就解决了裁剪范围的优化问题——参数效率极高。
3. **LET 的"等价变换"思想**：零额外推理开销是核心优势——变换融入权重后，推理路径与标准量化完全一致。
4. **W2 的突破性结果**（PPL 13.21 vs GPTQ 5500+）说明在极低比特下，手工设计的量化方案完全失效，而学习型方法仍能工作。
5. **单 A100 + 几小时**的训练成本使方法真正可用——不是理论上好但用不起的方法。
