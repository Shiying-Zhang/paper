# Paper Reading Notes

> **92 篇**深度论文笔记 | 扩散模型 / 图像生成 / 强化学习 / 机器人 / 多模态 / 智能体 | 持续更新中

[![GitHub Pages](https://img.shields.io/badge/Online-GitHub%20Pages-blue?logo=github)](https://Shiying-Zhang.github.io/paper)
[![Papers](https://img.shields.io/badge/Papers-92-green)](#coverage)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**[>>> 在线阅读 <<<](https://Shiying-Zhang.github.io/paper)**

---

## What is this?

这是一个**结构化的 AI 论文阅读笔记库**，每篇笔记不是简单的摘要翻译，而是包含：

- **核心思想**：用一句话讲清楚论文在解决什么问题
- **背景知识**：为入门读者补充必要的前置知识（含对比表格）
- **方法详解**：完整的公式推导链，不跳步骤
- **实验结果**：关键数据表格，一眼看清提升幅度
- **个人思考**：对方法的批判性分析和延伸思考

全部笔记使用**中文**撰写，适合中文母语读者快速理解前沿论文。

## Coverage

| 方向 | 篇数 | 代表论文 |
|------|------|---------|
| 图像生成 / 扩散模型 | 20+ | MeanFlow, sCM, RAE, Sana, DAPS |
| 强化学习 / LLM Agent | 15+ | SeeUPO, GRPO, AgeMem, MemAgent, EARL |
| 机器人操控 / 具身智能 | 10+ | SKT, Mantis-VLA, DiffGen, GRACE |
| 视频生成与理解 | 7+ | Vid-CamEdit, SoundCTM, LiveTalk |
| 3D 视觉 / 重建 | 5+ | GeoSplatting, Grendel-GS, ArtGS |
| 多模态 / 视觉语言 | 9+ | Cambrian-S, RL4VLM, RegionGPT |
| 扩散语言模型 | 5+ | LLaDA, RFG, ESPO, OTS |
| 科学应用 | 3+ | MOLEXA, SR-Scientist |

> 覆盖 NeurIPS / ICML / ICLR / ECCV / MLSys / TMLR 等顶会及最新 arXiv 预印本

## Quick Start

**在线浏览**：访问 [Shiying-Zhang.github.io/paper](https://Shiying-Zhang.github.io/paper)，支持：
- 关键词搜索（标题、作者、摘要）
- 标签筛选（点击标签云）
- 按年份 / 评分排序
- 直达原文和代码链接

**本地运行**：

```bash
git clone https://github.com/Shiying-Zhang/paper.git
cd paper && git checkout gh-pages
bundle install && bundle exec jekyll serve
# 访问 http://localhost:4000/paper
```

## Project Structure

```
main branch          # README + 项目说明
gh-pages branch      # Jekyll 网站源码
  ├── _papers/       # 92 篇论文笔记 (Markdown)
  ├── _layouts/      # 页面模板
  ├── _includes/     # 组件（搜索栏、标签云等）
  ├── assets/        # 样式和脚本
  └── index.md       # 首页
```

## Contributing

欢迎通过 Issue 推荐论文或指出笔记中的错误。如果你也想贡献笔记：

1. Fork 本仓库，切换到 `gh-pages` 分支
2. 在 `_papers/` 下新建 `.md` 文件，按以下格式填写：

```yaml
---
title: "论文标题"
authors: "作者列表"
year: 2025
venue: "会议/期刊"
tags: [标签1, 标签2]
rating: 4
paper_url: "https://arxiv.org/abs/xxxx"
code_url: ""
summary: "一句话中文摘要"
---

## 核心思想
## 背景知识
## 方法详解
## 实验结果
## 个人思考
```

3. 提交 PR

## Star History

如果这个项目对你有帮助，欢迎点个 Star 支持一下！
