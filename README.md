# Paper Reading Notes

个人论文阅读笔记归档，基于 Jekyll 构建，部署在 GitHub Pages。

## 在线访问

https://Shiying-Zhang.github.io/paper

## 如何添加新论文笔记

在 `_papers/` 目录下新建一个 `.md` 文件，按以下格式填写：

```yaml
---
title: "论文标题"
authors: "作者1, 作者2"
year: 2024
venue: "CVPR 2024"
tags: [标签1, 标签2]
rating: 4
paper_url: "https://arxiv.org/abs/xxxx"
code_url: "https://github.com/xxx"
summary: "一句话摘要"
---

## 核心思想
...

## 方法详解
...

## 实验结果
...

## 个人思考
...
```

文件名建议使用论文关键词的英文缩写，如 `vision-transformer.md`、`detr.md`。

## 本地预览

```bash
# 安装 Jekyll
gem install bundler jekyll

# 启动本地服务
cd paper
bundle exec jekyll serve
```

访问 `http://localhost:4000/paper` 预览。

## 部署

Push 到 GitHub 后，在仓库 Settings > Pages 中选择 `main` 分支即可自动部署。
