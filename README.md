# 个人技术空间

这是 [zhanghanleo10.github.io](https://zhanghanleo10.github.io/) 的 GitHub Pages 博客源码。

## 发布文章

在 `_posts/` 下创建 Markdown 文件，文件名使用：

```text
YYYY-MM-DD-article-slug.md
```

文章头部示例：

```yaml
---
layout: post
title: "文章标题"
description: "文章摘要"
date: 2026-07-24
category: "vLLM 源码"
tags: [vLLM, Scheduler]
reading_time: "约 10 分钟"
---
```

正文直接使用 Markdown 编写。提交到 `main` 分支后，GitHub Pages 会重新构建网站。

## Mermaid 图表

需要 Mermaid 的文章在 front matter 中增加：

```yaml
mermaid: true
```

正文使用标准 Mermaid fenced code block：

````markdown
```mermaid
flowchart LR
    A["输入"] --> B["处理"] --> C["输出"]
```
````

只有声明 `mermaid: true` 的页面会加载 Mermaid 渲染模块。若图表语法错误，页面会保留原始代码块作为回退，浏览器控制台会记录解析错误。
