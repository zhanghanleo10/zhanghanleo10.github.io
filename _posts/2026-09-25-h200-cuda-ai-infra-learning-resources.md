---
layout: post
title: "H200 CUDA 与 AI Infra 学习资料导航：主教材、阅读顺序与实践路线"
description: "以 NVIDIA CUDA Programming Guide 为主教材，整理性能分析、Hopper 优化和 AI 算子实践资料，并给出章节阅读顺序与 8 周实验路线。"
date: 2026-09-25 12:00:00 +0800
category: "GPU / CUDA"
tags: [CUDA, H200, Hopper, AI Infra, 学习资料]
reading_time: "约 10 分钟"
permalink: /articles/h200-cuda-ai-infra-learning-resources/
---

面向已有初步 GPU 概念、准备通过 H200 进入 AI Infra 开发的学习者。本文主要归档资料的用途、阅读顺序和实验入口，方便后续按主题查阅。

**推荐以 CUDA Programming Guide 为唯一主教材，配合性能工具与小实验；进入算子和系统项目后，再补充 CUTLASS、Triton 和多卡资料。**“理解编程模型”和“能独立优化真实系统”之间，需要通过可复现的实验建立联系。

资料核对日期：2026-09-25。下文章节编号依据当日官方 HTML 目录；在线文档会更新，查阅时优先匹配章节标题，并核对本机 Toolkit 与示例要求。

## 1. 主教材：CUDA Programming Guide

- [官方 PDF](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf)
- [官方 HTML 目录](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)

PDF 适合离线阅读和批注；HTML 适合搜索、跳转章节以及在实验记录中引用。本文章节依据 HTML 版核对，没有逐页审阅 PDF。

官方将前 3 部分组织为学习路径，将功能专题和附录作为开发时的参考。建议先掌握常用编程机制，再带着问题查阅高级功能。

| 阅读顺序 | 章节 | 阅读目的 | 建议配套实验 |
|---|---|---|---|
| 1 | 1.2 Programming Model | 建立 host/device、thread/block/grid、warp 和 SM 的基本关系 | 画出 vector add 的线程与数据映射 |
| 2 | 2.1 Intro to CUDA C++ | 理解 kernel、启动、分配、拷贝和错误处理 | 完成首个可验证的 CUDA 程序 |
| 3 | 2.3 Writing SIMT Kernels | 学习线程协作、访存和同步 | transpose、reduction |
| 4 | 2.5 Asynchronous Execution | 理解执行顺序与异步依赖 | stream/event、串行与重叠对照 |
| 5 | 3.2 Advanced Kernel Programming | 进入高级 kernel 编程 | 在已有算子上验证进阶机制 |
| 6 | 4.2 CUDA Graphs | 了解图的构建、执行与相关约束 | 小算子链 eager/graph 对照 |
| 7 | 4.10 Asynchronous Barriers、4.11 Pipelines、4.12 Asynchronous Data Copies | 深入异步同步、流水线与数据搬运 | 修改 Hopper 相关示例 |
| 8 | 3.4 Programming Systems with Multiple GPUs | 建立多 GPU 编程概念 | 有多卡时做设备间传输实验 |

直接入口：[编程模型](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)、[CUDA C++ 入门](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)、[SIMT kernel](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)、[异步执行](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)。

### 第一遍的阅读边界

优先走通“分配 → 上传 → 启动 → 等待 → 下载 → 校验”。随后研究线程组织、内存访问和同步，再进入性能优化。

CUDA Python、Tile Kernels、Driver API、动态并行、图形互操作等内容可按任务补充。文档涵盖多代架构，不能把目录中的每项新功能都理解成 H200 上可用；实际实验应核对硬件能力和工具链要求。

## 2. 从第一周开始使用的配套资料

### 2.1 CUDA Best Practices Guide：怎样做性能实验

[官方文档](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

围绕计时、有效带宽、内存访问和执行配置查阅。它适合回答“怎样证明一次修改有效”，与主教材配合使用。

建议练习：

- 同时记录 GPU 时间和包含数据传输的端到端时间，明确测量边界。
- 比较连续与跨步访问，预测流量变化后再测量。
- 一次主要改变一个因素，例如 block 大小或访存方式。
- 区分算法读写字节数、缓存流量与实际 HBM 流量。

每次优化都保留基线、修改、正确性结果和性能数据。性能数字应附带 shape、dtype、layout、版本和测量方法。

### 2.2 Nsight Systems：整个程序的时间花在哪里

[官方 User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

重点学习 CUDA 时间线和 NVTX 标记。先观察 CPU 提交、数据传输、kernel 执行和同步之间的关系，再定位需要深入研究的部分。

适合的练习是把一个串行任务改为分段流水线，用时间线确认传输和计算是否重叠，并解释仍然存在的等待。进入推理项目后，用同样的方法分析 prefill/decode 之间的空隙。

### 2.3 Nsight Compute：一个 kernel 为什么慢

[官方 Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

在找到热点后分析对应 kernel，按问题选择计算、内存和执行资源指标。不要一开始就对整个模型收集全部指标。

训练时要求自己提出具体问题，例如：“访存是否高效？”“寄存器使用是否限制并发？”“增加数据复用是否带来收益？”采集可能涉及重放并影响程序行为，最终性能应在关闭 profiler 后重新测量。

### 2.4 Compute Sanitizer：结果正确之前先排除执行错误

[官方文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

从 memcheck 开始，再根据问题使用 racecheck、initcheck 和 synccheck。工具检查与数值校验相互补充：没有报错不等于算法和并发逻辑已经得到完整证明。

## 3. H200 专项资料

### 3.1 先确认实际硬件

[H200 官方规格](https://www.nvidia.com/en-us/data-center/h200/)

H200 属于 Hopper 架构，官方列出的显存为 141 GB HBM3e、带宽为 4.8 TB/s。SXM 与 NVL 在算力、功耗和部署形态上存在差异；使用 MIG 时，实际可用资源也不同。

做算力对照时，注意区分 dtype、Tensor Core 与普通计算路径，以及稠密和稀疏峰值。官网带有 sparsity 标注的数据不能直接作为普通稠密 GEMM 的比较基准。

### 3.2 Hopper Tuning Guide

[官方文档](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)

重点关注资源限制、TMA、thread block clusters 和访存优化。该指南大量使用 H100 描述 Hopper 架构；学习时把架构机制与 H200 的具体设备规格分开核对。

推荐在 shared memory、同步和基础 GEMM 已经掌握后阅读。TMA 的价值与异步数据搬运、同步方式和计算安排有关，建议用可运行示例观察机制，而不是只记住术语。

### 3.3 CUTLASS：学习优化 GEMM 的工程实现

- [CUTLASS Overview](https://docs.nvidia.com/cutlass/latest/overview.html)
- [Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)

从示例的输入 shape、数据类型和 kernel 配置开始，逐步理解 tile、数据布局、流水线与 warp specialization。建议路线是运行一个 Hopper 示例、建立基线，再改变一个配置并解释结果。

涉及 Hopper 架构专用指令的示例，应按示例要求设置编译目标，尤其注意 `sm_90` 与 `sm_90a` 的区别。目标是理解设计与约束，初期不要求手写 GEMM 超过 cuBLAS。

## 4. AI 算子与多卡实践入口

| 资料 | 学习用途 | 建议开始时间 |
|---|---|---|
| [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples) | 查找功能示例，学习初始化、调用和错误处理方式 | 第 1 周，按需选择 |
| [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) | 从 vector add、softmax、matmul 等例子学习另一种 kernel 表达方式 | CUDA 基础形成后 |
| [NCCL Tests](https://github.com/NVIDIA/nccl-tests) | 测量 collective 通信，理解消息大小与吞吐的关系 | 第 7 周，有多卡时 |

使用这些资料时，固定仓库版本或 commit，并记录配套工具链。不要把最新版示例默认视为与本机环境兼容。

AI Infra 项目还需要结合实际框架理解 attention、KV cache、prefill/decode 和调度。建议先把某个热点抽成小实验，再回到真实执行链验证收益。有单卡也能完成大部分基础训练，多卡实验可以后补。

## 5. 把资料安排到 8 周训练中

下表是建议训练顺序与学习量，属于计划，不代表已经完成实验或获得性能结果。每周按 6 天、每天约 3 小时安排，核心学习量 18 小时；可选加练 0～2 小时，周日用于轻量复盘或补课。

| 周次 | 主要资料 | 实验主题 | 本周验收产出 |
|---|---|---|---|
| 第 1 周 | 主教材 1.2、2.1；Best Practices；工具入门 | vector add、copy、边界处理、计时 | 可复用的正确性与测量程序 |
| 第 2 周 | 主教材 2.3；访存最佳实践 | 连续/跨步访问、transpose、reduction | 解释访存和线程组织对性能的影响 |
| 第 3 周 | 主教材与工具按问题查阅 | softmax、RMSNorm、融合、PyTorch 接入 | 多 shape/dtype 校验及性能对照 |
| 第 4 周 | GEMM 相关章节、CUTLASS | naive GEMM、tiling、Tensor Core 示例 | 与 cuBLAS 的同口径比较 |
| 第 5 周 | 主教材 2.5、4.2；Nsight Systems | stream/event、双缓冲、CUDA Graph | 时间线证实依赖与重叠关系 |
| 第 6 周 | Hopper Guide、CUTLASS、Triton | 修改 Hopper 示例、重写一个算子 | 解释搬运、计算和同步的配合 |
| 第 7 周 | 多 GPU 章节、NCCL Tests、真实框架源码 | attention/KV cache、推理 trace；多卡可加 DDP | 区分计算、访存、提交和通信瓶颈 |
| 第 8 周 | 围绕项目选择资料 | decode 执行链或融合算子优化 | 可复现的正确性与端到端收益报告 |

第 7 周只有单卡时，用单卡 attention、KV cache 和训练显存分析替代多卡实验。推进以验收结果为依据；基础未达标时优先补实验。

8 周的目标是建立独立实验与优化能力。之后用真实项目持续巩固，逐步覆盖更复杂的形状、并发、数值和系统约束。

## 6. 每篇资料都应落到一个可验证的问题

建议把每次学习记录压缩为以下结构：

| 记录项 | 内容 |
|---|---|
| 阅读范围 | 文档标题、章节与链接 |
| 待验证问题 | 例如：增加 shared memory 复用是否减少显存访问？ |
| 实验预测 | 预计哪个指标改变，原因是什么？ |
| 实验条件 | GPU、软件版本、shape、dtype、layout、编译参数 |
| 正确性 | 参考实现、误差、边界与特殊输入 |
| 性能证据 | 时间、带宽或吞吐；相关 profiler 指标 |
| 结论与限制 | 哪些场景有效，哪些退化，下一步验证什么？ |

小规模任务和大规模任务应分别测试。前者可能更容易暴露启动与提交开销，后者更容易观察带宽或计算限制。局部算子优化完成后，仍需回到端到端测量。

### 第一次阅读的最小清单

1. 阅读 1.2 Programming Model，能解释 host/device、thread/block/grid 与 warp。
2. 阅读 2.1 Intro to CUDA C++，对应到一个 vector add 程序。
3. 用 CPU 参考结果验证 GPU 输出，处理不能整除 block 大小的长度。
4. 用 CUDA Event 计时，并说明包含和排除了哪些工作。
5. 运行 Compute Sanitizer，记录结果。
6. 修改输入长度，预测并解释时间变化。

**先把主教材里的概念对应到代码，再用实验和工具确认自己的理解。**资料的作用是帮助提出和解决工程问题；每周留下可解释、可复现的结果，学习才会持续积累。
