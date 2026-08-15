---
layout: post
title: "CUDA Graph 源码课程 11：Dispatcher、BatchDescriptor、Padding 与真实多流"
description: "拆解 vLLM v0.22.1 的配置/运行模式、BatchDescriptor、FULL→PIECEWISE→NONE 路由，以及 offloader 与 NCCL 的多流约束。"
date: 2026-08-16 11:00:00 +0800
category: "vLLM · CUDA Graph 源码课程"
series: "vLLM CUDA Graph 源码课程"
tags: [vLLM, CUDA Graph, Dispatcher, BatchDescriptor, Multi-Stream, NCCL]
reading_time: "约 24 分钟"
mermaid: true
permalink: /courses/cuda-graph/11-dispatcher-padding-multistream/
---

上一课：[一张真实 FULL Graph](/courses/cuda-graph/10-vllm-full-graph-trace/) · [课程目录](/courses/cuda-graph/) · 下一课预告：Physical Beam Geometry 与 Persistent Suffix KV

> 源码基线：[vLLM v0.22.1 / `0decac0d`](https://github.com/vllm-project/vllm/commit/0decac0d96c42b49572498019f0a0e3600f50398)。当前版本的 `ForwardContext` 没有后续主线中的 `is_padding` 字段；图相关字段是 `cudagraph_runtime_mode` 与 `batch_descriptor`。

第 11 课把前面的 Stream、Event、静态地址和生命周期收束到 vLLM 0.22.1 的真实路由。核心问题不是“这个 batch 大小是否见过”，而是：**它能否安全复用某个已捕获的执行拓扑。**

## 一、先纠正两个名字上的误区

第一，配置层有五种 `CUDAGraphMode`：`NONE`、`PIECEWISE`、`FULL`、`FULL_DECODE_ONLY`、`FULL_AND_PIECEWISE`；runtime 真正传给 wrapper 的只有三种具体模式：`NONE`、`PIECEWISE`、`FULL`。后两个复合模式只是把 uniform decode 与 mixed batch 映射到不同 runtime mode，不会作为 replay 模式出现。

第二，vLLM 0.22.1 **没有 `GraphKey` 类**。`CudagraphDispatcher` 按 runtime mode 保存 `set[BatchDescriptor]`，`CUDAGraphWrapper` 内部再用 `dict[BatchDescriptor, CUDAGraphEntry]` 找具体实例。因此可以把有效身份理解为：

```text
(wrapper identity / runtime mode, BatchDescriptor)
```

“GraphKey”适合作为架构概念，但不能伪装成这个版本的源码类名。

## 二、BatchDescriptor 到底描述什么

源码中的 frozen dataclass 只有五个字段：

| 字段 | 含义 |
|---|---|
| `num_tokens` | padding 后的物理 token 容量 |
| `num_reqs` | request 数；PIECEWISE 可为 `None` |
| `uniform` | 所有请求是否具有同一 query length |
| `has_lora` | 本批是否启用 LoRA |
| `num_active_loras` | 不同 active adapter 的数量 |

假设 capture sizes 为 `[1, 2, 4, 8]`，真实 token 数为 3，dispatcher 会归一到 `num_tokens=4`。padding 解决的是“物理容量有限化”，不是“语义自动等价”：真实 token、position、mask、block table 与 slot mapping 仍要写入固定 buffer；多出的槽必须被 mask 或安全哨兵处理。

FULL key 更严格，因为 attention 也在图内。`num_reqs`、uniform geometry 可能影响 scheduler metadata、kernel 数量或 workspace。PIECEWISE 把不兼容 attention 留在图外，所以初始化 key 时会执行：

```python
replace(batch_desc, num_reqs=None, uniform=False)
```

注意它只放松这两个字段，**不会删除 `has_lora` 和 `num_active_loras`**。因此“PIECEWISE 不支持 LoRA”不是 0.22.1 的普遍结论；LoRA 是否命中取决于配置生成了哪些捕获 case，以及 active adapter 数如何特化/向上归一。

`uniform=True` 也不是所有 FULL 配置都保留。只有 `FULL_DECODE_ONLY` / `FULL_AND_PIECEWISE` 这类具有独立 decode routine 的复合策略，uniform decode 才用专门 FULL key；纯 `FULL` 会把 uniform 归一为 `False`，复用更通用的 FULL entry。这里的 `uniform` 指每个 request 的 query length 相同，不是总 token 数固定。

## 三、哪些是内容，哪些必须进 key 或回退

| 变化 | 正确归属 |
|---|---|
| token id、position、有效长度、固定宽度 block table/slot mapping 的内容 | 写入静态 device buffer |
| padded capacity、request geometry、active LoRA 数 | descriptor / entry 选择 |
| attention plan 改变 kernel、grid、workspace 或参数解释 | 扩 key、换图，或回退 |
| collective 类型、数量、顺序变化 | 不得复用旧图；所有 rank 一致地换路由 |
| model、device、dtype、backend、TP 拓扑变化 | 通常是 manager 生命周期不变量，重建整组图 |

判断标准不是“Tensor shape 看起来一样”，而是执行 DAG 与节点参数解释是否仍兼容。同形状但 attention backend 在 Host 侧换了 plan，仍可能静默算错。

## 四、AttentionCGSupport 如何约束路由

backend builder 暴露四级能力，数值从弱到强为：

```text
NEVER < UNIFORM_SINGLE_TOKEN_DECODE < UNIFORM_BATCH < ALWAYS
```

多个 attention group/backend 并存时，`GPUModelRunner._check_and_update_cudagraph_mode()` 取最弱能力，再由 `resolve_cudagraph_mode_and_sizes()` 降级配置。`NEVER` 表示 attention 不能进入 FULL Graph；若 attention 已作为 split op 且使用 piecewise compilation，其他安全分区仍可走 PIECEWISE。否则才退到 `NONE`。所以“不支持 FULL”与“完全不能用 CUDA Graph”不是同一句话。

Speculative decode 的 uniform query length 是 `1 + num_speculative_tokens`。因此只声明 `UNIFORM_SINGLE_TOKEN_DECODE` 的 backend 不能据此覆盖 `q>1`；要保留 FULL decode 至少需要 `UNIFORM_BATCH`，否则继续降到 PIECEWISE 或 `NONE`。

runtime dispatch 的次序是：

1. 生成 padded descriptor；
2. 在允许模式中先查精确 FULL；
3. 再查放松 `num_reqs/uniform` 的 PIECEWISE；
4. 都不命中则返回 `NONE`。

这里的 `FULL → PIECEWISE → NONE` 是查找优先级，不保证三层 fallback 在每种配置中都存在。例如纯 `FULL` 配置若本轮因 cascade attention 禁止 FULL，又没有初始化 PIECEWISE key，就会直接落到 `NONE`。

设策略为 `FULL_AND_PIECEWISE`、capture sizes 为 `[1, 2, 4, 8]`、`max_num_seqs=8`、无 LoRA，可用三个 case 检查自己的预测：

| Runtime batch | 归一化与查找 | 结果 |
| --- | --- | --- |
| 3 token / 3 req，全部 `query_len=1` | token pad 到 4，FULL descriptor 为 `(4, 4, uniform=True)` | FULL |
| 3 token / 2 req，query lens 为 `[1,2]` | FULL uniform key miss；放松为 `(4, None, False)` | PIECEWISE |
| 9 token | 超过最大 capture size，不再 padding | NONE，保留 raw token 数 |

FULL 中的 `num_reqs=4` 是 padded physical capacity，不是凭空多出一个业务请求；attention metadata 与 mask 必须保证虚拟容量不产生语义影响。

这与 capture 次序恰好不同：`get_capture_descs()` 返回 PIECEWISE → FULL，且每组 large → small，以利于共享 pool 和降低碎片。

## 五、Graph、capture stream 与 replay stream

Graph 不是 Stream。capture stream 只负责记录 DAG；`replay()` 会把整张图提交到调用时的 current stream。于是：

- 在 `Sc` capture、在 current `Se` 调用 replay，本次图提交到 `Se`；
- `static_input.copy_()` 与 `replay()` 都在 `Se`，同流 FIFO 已足够，无需额外 Event；
- staging 在 `Sin`、replay 在 `Se`，消费者 `Se` 必须等待生产者 `Sin`，例如 `Se.wait_stream(Sin)` 或等待由 `Sin` 记录的 Event。

“图内所有辅助流必须在图结束前 join”指的是捕获 DAG 必须闭合。标准结构是：origin stream 记录 fork Event → auxiliary stream 等待并开始工作 → auxiliary 记录 done Event → origin 等待 done → 才能结束 capture。最后一条 wait 不是 Host 阻塞，而是把依赖边放进图。漏掉它会产生 unjoined capture（CUDA 错误 904）；拿 capture 外的未完成 Event 跨隔离边界等待，还可能触发 isolation 错误 905。

## 六、`join_after_forward` 的具体实现

`PrefetchOffloader` 是最完整的真实案例。它创建 `copy_stream` 并预分配 `StaticBufferPool`；这里的索引是 `module_offloaders` 中被选中的 offload module 序号，不等同于原模型层号，静态槽按其 `slot_idx % prefetch_step` 复用。对应 hook 的顺序是：等待当前模块权重 → 执行 forward → 启动后续模块预取。

`start_onload_to_static()` 做四件事：

1. 在 compute current stream 记录临时 fork Event；
2. `copy_stream.wait_event(fork_event)`，把 copy stream 接入 capture；
3. 在 copy stream 向静态 GPU buffer 发起 non-blocking H2D；
4. 在 copy stream 记录 `_copy_done_event`。

正常情况下，未来某层的 `_wait_for_layer()` 会让 compute stream 等待对应 done Event，形成 join。但最后几层启动的“下一轮预取”，在本轮 forward 结束时还没有消费者来等待；若此刻直接结束 capture，copy stream 就是悬空分支。`join_after_forward()` 遍历所有 `_prefetch_in_capture=True` 的 layer，让 origin stream 等待其 `_copy_done_event`，然后清 flag，正是对这批尾部叶子做最终 join。

这里的 `_prefetch_in_capture`、`_event_valid_for_eager` 是 **capture 时的 Python 记账**。replay 不会再运行这段 Python 循环；它重放的是已经记录进 Graph 的 Event、memcpy 和 wait 节点。capture/replay 前的 `sync_prev_onload()` 则负责排空来自图外 eager 阶段的旧 copy，避免把外部 Event 错接进捕获域。

另外，`wait_prefetch(input_tensor)` 和 `start_prefetch(output_tensor)` 注册成带 `mutates_args` 的 custom op：Tensor mutation 让 `torch.compile` 看见先后关系；CUDA Event 让设备的两条 Stream 真正形成 happens-before。两层缺一不可。

## 七、NCCL 不需要凭空再造一条通信流

若 collective 在模型 forward 的 current capture stream 上发射，且 NCCL/CUDA 版本与调用受支持，它会成为 Graph 节点；不必为了“看起来并行”额外创建 comm stream。真正的硬约束是所有参与 rank 对 communicator、collective 类型、count、dtype、reduction op/root（适用时）、buffer/address 契约、调用顺序及兼容 runtime mode 达成一致；参与 capture 的每个 rank 都必须 replay 含有对应 collective 的图。rank 0 replay FULL、rank 1 走 eager 或少一次 collective，不是性能问题，而是卡死或错误的来源。

## 本课验收

1. 配置为 `FULL_AND_PIECEWISE` 时，runtime 为什么不会出现同名 mode？
2. 相同 padded token 数、不同 request 数，为什么 FULL 可能不是同一个 entry，而 PIECEWISE 可以复用？
3. `AttentionCGSupport.NEVER` 为什么仍可能允许 PIECEWISE？
4. Graph 在 `Sc` capture、在 `Se` replay，实际提交到哪条流？
5. `join_after_forward()` 加入图的是 Python 循环，还是 Event/wait 节点？
6. 两个 rank 选择不同路由时，为什么不能靠某个 rank 单独加 synchronize 修复？

## 源码与文档

- [CUDAGraphMode：五种配置模式、三种 runtime mode](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/config/compilation.py)
- [BatchDescriptor 与 ForwardContext](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/forward_context.py)
- [CudagraphDispatcher：padding、LoRA case、FULL/PIECEWISE 路由](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/cudagraph_dispatcher.py)
- [AttentionCGSupport](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/attention/backend.py)
- [PrefetchOffloader：fork、wait 与 join_after_forward](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/model_executor/offloader/prefetch.py)
- [Prefetch custom ops：compiler-visible ordering](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/model_executor/offloader/prefetch_ops.py)
- [NCCL CUDA Graphs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)

上一课：[一张真实 FULL Graph](/courses/cuda-graph/10-vllm-full-graph-trace/) · [返回课程目录](/courses/cuda-graph/)

下一课将从一个新矛盾开始：BatchDescriptor 描述的是原生 batch 几何；Beam FULL Graph 还必须先把 logical request 展开成可准入、可持久化的 physical rows。
