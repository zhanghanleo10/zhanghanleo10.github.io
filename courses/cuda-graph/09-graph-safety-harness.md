---
layout: post
title: "CUDA Graph 源码课程 09：建立通用 Graph-Safety Harness"
description: "把地址、拓扑、副作用与生命周期变成可自动证伪的断言，统一审计 CUDA、Triton、Attention 与 Beam 算子。"
date: 2026-08-16 09:00:00 +0800
category: "vLLM · CUDA Graph 源码课程"
series: "vLLM CUDA Graph 源码课程"
tags: [CUDA Graph, Testing, Graph Safety, Attention, Correctness]
reading_time: "约 17 分钟"
mermaid: true
permalink: /courses/cuda-graph/09-graph-safety-harness/
---

上一课：[PyTorch CUDAGraph](/courses/cuda-graph/08-pytorch-cudagraph/) · [课程目录](/courses/cuda-graph/) · 下一课：[一张真实 FULL Graph](/courses/cuda-graph/10-vllm-full-graph-trace/)

CUDA Graph 能成功 capture，只能证明“这一次没有立刻报错”，不能证明它能安全 replay。真正要验证的是一个长期契约：相同执行签名下，地址、形状、布局和执行拓扑保持兼容；每次变化只通过静态 buffer 的内容进入；副作用、输出寿命和跨流依赖在重复 replay 后仍然正确。

这也是本课要建立 Graph-Safety Harness 的原因：把“看起来能成图”变成一组可以被自动证伪的断言。

## 一、先定义 graph-safe 的最小不变量

对一个算子或模型片段，至少要核对四层不变量：

| 层次 | 必须稳定或受控的内容 |
|---|---|
| Tensor 契约 | `data_ptr`、shape、stride、dtype、device、layout |
| 执行契约 | kernel 数量/顺序、grid、算法 plan、workspace 容量、Stream/Event DAG |
| 副作用契约 | KV 写入位置、RNG 推进、原子更新、collective 顺序 |
| 生命周期契约 | 静态输入、输出、workspace、Graph pool 在最后一次 replay 完成前存活 |

“输出不是纯函数”并不等于不能成图。Attention 可以写 KV，NCCL 可以通信，状态 Tensor 也可以原地更新；关键是副作用的目标地址、先后关系和解释方式稳定。反过来，一个数学上纯粹的算子若每轮扩容 workspace，仍然不安全。

## 二、Harness 的固定流程

对任何 CUDA、Triton、Attention 或 Beam 算子，都执行同一套十步测试：

1. 用 eager 路径计算 reference；
2. 在 side stream 做多轮 warmup，排除 lazy init、JIT 和 autotune；
3. 预分配静态输入、输出、metadata、state 和 workspace；
4. capture 一次；
5. 用相同签名、不同内容连续 replay；
6. 每轮前后检查地址、shape、stride、dtype 和 device 哨兵；
7. 比较 eager 与 graph 的输出及副作用状态；
8. 记录 replay 前后的 allocator 变化；
9. 对不兼容签名执行 fail-closed：换图或回退 eager，绝不“凑合 replay”；
10. 销毁 owner 后做生命周期负测，确认不会 use-after-free。

下面是最小骨架。为便于测试，它故意包含 `state += output` 这个副作用：

```python
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class StaticContract:
    ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def of(cls, x):
        return cls(x.data_ptr(), tuple(x.shape), tuple(x.stride()),
                   x.dtype, x.device)

def step(x, scale, state, out):
    torch.mul(x, scale, out=out)
    state.add_(out)

static_x = torch.empty((16,), device="cuda")
static_scale = torch.empty((), device="cuda")
static_state = torch.zeros_like(static_x)
static_out = torch.empty_like(static_x)
static_x.zero_()
static_scale.fill_(1.0)

# warmup 后再恢复初始 state
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for _ in range(3):
        step(static_x, static_scale, static_state, static_out)
torch.cuda.current_stream().wait_stream(side)
static_state.zero_()

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    step(static_x, static_scale, static_state, static_out)

# capture 只记录、不执行；确保首次 replay 前仍是业务初态
static_state.zero_()
expected_state = torch.zeros_like(static_state)

contracts = {"x": StaticContract.of(static_x),
             "state": StaticContract.of(static_state),
             "out": StaticContract.of(static_out)}

for i in range(1, 20):
    runtime_x = torch.full_like(static_x, i)
    static_x.copy_(runtime_x)       # 只改内容，不换捕获地址
    static_scale.fill_(i + 0.5)
    g.replay()
    expected_state.add_(runtime_x * (i + 0.5))
    assert contracts["x"] == StaticContract.of(static_x)
    assert contracts["state"] == StaticContract.of(static_state)
    assert contracts["out"] == StaticContract.of(static_out)
    torch.testing.assert_close(static_out, runtime_x * (i + 0.5))
    torch.testing.assert_close(static_state, expected_state)
```

这里 `copy_()` 与 `replay()` 在同一 current stream，FIFO 已建立依赖；若 staging 在 `Sin`、replay 在 `Se`，必须让消费者 `Se` 等待生产者 `Sin`。

## 三、三个最容易漏掉的负向实验

第一，**旧数据但不报错**。把 `static_x.copy_(runtime_x)` 删除，Graph 仍可 replay，只是一直读取捕获地址中的旧内容。因此“指针不变”是必要条件，不是充分条件。

第二，**静态输出别名**。`static_out` 每轮都是同一块内存：

```python
old = static_out          # 只是别名
snapshot = static_out.clone()
g.replay()
# old 已被新一轮覆盖；snapshot 才保存上一轮结果
```

第三，**签名 miss 被错误吞掉**。shape 从 `(16,)` 变成 `(17,)`，不能把新 Tensor 直接交给旧图。Harness 应在 replay 前拒绝该请求，并验证上层确实选择新 bucket 或 eager。

测试代码里可以自定义一个 `GraphKey` 数据类表示“本项目的准入签名”，但要注意：这只是 Harness 抽象，**不是 vLLM 0.22.1 源码中的同名类**。vLLM 的真实 dispatch 身份将在第 11 课拆解。

## 四、拿到新算子时先问什么

- capture 前是否完成 handle 创建、autotune、JIT 和 communicator 初始化？
- 是否隐藏 `.item()`、D2H、Host callback 或依赖设备值的 Python 分支？
- 每轮是否 `empty`、resize，或按输入扩 workspace？
- metadata 是否来自固定 device buffer，还是捕获时临时对象？
- 重复 replay 时 KV、RNG、scratch、原子写和 collective 的语义是否仍正确？
- unsupported 条件是否显式回退？

如果这些问题不能逐条回答，就还没有资格做性能比较。先让 Harness 尽可能破坏它，再谈 replay 节省了多少 launch overhead。

## 本课验收

1. 为什么 capture 成功不能证明 graph-safe？
2. 指针保持不变，为什么仍可能读到旧数据或得到错误结果？
3. 为什么保存静态输出引用会被下一次 replay 覆盖？
4. 对一个会写 KV 的 Attention 算子，至少要补哪些副作用断言？
5. shape miss 时，正确策略为什么是新 key、重捕获或 eager，而不是修改旧图的输入对象？

能够在运行前列出一个新算子的全部假设，并用上述 Harness 逐条证明或证伪，才算通过。

## 源码与文档

- [PyTorch CUDA Graphs 语义](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [vLLM 0.22.1：CUDAGraphWrapper 的地址检查与生命周期](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/compilation/cuda_graph.py)
- [vLLM 0.22.1：Attention metadata 与 graph support](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/attention/backend.py)
- [vLLM 0.22.1：GPUModelRunner 的静态输入与 capture 路径](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/worker/gpu_model_runner.py)

上一课：[PyTorch CUDAGraph](/courses/cuda-graph/08-pytorch-cudagraph/) · 下一课：[一张真实 FULL Graph](/courses/cuda-graph/10-vllm-full-graph-trace/)
