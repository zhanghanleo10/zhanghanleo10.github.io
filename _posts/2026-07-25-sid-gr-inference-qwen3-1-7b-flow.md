---
layout: post
title: "SID-GR Inference 中 Qwen3-1.7B 的端到端推理流程：从 HTTP 请求到 GR Decode Attention"
description: "沿着 NVIDIA recsys-examples 的真实代码路径，拆解 Qwen3-1.7B 在 SID-GR Inference 中的模型加载、Continuous Batching、Prefill、ContextKV、BeamKV、BeamPath、专用 Decode Attention、TopK 与结果回溯。"
date: 2026-07-25 14:30:00 +0800
category: "生成式推荐"
tags:
  - SID-GR
  - Qwen3
  - Beam Search
  - KV Cache
  - Continuous Batching
  - CUDA Graph
reading_time: "约 35 分钟"
math: false
---

> 本文分析 NVIDIA [`recsys-examples`](https://github.com/NVIDIA/recsys-examples) 仓库中 `examples/sid-gr-inference` 的真实实现，重点关注 **Batch 怎样形成、Beam Search 怎样重排、Prefill/Decode 怎样流动，以及专用 GR Decode Attention 怎样消费两段 KV**。分析基于提交 [`bdf16e3`](https://github.com/NVIDIA/recsys-examples/tree/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference)，避免后续代码变化导致文中路径和行为无法对应。

这套系统面对的不是典型聊天模型工作负载，而是：

```text
长用户上下文
+ 极短的 Semantic ID 生成
+ 很大的 Beam Width
```

例如：

```text
context_len = 1000 / 5000
beam_width = 128 / 256
output length = 3 / 5
```

如果直接使用普通 LLM Serving 的思路，把 `batch × beam_width` 展平成大量独立序列，就会重复保存和访问相同的长上下文 KV。SID-GR Inference 的核心设计，是把长上下文和短 Beam 历史彻底分开：

```text
ContextKV：每个请求的长上下文，只保存一次，所有 Beam 共享
BeamKV：每个 Beam 的短 Decode 历史，按 step × beam 保存
BeamPath：保存 Beam 每一步的父子关系
```

这三个数据结构共同决定了整个推理流程。

---

## 1. 先看完整调用链

生产 HTTP Serving 路径可以概括为：

```text
POST /generate
    ↓
GRHTTPServingAdapter
    ↓
GRServingRequest
    ↓
GRServingWorker / GRInProcessServingFacade
    ↓
GRContinuousServingExecutor.submit
    ↓
GRContinuousScheduler.waiting_prefill
    ↓
Scheduler Tick
    ├── Prefill Admission
    │     ↓
    │   Qwen3GRModel.forward_prefill
    │     ↓
    │   ContextKV + Last-token Logits
    │     ↓
    │   Initial TopK
    │     ↓
    │   初始化 BeamPath / BeamKV
    │
    └── Decode Batch Planning
          ↓
        构造 beam_token_ids
          ↓
        根据 BeamPath 构造 topk_indices
          ↓
        Qwen3GRModel.forward_decode_step
          ↓
        每层写 BeamKV + GR Decode Attention
          ↓
        LM Head 得到 [B, W, V] Logits
          ↓
        Next TopK / 更新 BeamPath
          ↓
        达到停止条件后回溯完整 SID
          ↓
        beam_results / item_results
```

对应的核心文件：

- [`tools/serve_qwen3_gr_http.py`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/tools/serve_qwen3_gr_http.py)：HTTP 服务装配入口；
- [`tools/run_qwen3_real_weight_serving.py`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/tools/run_qwen3_real_weight_serving.py)：模型加载与离线/在线运行入口；
- [`gr_serving/continuous.py`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/src/gr_inference/gr_serving/continuous.py)：Continuous Scheduler 与 Executor；
- [`gr_models/qwen3/model.py`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/src/gr_inference/gr_models/qwen3/model.py)：Qwen3 模型级 Prefill / Decode；
- [`gr_models/qwen3/layers.py`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/src/gr_inference/gr_models/qwen3/layers.py)：Qwen3 单层计算；
- [`gr_runtime/batched_beam_search.py`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/src/gr_inference/gr_runtime/batched_beam_search.py)：Batch-aware Initial/Next TopK；
- [`gr_runtime/batched_topk_indices.py`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/src/gr_inference/gr_runtime/batched_topk_indices.py)：把 Beam 父链编译为算子索引；
- [`gr_kernels/attention/existing_kernel_backend.py`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/src/gr_inference/gr_kernels/attention/existing_kernel_backend.py)：外部 `gr-decode_atten` Kernel 适配；
- [`corelib/gr_decode_atten/interface.py`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/corelib/gr_decode_atten/interface.py)：专用 Attention 的 Context、Sparse Beam、LSE Combine 与 Fused 路径；
- [`corelib/gr_decode_atten/src/decode/flash_fwd.py`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/corelib/gr_decode_atten/src/decode/flash_fwd.py)：Sparse Beam KV Gather 与 Decode Attention Kernel。

---

## 2. Qwen3-1.7B 在这里是什么模型

仓库当前默认模型是：

```text
Qwen/Qwen3-1.7B
```

在 [`variants.py`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/src/gr_inference/gr_models/qwen3/variants.py) 中，它的主要结构参数是：

| 参数 | 数值 |
| --- | ---: |
| Transformer 层数 | 28 |
| Hidden Size | 2048 |
| Intermediate Size | 6144 |
| Query Heads | 16 |
| KV Heads | 8 |
| Head Dim | 128 |
| Vocabulary Size | 151936 |
| Embedding / LM Head | 权重共享 |

这里并没有直接调用 Hugging Face 的 `Qwen3ForCausalLM.forward()`，而是自行构建了一个 `Qwen3GRModel`，再把 Hugging Face Checkpoint 转成内部逻辑权重。

模型加载链路为：

```text
resolve_model_dir
    ↓
HFCheckpointLoader.manifest
    ↓
Qwen3GRConfig.from_hf_config
    ↓
materialize_qwen3_checkpoint
    ↓
Qwen3GRModel(...)
    ↓
model.load_logical_weights(weights)
    ↓
model.eval()
```

### 2.1 为什么要先 materialize logical weights

[`weights.py`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/src/gr_inference/gr_models/qwen3/weights.py) 会把 Hugging Face 权重映射为框架内部名称，并做两类拼接：

```text
q_proj + k_proj + v_proj
    -> qkv_proj

gate_proj + up_proj
    -> gate_up_proj
```

于是每层可以使用一次 Packed QKV Projection 和一次 Packed Gate/Up Projection。Runtime 不需要理解不同模型仓库的权重命名，模型差异被限制在 Adapter 层。

---

## 3. HTTP 请求怎样进入 Runtime

HTTP 服务启动时依次组装：

```python
model, config, device = load_model(args, torch)

decode_engine = GRDecodeEngine(
    attention=GRDecodeAttention(
        backend=make_decode_backend(args, device)
    ),
    fixed_beam_width=args.beam_width,
)

engine = GRServingEngine(
    model=model,
    decode_engine=decode_engine,
    config=GRServingConfig(...),
)

scheduler = GRContinuousScheduler(...)
executor = GRContinuousServingExecutor(
    engine=engine,
    scheduler=scheduler,
    beam_kv_pool=beam_kv_pool,
    context_kv_pool=context_kv_pool,
)
```

`/generate` 当前要求调用方直接传入 `input_ids`。框架内部请求对象为：

```python
GRServingRequest(
    request_id=request_id,
    input_ids=input_ids,
    max_decode_steps=max_decode_steps,
    beam_width=beam_width,
    metadata=metadata,
    beam_width_policy=beam_width_policy,
    stop_token_ids=stop_token_ids,
    logits_processors=logits_processors,
)
```

其中单请求输入必须是：

```text
[S]
或
[1, S]
```

### 3.1 `max_new_tokens` 与 `max_decode_steps`

SGLang 兼容 `/generate` 接口会执行：

```python
max_new_tokens = ...
gr_decode_steps = max(1, max_new_tokens - 1)
```

原因是第一枚输出 token 直接由 Prefill 的最后位置 Logits 做 Initial TopK 得到；后续 token 才需要执行真正的模型 Decode Step。

因此对于常见的：

```text
max_new_tokens = 3
```

内部语义是：

```text
Prefill Initial TopK：产生第 1 枚 token
Decode Step 0：产生第 2 枚 token
Decode Step 1：产生第 3 枚 token
```

最终响应还会按 `requested_max_new_tokens` 截断输出，保证 HTTP 语义与调用方一致。

这里有一个边界细节：当 `max_new_tokens=1` 时，内部仍会保留至少一个 Decode Step 的容量，所以使用的是 `max(1, N-1)`，而不是严格的 `N-1`。HTTP 输出层再截断为调用方请求的长度。对常见的三级、五级 SID，内部有效关系就是：

```text
BeamPath Entry 数 = 1 个 Initial TopK + max_decode_steps 个 Decode TopK
```

---

## 4. Continuous Scheduler 管理的不是普通 Sequence

每个请求由 `GRContinuousRequestState` 管理，阶段只有三种：

```text
waiting_prefill
    ↓
decoding
    ↓
finished
```

状态中除了请求本身，还保存：

```text
current_decode_step
active_beam_width
generation
beam_kv_pool_lease
context_kv_pool_lease
decode_selection_token_ids
decode_selection_scores
decode_parent_history
```

Scheduler 只负责状态机和 Batch 规划，真正的模型执行由 `GRContinuousServingExecutor` 提供回调：

```python
scheduler.tick(
    prefill_executor=self._run_prefill,
    decode_executor=self._run_decode_batches,
)
```

一个 Tick 可以理解为：

```text
1. 从 waiting_prefill 中接纳一批请求
2. 为请求分配 KV Lease
3. 执行 Prefill
4. 对 decoding 请求规划 Decode Microbatch
5. 每个请求推进一个逻辑 Decode Step
6. 完成请求并释放 KV Lease
```

### 4.1 系统里其实有三层 Batch

“组 Batch”不能只看一个 `torch.cat`。这套实现里有三层含义不同的 Batch：

| 层次 | 何时发生 | 分组条件 | 目的 |
| --- | --- | --- | --- |
| 请求积累 | HTTP Worker | 到达时间与队列状态 | 把并发请求交给同一个 Scheduler |
| Prefill Microbatch | 每个 Scheduler Tick | KV 预算允许，且执行时 `input_ids.shape` 相同 | 合并长 Context 的模型前向 |
| Decode Microbatch | 每个 Scheduler Tick | `step, W_current, W_next, context_len` 相同 | 合并相同专用 Attention Shape 的一步 Decode |

Scheduler 的 `max_prefill_batch_size` 和 `max_decode_batch_size` 是两套独立上限。请求被 Prefill 准入后立即进入 `decoding` 字典；随后同一个 Tick 的 `_plan_decode_batches()` 会看到这些新请求。因此在正常生产路径里：

```text
Tick N:
  admit request
  → run prefill
  → initial topK
  → 立刻进入 decode step 0
```

并不是先用一个 Tick 只做 Prefill，再等下一个 Tick 才做 Decode。

Prefill 准入还受 KV Memory Budget 控制。Scheduler 在弹出队首请求后，先检查：

```text
running requests
context tokens
beam slots = max_decode_steps × beam_width
```

如果当前资源不够，请求会被放回队首等待；如果系统里没有任何运行请求，而单个请求仍超预算，则直接报错，而不是永久饥饿。

### 4.2 Decode Batch 的分组键

正在 Decode 的请求按以下四个维度分组：

```python
(
    current_decode_step,
    active_beam_width,
    next_beam_width,
    context_len,
)
```

也就是说，只有满足以下条件的请求才进入同一个 Decode Batch：

```text
Decode Step 相同
+ 当前 Beam Width 相同
+ 下一步 Beam Width 相同
+ Context Length 相同
```

真正的执行单位是：

```text
Request Batch × Active Beams
```

而不是：

```text
把 B × W 展平成 B×W 个独立请求
```

这是 SID-GR Inference 与通用 LLM Serving 在抽象层面的关键差异。

例如当前有六个请求：

| Request | Step | 当前 W | 下一步 W | Context |
| --- | ---: | ---: | ---: | ---: |
| A | 0 | 256 | 256 | 1000 |
| B | 0 | 256 | 256 | 1000 |
| C | 0 | 256 | 128 | 1000 |
| D | 1 | 256 | 256 | 1000 |
| E | 0 | 256 | 256 | 5000 |
| F | 0 | 256 | 256 | 1000 |

那么逻辑分组是：

```text
(0, 256, 256, 1000) → [A, B, F]
(0, 256, 128, 1000) → [C]
(1, 256, 256, 1000) → [D]
(0, 256, 256, 5000) → [E]
```

之后每组再按照 `max_decode_batch_size` 切块。这样做看似保守，却换来了统一的 `decode_nums`、Beam 维度和 Context 长度，使 Dense Pool View、CUDA Graph 和专用 Attention 都可以使用固定 Shape。

---

## 5. Prefill：长上下文只计算和保存一次

Executor 首先按 `input_ids.shape` 对请求分组：

```python
requests_by_shape[input_ids.shape].append(request)
```

同 Shape 请求拼接为：

```python
input_ids = torch.cat(
    [request.input_ids for request in requests],
    dim=0,
)
```

假设：

```text
B = 4
S = 1000
```

则输入为：

```text
input_ids: [4, 1000]
```

随后调用：

```python
model.forward_prefill(
    input_ids,
    context_kv=context_kv,
    return_result=True,
    last_token_logits_only=True,
)
```

### 5.1 Prefill Batch 的形成条件

`_run_prefill()` 会先按完整 `input_ids.shape` 分桶。因此 `[1, 1000]` 与 `[1, 5000]` 不会进入同一次 Prefill Forward，即使它们在同一个 Scheduler Tick 被准入。每个桶内部才执行：

```text
[1, S] + [1, S] + ... → torch.cat(dim=0) → [B, S]
```

如果配置了 `GRDenseContextKVPool`，Executor 会尝试一次性为这批请求申请连续 Slot：

```text
Pool: [layers, capacity, max_context_len, kv_heads, head_dim]
                  └──────── B 个连续 slot ────────┘
```

申请成功后，模型直接把 Prefill K/V 写入 Pool Slice；失败时才让模型自己分配普通 `ContextKV`。连续 Slot 很重要，因为后续 Decode Batch 能把多个 request-local `[L,1,S,Hkv,D]` View 恢复成零拷贝 `[L,B,S,Hkv,D]` View。

Prefix Cache 是另一条可选路径：

```text
exact hit  → 直接复用缓存的 PrefillResult
prefix hit → 只计算 suffix，并物化完整 ContextKV
miss       → 执行完整 Batched Prefill
```

默认生产 benchmark 可以关闭 Prefix Cache，以便测量不依赖前缀复用的稳定性能；它不改变后续 Beam Search 语义。

### 5.2 ContextKV 的布局

[`ContextKV`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/src/gr_inference/gr_kv/context_kv.py) 的布局是：

```text
[layers, batch, context_len, kv_heads, head_dim]
```

对于 Qwen3-1.7B：

```text
[28, B, S, 8, 128]
```

这里没有 Beam 维度。无论 `beam_width` 是 16、128 还是 256，同一个请求的长上下文都只保存一份。

这正是该方案最直接的内存收益来源：

```text
长的部分不按 Beam 复制
短的部分才按 Beam 保存
```

### 5.3 Qwen3 Prefill 的逐层过程

`Qwen3GRModel.forward_prefill()` 执行：

```text
input_ids
  ↓
Embedding
  ↓
28 × Qwen3 Decoder Layer
  ↓
Final RMSNorm
  ↓
LM Head
```

每一层内部为：

```text
Hidden States
  ↓
Input RMSNorm
  ↓
Packed QKV Projection
  ↓
Q Norm / K Norm
  ↓
RoPE
  ↓
把 K/V 写入 ContextKV
  ↓
Prefill Attention
  ↓
O Projection
  ↓
Residual + Post-Attention RMSNorm
  ↓
Gate/Up Projection
  ↓
SiLU(gate) × up
  ↓
Down Projection
  ↓
Residual
```

Q、K、V 的形状分别为：

```text
Q: [B, S, 16, 128]
K: [B, S,  8, 128]
V: [B, S,  8, 128]
```

### 5.4 Serving 路径只算最后位置 Logits

生产路径设置：

```python
last_token_logits_only=True
```

所以最终投影的是：

```python
logits_input = hidden_states[:, -1, :]
logits = lm_head(logits_input)
```

输出形状为：

```text
[B, vocab_size]
```

而不是：

```text
[B, context_len, vocab_size]
```

SID-GR 只需要基于完整用户上下文选择第一层 Semantic ID，没有必要物化所有上下文位置的 Logits。

### 5.5 Prefill CUDA Graph 在哪里介入

Executor 先尝试 `GRPrefillCudaGraphRunner.forward_prefill()`；如果当前 Shape 没有可复用 Graph，再回退到普通 `model.forward_prefill()`。Piecewise Graph 并没有改变 Transformer 语义，只把 Embedding、若干 Layer Chunk 和 Output 投影拆成稳定片段捕获。关键边界仍是：

```text
输入  [B, S]
输出  ContextKV + [B, V] last-token logits
```

---

## 6. Initial TopK：Prefill 直接产生第一枚 SID Token

Prefill 返回：

```python
PrefillResult(
    logits=logits,
    context_kv=context_kv,
    hidden_states=None,
)
```

随后创建 `GRGenerationState`：

```text
GRGenerationState
├── PrefillResult
├── BeamKV
├── BeamPath
├── fixed_beam_width
└── beam_score_mode
```

Initial TopK 直接在 Prefill Logits 上选择第一层 Beam：

```python
selection = select_initial_topk_batched(
    logits,
    beam_width=W,
)
```

得到：

```text
token_ids:    [B, W]
scores:       [B, W]
parent_beams: [B, W]
```

这一结果成为 `BeamPath` 的第 0 个 Entry。

初始阶段所有 Beam 都来自同一个 Prompt 根节点；从下一步开始，Beam 才会产生真正的父子重排关系。

Initial TopK 有两条路径：

```text
纯 Batch 快路径：
  所有请求 beam_width 相同
  + 无 item mask
  + 无 dynamic beam policy
  + 无 logits processor
  → 一次 select_initial_topk_batched([B,V])

通用路径：
  对每个请求应用 processor / item mask / beam policy
  → request-local Initial TopK
```

在 `beam_score_mode="logprob"` 下，先对词表维执行 `log_softmax`；在 `"raw_logits"` 下直接使用 Logits。Initial 阶段的 `parent_beams` 全为 0，因为所有候选都来自唯一的 Prompt 根。

---

## 7. BeamKV：只保存很短的 Beam Decode 历史

[`BeamKV`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/src/gr_inference/gr_kv/beam_kv.py) 的布局为：

```text
[layers, batch, max_decode_steps, max_beam_width, kv_heads, head_dim]
```

对于 Qwen3-1.7B：

```text
[28, B, T, W, 8, 128]
```

三类状态的职责可以概括为：

| 数据结构 | 保存内容 | 是否包含长上下文 | 是否按 Beam 展开 |
| --- | --- | ---: | ---: |
| ContextKV | Prompt / 用户历史 K/V | 是 | 否 |
| BeamKV | SID Decode 的短 K/V 历史 | 否 | 是 |
| BeamPath | Token、Score、Parent Beam | 否 | 逻辑关系 |

因为 SID 通常只生成 3～5 层，即使 `beam_width=256`，BeamKV 的 Step 维度也非常短。

---

## 8. Decode Microbatch 是怎样构造的

假设当前：

```text
Batch Size = B
Active Beam Width = W
Decode Step = t
```

Executor 会构造：

```text
beam_token_ids: [B, W]
batched_generation
topk_indices
```

然后调用：

```python
model.forward_decode_step(
    beam_token_ids,
    batched_generation,
    decode_engine,
    step=t,
    active_beam_width=W,
    topk_indices=topk_indices,
    decode_nums=t + 1,
)
```

在 `_run_decode_batch` 中，主要工作依次是：

```text
1. 读取当前 Beam Selection
2. 合并多个请求的 BeamPath
3. 组合多个请求的 ContextKV
4. 组合多个请求的 BeamKV
5. 构造本轮 beam_token_ids
6. 构造 Beam 历史 topk_indices
7. 执行 Qwen3 Forward Decode Step
8. 应用 Logits Processor / Item Mask
9. 执行 Next TopK
10. 把新增 KV Scatter 回各请求 BeamKV
11. 更新请求状态或完成请求
```

### 8.1 request-local 状态怎样重新组合成 Tensor Batch

每个请求持有的是自己的：

```text
ContextKV: [L, 1, S, Hkv, D]
BeamKV:    [L, 1, T, Wmax, Hkv, D]
BeamPath:  Python request-local path
```

执行一个 Decode Microbatch 时，Executor 会构造临时 `batched_generation`：

```text
B 个 ContextKV → [L, B, S, Hkv, D]
B 个 BeamKV    → [L, B, T, Wmax, Hkv, D]
B 个 BeamPath  → BatchedBeamPath(paths)
```

组合 KV 有两个分支：

1. 请求占用同一个 Dense Pool 中按顺序连续的 Slot：用 `as_strided` 恢复 Batch View，零拷贝；
2. Slot 不连续或不是同一 Storage：使用 `torch.cat(dim=1)` 构造临时 Batch Tensor。

模型执行后，`_scatter_batched_beam_kv()` 只把当前 `step`、当前活动 Beam 范围写回 request-local BeamKV。如果临时 Batch 本来就是 Pool 的同一 View，则通过 Tensor View 签名发现源和目标相同，跳过复制。

### 8.2 为什么 Dynamic Beam Width 会触发 History Compaction

外部算子只接收切到当前宽度的：

```text
BeamKV[:, :, :decode_nums, :active_width]
```

若 Beam 从 256 缩到 64，某个当前 Beam 的祖先可能仍位于旧 Step 的 Slot 173。此时直接切 `:64` 会丢掉合法祖先。框架会检测：

```text
任一历史 ancestor_beam >= active_width
```

若成立，就把每个当前 Beam 的祖先 K/V Gather 到新的紧凑布局：

```text
历史 step s 的 ancestor KV → compact BeamKV[s, current_query_beam]
```

紧凑后，`topk_indices` 可以退化为恒等模式：

```text
step * active_width + current_query_beam
```

固定 Beam Width 不需要这次 compaction，这也是固定大 Beam Fast Path 更简单、更适合 CUDA Graph 的原因。

---

## 9. Qwen3 单步 Decode 的逐层计算

输入 token 先经过 Embedding：

```text
beam_token_ids: [B, W]
      ↓
hidden_states: [B, W, 2048]
```

然后进入 28 层 Qwen3 Decoder。每一层执行：

```text
1. Input RMSNorm
2. QKV Projection
3. Q/K Norm
4. RoPE(position = context_len + step)
5. 当前 K/V 写入 BeamKV
6. GR Decode Attention
7. O Projection
8. Residual + RMSNorm
9. MLP
10. Residual
```

最后经过 Final RMSNorm 和 LM Head：

```text
logits: [B, W, 151936]
```

即每个请求、每个当前 Beam 都有一份下一 token 分布。

单层 QKV 的 Shape 是：

```text
hidden: [B, W, 2048]
Q:      [B, W, 16, 128]
K/V:    [B, W,  8, 128]
```

位置编码使用：

```text
position = context_len + step
```

注意执行顺序是“先写当前 Step 的 K/V，再调用 Attention”。因此传给算子的：

```text
decode_nums = step + 1
```

表示 Beam 段中已经有多少个有效 token。当前 Query 会同时看到共享 Context 和包含当前输入 token 在内的 Beam 历史；随后 LM Head 预测下一枚 SID token。

### 9.1 当前 K/V 写入 BeamKV

每层通过：

```python
BeamKVWriter(generation.beam_kv).write_layer_step(
    layer_idx=layer_idx,
    step=step,
    active_beam_width=active_beam_width,
    k=k,
    v=v,
)
```

写入位置可以理解为：

```text
BeamKV[layer, batch, step, beam]
```

注意：当前 Beam 在历史各 Step 上的祖先，未必一直位于同一个 Beam Slot。这就是为什么仅有 BeamKV 还不够，还需要 BeamPath 和 `topk_indices`。

---

## 10. 最关键的部分：GR Decode Attention

普通自回归 Attention 通常认为某条 Sequence 的 KV 历史是连续的：

```text
Prompt KV + Decode KV
```

但 Beam Search 每轮都会重排：

```text
Step 0 的 Beam 0
    ↓
Step 1 的新 Beam 0 可能来自旧 Beam 7
    ↓
Step 2 的新 Beam 0 又可能来自上一轮 Beam 3
```

因此，当前 Query Beam 的正确历史并不等于：

```text
BeamKV[:, :, 0:t, current_beam]
```

必须沿着 `parent_beams` 追踪每一步的祖先。

### 10.1 GR Decode Attention 的输入契约

[`GRDecodeAttentionInputs`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/src/gr_inference/gr_kernels/attention/gr_decode_attention.py) 直接接收：

```text
Q
ContextKV
BeamKV
BeamPath
layer_idx
step
active_beam_width
topk_indices
decode_nums
```

Q 的形状是：

```text
[B, W, Hq, D]
```

而不是先展平为：

```text
[B × W, 1, Hq, D]
```

Kernel 从接口层面就知道 Batch 和 Beam 是两个不同维度。

---

## 11. `topk_indices` 如何恢复正确的 Beam 历史

[`make_batched_topk_indices`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/src/gr_inference/gr_runtime/batched_topk_indices.py) 构造：

```text
topk_indices: [B, 1, Hq, decode_nums, W]
```

对于每个当前 Query Beam，它记录：

```text
历史 Step 0 应读取哪个 Beam Slot
历史 Step 1 应读取哪个 Beam Slot
...
历史 Step t 应读取哪个 Beam Slot
```

核心回溯逻辑可以简化为：

```python
current = query_beam

for step in reversed(history):
    ancestry[step] = current
    current = entry.parent_beams[current]
```

BeamKV 的扁平索引为：

```text
decode_step * beam_width + ancestor_beam
```

### 11.1 一个宽度为 3 的具体例子

假设 Initial TopK 产生三个 Beam；Decode Step 0 选出的三个新 Beam，其父 Beam 是：

```text
BeamPath Entry 0 parents = [0, 0, 0]   # Prompt 根
BeamPath Entry 1 parents = [2, 0, 2]   # Decode Step 0 重排
```

下一轮计算 `decode_nums=2`。三个当前 Query Beam 的祖先链分别是：

| 当前 Query Beam | 历史 Step 0 Slot | 历史 Step 1 Slot | BeamKV 扁平索引 |
| ---: | ---: | ---: | --- |
| 0 | 2 | 0 | `[2, 3]` |
| 1 | 0 | 1 | `[0, 4]` |
| 2 | 2 | 2 | `[2, 5]` |

因为 `W=3`：

```text
Step 0 区间 = [0, 1, 2]
Step 1 区间 = [3, 4, 5]
```

所以 Query Beam 0 不能读取 `[0,3]`，而必须读取 `[2,3]`。`topk_indices[..., history_step, query_beam]` 正是在表达这张表。它会沿 Query Head 维扩展；同一个 GQA Group 的 Query Heads 实际共享对应 KV Head 的祖先索引。

所以该方案不需要在每轮 Beam 重排后物理搬运整段历史 KV，而是：

```text
BeamPath
    ↓
生成 topk_indices
    ↓
Attention Kernel 按索引读取正确祖先 KV
```

这是整个 SID-GR Decode 设计中最关键的一步。

---

## 12. 外部 `gr-decode_atten` Kernel 实际收到什么

框架内部张量：

```text
Q:         [B, W, Hq, D]
ContextKV: [L, B, Sctx, Hkv, D]
BeamKV:    [L, B, T, W, Hkv, D]
```

[`ExistingGRDecodeAttentionBackend`](https://github.com/NVIDIA/recsys-examples/blob/bdf16e32694127f8d196cfafc87f90765ab4b8b1/examples/sid-gr-inference/src/gr_inference/gr_kernels/attention/existing_kernel_backend.py) 会选择当前层，并转换为外部 Kernel 所需格式：

```text
Q:          [B, 1, W, Hq, D]
K Context:  [B, Sctx, Hkv, D]
V Context:  [B, Sctx, Hkv, D]
K Beam:     [B, decode_nums × W, Hkv, D]
V Beam:     [B, decode_nums × W, Hkv, D]
TopK Index: [B, 1, Hq, decode_nums, W]
```

最终调用：

```python
beam_decode_attn(
    q,
    k_context,
    v_context,
    k_beam,
    v_beam,
    topk_indices,
    decode_nums,
)
```

一次 Kernel 调用同时处理：

```text
共享长 Context Attention
+
按 BeamPath Gather 的短 Beam History Attention
```

这比把每个 Beam 伪装成独立 Sequence 更符合真实数据复用关系。

### 12.1 算子内部不是把两段 KV 简单拼接

从数学上，对某个 Query Beam，Attention 的合法 Key 集合是：

```text
所有 Context token
+
该 Beam 父链上的 decode token
```

实现上不需要真的构造：

```text
cat(ContextKV, gathered BeamKV)
```

`corelib/gr_decode_atten/interface.py` 把它分成三个逻辑阶段：

| 阶段 | 计算 | 访问特征 | 主要实现 |
| --- | --- | --- | --- |
| K1 Context Attention | `Q × ContextKV` | 长、连续、所有 Beam 共享 | FlashAttention 风格，Tensor Core |
| K2 Sparse Beam Attention | `Q × BeamKV[topk_indices]` | 极短、按祖先索引 Gather | Decode Kernel，CUDA Core scalar FMA |
| K3 Combine | 合并两段归一化结果 | 小规模 FP32 | Log-Sum-Exp Merge |

设两段各自的归一化输出和 LSE 为：

```text
(Octx, LSEctx)
(Obeam, LSEbeam)
```

全局 Softmax 的合并为：

```text
LSE = log(exp(LSEctx) + exp(LSEbeam))

O = exp(LSEctx  - LSE) * Octx
  + exp(LSEbeam - LSE) * Obeam
```

因此 K1、K2 可以使用最适合各自数据形态的 Kernel，K3 仍能得到与对完整 Key 集合做一次 Softmax 等价的结果。若长 Context 为提高 SM Occupancy 被切成多个 Split，K3 会把多个 Context Partial 和一个 Beam Partial 一起合并。

### 12.2 Sparse Beam Kernel 怎样使用 `topk_indices`

K2 首先把：

```text
Q [B,1,W,Hq,D] → [B×W,1,Hq,D]
```

Grid 的第一维因此是 `B×W`。每个 Block 可以恢复：

```text
batch_idx = flat_batch_idx // W
beam_idx  = flat_batch_idx % W
```

随后从：

```text
topk_indices[batch, kv_head, :decode_nums, beam]
```

取出绝对 BeamKV 行号，用 128-bit `cp.async` 把离散 K/V 行 Gather 到 Shared Memory，再对仅有 `decode_nums` 个历史位置做在线 Softmax。GQA 下，同组 Query Heads 共用一个 KV Head 和同一组祖先索引，K/V 不需要为每个 Query Head 重复加载。

### 12.3 Fused 路径与 3-Kernel 路径

接口支持：

```text
backend="dsl"      → 默认 Fused Context + Beam
backend="3kernel"  → 显式 K1 + K2 + K3
```

当前实现按架构选择：

```text
SM80 / SM90 / SM120：默认使用 Fused 路径
SM100 / SM110：使用 3-Kernel 路径
```

在 H100（SM90）常用生产路径上，Context 和 Sparse Beam 工作被放进同一个 Fused Kernel；当 Context 需要 Split-KV 时，再调用 Combine Kernel。`num_splits_heuristic` 会根据：

```text
B × Hq × ceil(W / tile_m)
Context KV block 数
GPU SM 数量
单个 KV Head 占用字节
```

选择 Split 数，目标是在不过度增加 Partial/Combine 成本的前提下提高 SM Occupancy。

### 12.4 Framework Wrapper 的责任边界

`GRDecodeAttention` 负责验证：

```text
Q、ContextKV、BeamKV 的 B/H/D 是否一致
step 与 active_beam_width 是否越界
topk_indices 的 B/Hq/decode_nums/W 是否足够
```

`ExistingGRDecodeAttentionBackend` 负责：

```text
选择当前 layer
给 Q 增加 Sq=1 维
把 [T,W] BeamKV Slice reshape 为 [T×W]
调用 beam_decode_attn
去掉不需要的 LSE 返回
```

它不负责 Beam Search，也不负责生成祖先索引。Beam 父链必须在进入算子前由 Runtime 转为 `topk_indices`；算子只按照 ABI 做高效 Gather 和 Attention。

---

## 13. Next TopK 与 BeamPath 更新

模型输出：

```text
logits: [B, W_old, V]
```

候选空间为：

```text
W_old × V
```

每个候选的累计得分为：

```text
previous_beam_score + current_token_score
```

代码没有直接对完整 `[B, W_old × V]` 先物化所有累计候选再做一次 TopK，而是做两级筛选：

```text
local_k = min(W_new, V)

第一层：每个旧 Beam 在 Vocab 内取 local_k
        [B, W_old, V] → [B, W_old, local_k]

第二层：加上 previous beam score 后展平
        [B, W_old × local_k] → 每个请求取 W_new
```

这在数学上是安全的：某个旧 Beam 若连自己的前 `W_new` 个 token 都进不了，来自该 Beam 更靠后的 token 不可能进入全局前 `W_new`。

在 `logprob` 模式下，局部候选分数为：

```text
top_logits - logsumexp(all_vocab_logits)
```

再加累计 `previous_scores`；在 `raw_logits` 模式下则直接加 Logit。最终用：

```text
parent_beam = flat_index // local_k
token_id    = gathered local_token_ids[flat_index]
```

得到新的 `W_new` 个 Beam：

```text
next_token_ids
next_scores
next_parent_beams
```

结果追加到 BeamPath：

```text
Entry 0：Prefill Initial TopK
Entry 1：Decode Step 0 的选择
Entry 2：Decode Step 1 的选择
...
```

下一轮 `topk_indices` 会根据新的 `parent_beams` 重新追踪祖先。

在 TopK 之前，框架还可以应用：

- Item Trie Mask；
- Token Suppression；
- Stop Token；
- Scheduled Beam Width；
- Score-margin Beam Width；
- 自定义 Logits Processor。

没有这些动态约束时，Continuous 路径可以使用 Tensor Selection Fast Path，把 Token、Score 和 Parent 信息尽可能保留在 GPU Tensor 中，减少 Python Materialization 和同步开销。

### 13.1 普通路径与 Tensor Selection Fast Path

普通路径每步会把 `token_ids/scores/parent_beams` materialize 成 CPU Tuple，立即追加到每个请求的 `BeamPath`。它支持完整功能，但每步存在 GPU→CPU 同步。

Tensor Selection Fast Path 只有在以下条件全部满足时启用：

```text
不返回 beam_details
当前 Beam Width == 下一步 Beam Width
无 Item Constraint
无 Dynamic Beam Policy
无 Logits Processor
无 Stop Token
Token / Score / Parent History 都在 CUDA Tensor
```

此时 `select_next_topk_batched(..., materialize=False)` 返回 GPU Tensor-backed Selection；父链暂存在 Tensor History，直到请求结束才批量回填 `BeamPath` 和构造最终输出。它优化的是 Host 同步和 Python 对象构造，不改变 TopK 公式。

### 13.2 Item Constraint 会怎样影响整个 Batch

约束 Mask 可以是：

```text
[B, V]       # 每请求统一
[B, W, V]    # 每个 Beam 不同
```

非法候选被填成 `-inf`。当前 Batched 实现要求同一个 Decode Microbatch 使用公共 `next_beam_width`，因此会取所有 Batch Row 中合法候选数量的最小值：

```text
W_effective = min(requested_width, min(valid_candidates_per_row))
```

这保证 Batch 内每行都能产生同样宽度的 Tensor，但也意味着一个约束更强的请求可能收缩整组请求的有效宽度。Scheduler 已把预期 `next_beam_width` 放入分组键，不过运行时 Item Mask 仍可能进一步缩小它。

---

## 14. 请求完成后如何恢复完整 SID

请求达到任一条件后结束：

```text
达到 max_decode_steps
所有 Beam 命中 Stop Token
所有 Beam 对应 Item 已完成
请求被取消、失败或超时
```

完整 SID 并不是每一步都复制成 `[W, T]` 保存，而是在结束时从最终 Beam 反向回溯：

```python
current_beam = final_beam

for step in reversed(beam_path):
    token = entry.token_ids[current_beam]
    current_beam = entry.parent_beams[current_beam]
```

反转后得到：

```text
[sid_level_0, sid_level_1, sid_level_2, ...]
```

默认 HTTP 输出中的 `beam_results` 类似：

```json
[
  {
    "output_ids": [101, 33, 7],
    "text": "",
    "meta_info": {
      "finish_reason": "max_decode_steps",
      "sequence_score": 12.34
    }
  }
]
```

如果配置了 Item Catalog，框架还会把 SID Token Path 解析成业务 Item，并附加 `item_results`。

---

## 15. KV Pool 和 CUDA Graph 为什么必须一起看

Continuous Serving 使用：

```text
GRDenseContextKVPool
GRDenseBeamKVPool
```

请求进入运行态时获得固定 Pool Slot；完成后释放 Lease。

固定、连续的 Pool Slice 有两个作用：

1. 避免请求期间频繁分配 KV Tensor；
2. 为 CUDA Graph 提供稳定的内存地址。

Decode CUDA Graph 默认使用 Batch Bucket：

```text
1, 2, 4, 8
```

实际 Batch 不足 Bucket 时可以 Padding；但只有当 ContextKV 和 BeamKV 对应稳定的连续 Pool Window 时，才允许复用 Graph。若 Pool Slice 非连续或指针不匹配，则回退到 Eager Decode。

Graph Replay 时主要更新小输入：

```text
beam_token_ids
topk_indices
少量动态状态
```

而大块 ContextKV / BeamKV 仍绑定到稳定 Pool View。

因此这里的 CUDA Graph 不是独立优化点，而是建立在以下前提上：

```text
SID-GR 固定短 Decode Shape
+ KV Pool 固定地址
+ Request × Beam 的稳定批处理结构
```

### 15.1 Decode CUDA Graph 的边界

当前 Graph 捕获的是模型 Decode Forward：

```text
beam_token_ids
→ Embedding
→ 28 层 Qwen3 Decode
→ LM Head
→ logits
```

Beam Selection 仍在 Graph 外执行。也就是说：

```text
CUDA Graph Replay
→ logits processor / item mask
→ logprob + TopK
→ 更新 Beam Parent State
```

当实际 Batch 小于 Graph Bucket 时，Executor 会尝试 Padding 到 `1/2/4/8` 等 Bucket。Padding 只有在对应的 ContextKV/BeamKV Pool Window 连续、空闲 Slot 足够且指针签名一致时才安全；否则记录 skip reason 并回退 Eager。Graph Cache 还带 LRU、最大 Entry 数和 Pointer Guard，避免错误复用旧 Pool 地址。

---

## 16. 关键 Tensor Shape 总结

设：

```text
B   = Request Batch Size
S   = Context Length
W   = Active Beam Width
T   = Decode Steps
H   = Hidden Size = 2048
Hq  = Query Heads = 16
Hkv = KV Heads = 8
D   = Head Dim = 128
V   = Vocabulary Size = 151936
L   = Layers = 28
```

| 阶段 | Tensor | Shape |
| --- | --- | --- |
| 输入 | `input_ids` | `[B, S]` |
| Prefill Embedding | `hidden_states` | `[B, S, 2048]` |
| Prefill Q | `q` | `[B, S, 16, 128]` |
| Prefill K/V | `k/v` | `[B, S, 8, 128]` |
| 长上下文缓存 | `ContextKV` | `[28, B, S, 8, 128]` |
| Prefill 输出 | Last-token Logits | `[B, 151936]` |
| Initial Beam | Token IDs | `[B, W]` |
| Decode Embedding | `hidden_states` | `[B, W, 2048]` |
| Decode Q | `q` | `[B, W, 16, 128]` |
| 短 Beam 缓存 | `BeamKV` | `[28, B, T, W, 8, 128]` |
| Beam 历史索引 | `topk_indices` | `[B, 1, 16, t+1, W]` |
| 算子 Context K/V | layer-local context | `[B, S, 8, 128]` |
| 算子 Beam K/V | flattened beam history | `[B, (t+1)×W, 8, 128]` |
| 算子输出 | Decode Attention output | `[B, 1, W, 16, 128]` |
| 算子 LSE | Combined log-sum-exp | `[B, 1, W, 16]` |
| Decode 输出 | `logits` | `[B, W, 151936]` |

---

## 17. 这套实现真正优化了什么

可以把整个系统浓缩成：

```text
Qwen3 Model
    负责标准 Transformer 计算

ContextKV
    保存长上下文，每请求一份

BeamKV
    保存短 Decode 历史

BeamPath
    保存 Beam 逻辑父子关系

topk_indices
    把 BeamPath 转换为 Kernel 可消费的 KV 索引

gr-decode_atten
    直接消费 ContextKV + BeamKV + topk_indices

Continuous Scheduler
    按 Request × Active Beam 组织执行
```

因此，它的性能收益并不只是“替换了一个 Attention 算子”，而是来自整条链路的协同设计：

1. KV 数据结构与 SID-GR Workload 对齐；
2. 长 ContextKV 不按 Beam 复制；
3. Beam 历史通过索引追踪，而不是反复物理重排；
4. Decode Batch 保留 Request 和 Beam 两个维度；
5. Prefill 只计算 Last-token Logits；
6. KV Pool 提供稳定地址；
7. Decode 使用固定 Shape CUDA Graph；
8. 简单场景下 TopK Selection 尽量留在 GPU Tensor 路径。

---

## 18. 推荐的源码阅读顺序

建议不要从 Kernel 直接开始，而是沿请求生命周期阅读：

```text
1. tools/serve_qwen3_gr_http.py
2. gr_serving/http.py
3. gr_serving/request.py
4. tools/run_qwen3_real_weight_serving.py
5. gr_serving/continuous.py
6. gr_models/qwen3/model.py
7. gr_models/qwen3/layers.py
8. gr_runtime/generation.py
9. gr_kv/context_kv.py
10. gr_kv/beam_kv.py
11. gr_runtime/batched_beam_search.py
12. gr_runtime/batched_topk_indices.py
13. gr_runtime/beam_kv_compaction.py
14. gr_runtime/engine.py
15. gr_kernels/attention/gr_decode_attention.py
16. gr_kernels/attention/existing_kernel_backend.py
17. corelib/gr_decode_atten/interface.py
18. corelib/gr_decode_atten/src/decode/flash_fwd.py
19. gr_serving/beam_metadata.py
```

其中最值得重点精读的函数是：

```text
GRContinuousServingExecutor._run_prefill
GRContinuousServingExecutor._store_single_prefill_result
GRContinuousServingExecutor._run_decode_batch
GRContinuousServingExecutor._forward_decode_step
Qwen3GRModel.forward_prefill
Qwen3GRModel.forward_decode_step
Qwen3SingleLayerPrefill.forward_decode
make_batched_topk_indices
select_next_topk_batched
ExistingGRDecodeAttentionBackend.__call__
BeamDecodeAttn.forward
FlashAttentionForwardDecode._gather_load_tile
```

---

## 19. 最终理解

SID-GR Inference 的核心不是把通用 LLM Serving 缩小，而是重新定义适合生成式推荐的运行时对象：

```text
通用 LLM Serving：
Sequence / Token Block / Paged KV

SID-GR Inference：
Request-level ContextKV
+ Step-major BeamKV
+ BeamPath
+ Request × Active-beam Batch
```

对于“长上下文、短生成、大 Beam”的推荐检索场景，这种抽象比把每个 Beam 当作独立 Sequence 更自然：

- 上下文共享关系在数据结构中被显式表达；
- Beam 重排关系通过索引表达；
- Attention Kernel 直接理解两类 KV；
- Scheduler 也按真实执行维度组织 Batch。

理解了 `ContextKV + BeamKV + BeamPath + topk_indices` 这条主线，Qwen3-1.7B 在 SID-GR Inference 中的完整推理流程就基本串起来了。
