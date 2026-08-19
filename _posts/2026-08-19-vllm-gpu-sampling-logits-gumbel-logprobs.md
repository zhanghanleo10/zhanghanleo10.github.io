---
layout: post
title: "vLLM 源码课程 10：一次 GPU Sampling 事务——从 Logits 处理到 Gumbel-max 与 Logprobs"
date: 2026-08-19 09:00:00 +0800
categories: [AI Infra, vLLM]
tags: [vllm, sampling, logits, gumbel-max, rng, logprobs]
mermaid: true
---

> 本文基于 vLLM `main` 的 [`f1178f3a`](https://github.com/vllm-project/vllm/commit/f1178f3a06fa30a0cc282376924210cedad08c44)。源码事实固定到该 commit；本文没有运行 GPU 实验。测试事实、未合入 PR 与基于实现的推断会分别标注。

## 本篇在课程路线中的位置

前九篇已经追完 `SchedulerOutput → ModelRunner → Attention/KV cache → ModelRunnerOutput → Scheduler commit`。本篇进入 Sampling 阶段，只研究普通自回归路径上的一笔事务：

**最后一层 hidden states 如何变成经过约束的 logits，per-request 参数与 RNG 状态怎样跨 step 保持，Gumbel-max 如何产生 token，raw/processed logprobs 又如何与这次选择对齐。**

speculative decoding 的 `RejectionSampler`、多 rank 聚合和 CUDA Graph capture 不在本篇展开；否则会掩盖 Sampler 自身最重要的所有权与确定性边界。

## 前置知识回顾

第 07 章确认，ModelRunner v2 用稳定的 request slot 保存跨 step 状态，再以 `idx_mapping` 把稀疏 slot gather 成本轮紧凑 batch。第 08 章确认，GPU 输出只有经过异步 D2H 和 Scheduler commit 才成为请求历史。Sampling 延续同一套事务模型：参数属于 request slot，processed logits 属于当前 step，sampled token 在 Scheduler 接纳前只是候选结果。

## 本篇要回答的核心问题

从第一性原理看，一个可维护的 batched sampler 必须同时满足五条约束：

1. 同一 batch 可混合 greedy、temperature、top-k/top-p、penalty 和 bad-word 请求；
2. batch 压缩、重排和 padding 不应改变显式 seed 请求的随机流；
3. 被 mask 为 `-inf` 的 token 永远不可被采样；
4. 返回的 logprobs 必须明确是处理前还是处理后的分布；
5. 持久参数地址可复用，但 request slot 复用时不得泄漏上一请求的状态。

这些约束决定了当前实现为何不只是 `softmax + torch.multinomial`。

## 组件在全局架构中的位置

~~~mermaid
flowchart LR
    H["hidden_states [T,H]"] --> I["logits_indices<br/>选择需要采样的 rows"]
    I --> L["model.compute_logits<br/>logits [R,V]"]
    L --> G["grammar bitmask<br/>masked token = -inf"]
    G --> P["Sampler.apply_sampling_params<br/>临时 FP32 processed_logits"]
    S["SamplingStates<br/>request slot → temp/top-k/top-p/seed"] --> P
    P --> K["top-k/top-p masking"]
    K --> M["gumbel_sample<br/>seed + logical position + vocab lane"]
    M --> O["SamplerOutput<br/>sampled_token_ids [R,1] + logprobs"]
    O --> A["AsyncOutput D2H"]
    A --> C["Scheduler commit / rollback"]
~~~

owner map 很关键：`GPUModelRunner` 拥有 `Sampler`；`Sampler` 拥有按 request slot 索引的持久状态；当前 step 的 logits/候选 token 由执行流暂时拥有；Scheduler 才拥有“这个 token 已进入请求历史”的语义所有权。

## 完整调用链

[`GPUModelRunner.sample()`](https://github.com/vllm-project/vllm/blob/f1178f3a06fa30a0cc282376924210cedad08c44/vllm/v1/worker/gpu/model_runner.py#L1335-L1372) 先执行：

~~~text
sample_hidden_states = hidden_states[input_batch.logits_indices]
logits = model.compute_logits(sample_hidden_states)
apply_grammar_bitmask(logits)     # 可选，原地把非法 token 置为 -inf
sampler_output = sampler(logits, input_batch)
~~~

因此 `T` 是本轮执行的 packed token 数，而 `R` 是真正需要产出 token 的 row 数；chunked prefill row 可以执行模型但暂不采样。正常路径进入 [GPU v2 `Sampler.__call__`](https://github.com/vllm-project/vllm/blob/f1178f3a06fa30a0cc282376924210cedad08c44/vllm/v1/worker/gpu/sample/sampler.py#L97-L179)，而 speculative path 会转入 `RejectionSampler`，两者不能混为一套 RNG 契约。

Sampler 先用 `idx_mapping/expanded_idx_mapping` 把当前 rows 映射回持久 request slots，从 `positions[logits_indices]` 得到逻辑 token position；随后依次处理 logits、采样、计算可选 logprobs，并通过 `get_num_sampled_and_rejected` 得出每请求真实输出长度。最终 `sampled_token_ids` reshape 为 `[R,1]`。

[`GPUModelRunner.sample_tokens()`](https://github.com/vllm-project/vllm/blob/f1178f3a06fa30a0cc282376924210cedad08c44/vllm/v1/worker/gpu/model_runner.py#L1707-L1775) 将结果封装为 `AsyncOutput`：copy stream 先把 GPU 结果搬到 Host，随后 `postprocess_sampled` 更新下一 step 使用的 token/penalty 状态。这个顺序仍不等于最终提交；Scheduler 收到并校验 stale/spec/stop 状态后才更新请求。

## 关键类型、字段和状态生命周期

[`SamplingStates`](https://github.com/vllm-project/vllm/blob/f1178f3a06fa30a0cc282376924210cedad08c44/vllm/v1/worker/gpu/sample/states.py#L16-L119) 为 `max_num_reqs` 个稳定槽持有五个 `UvaBackedTensor`：

| 状态 | shape / dtype | 语义 |
| --- | --- | --- |
| `temperature` | `[max_num_reqs] float32` | 0 表示 greedy，1 表示不缩放 |
| `top_k` | `[max_num_reqs] int32` | 负值在 admission 时规范化为 vocab size |
| `top_p` / `min_p` | `[max_num_reqs] float32` | 概率截断参数 |
| `seeds` | `[max_num_reqs] int64` | 请求生命周期内稳定的 RNG 根 seed |
| `seeds_set` | Host `bool[]` | 区分用户显式 seed 与内部随机 seed |

请求加入时，`Sampler.add_request()` 同时写 penalties、logit bias、bad words、thinking budget 和上述 sampling states，并缓存 `needs_logits_processing[req_idx]`。参数先 staged 在 Host/UVA backing storage，再由 `apply_staged_writes()` 进入设备可见状态。无显式 seed 时只在 admission 生成一次随机 int64；不是每 step 重抽。

生命周期如下：

~~~mermaid
sequenceDiagram
    participant MR as GPUModelRunner
    participant SS as SamplingStates[request_slot]
    participant SP as Sampler
    participant GPU as Gumbel kernel
    participant SCH as Scheduler
    MR->>SS: add_request(slot, SamplingParams)
    MR->>SS: apply_staged_writes()
    MR->>SP: logits + idx_mapping + logical positions
    SP->>SS: gather temp/top-k/top-p/seed by slot
    SP->>GPU: processed logits + seed + pos
    GPU-->>SP: sampled_token_ids
    SP-->>MR: SamplerOutput (GPU)
    MR-->>SCH: AsyncOutput D2H
    SCH->>SCH: commit/rollback token
    Note over MR,SS: slot 释放后，下个请求必须完整覆盖旧状态
~~~

这里的 ABA 风险不是理论问题：紧凑 batch row 0 今天可能映射 slot 7，下一 step 可能映射 slot 2。kernel 必须经 `expanded_idx_mapping` 取状态，不能直接用 row index。

## 逐函数源码解读

### 1. `apply_sampling_params`：按语义顺序构造临时分布

[实现](https://github.com/vllm-project/vllm/blob/f1178f3a06fa30a0cc282376924210cedad08c44/vllm/v1/worker/gpu/sample/sampler.py#L181-L246) 先检查当前 active slots 的 `needs_logits_processing`。若全为 false，原 logits 可直接向下传递；只要一个请求需要处理，就复制出 FP32 tensor，然后严格按以下顺序原地修改：

1. logit bias、allowed tokens、`min_tokens`；
2. repetition/presence/frequency penalties；
3. bad words；
4. thinking budget 的强制结束；
5. temperature；
6. `min_p`；
7. top-k/top-p（可延后给 sampler backend）。

顺序是 contract，不是实现细节。例如 thinking budget 必须在 temperature 前把 EOS 变成必选候选；mask/penalty 后的 `-inf` 必须跨 temperature 和 Gumbel 仍保持不可能。

### 2. `sample`：加速路径必须服从请求语义

[`Sampler.sample()`](https://github.com/vllm-project/vllm/blob/f1178f3a06fa30a0cc282376924210cedad08c44/vllm/v1/worker/gpu/sample/sampler.py#L248-L299) 可调用 FlashInfer top-k/top-p sampling，但出现任一条件就回退 native Gumbel：batch 中含 greedy、用户设置 per-request seed、需要 processed logprobs，或 backend 不支持。这里优先保持 API 语义，而非强行统一到最快 kernel。

native 路径先 mask top-k/top-p，再调用 [`gumbel_sample()`](https://github.com/vllm-project/vllm/blob/f1178f3a06fa30a0cc282376924210cedad08c44/vllm/v1/worker/gpu/sample/gumbel.py#L215-L260)。每 1024 个 vocab lanes 做一次局部 argmax，再在 PyTorch 中对 block maxima 做第二级 argmax。

### 3. Gumbel-max：随机流绑定逻辑 token，而非 batch row

[`gumbel_block_argmax`](https://github.com/vllm-project/vllm/blob/f1178f3a06fa30a0cc282376924210cedad08c44/vllm/v1/worker/gpu/sample/gumbel.py#L85-L169) 的关键不是公式，而是随机 key：

~~~text
request_seed = seeds[expanded_idx_mapping[token_row]]
gumbel_seed  = randint(request_seed, logical_position)
noise[v]     = Gumbel(gumbel_seed, vocab_block_and_lane)
token        = argmax_v(logit[v] / temperature + noise[v])
~~~

这使显式 seed 的序列原则上不依赖紧凑 batch 中的 row 顺序。`temperature == 0` 时 kernel 跳过噪声并返回精确 argmax。FP32 路径采用翻转尾部的 `-log(-log1p(-u))`，利用 float32 在 0 附近更细的分辨率，避免极小概率 token 的 Gumbel 尾被粗化；这是已合入 [PR #45996 对应提交](https://github.com/vllm-project/vllm/commit/16908e132e10f75af93049e865130f8987573f5d) 的修正。

### 4. logprobs：必须声明观察哪个分布

`Sampler.__call__` 在采样后根据 `logprobs_mode` 选择 raw 或 processed logits，再调用 [`compute_topk_scores`](https://github.com/vllm-project/vllm/blob/f1178f3a06fa30a0cc282376924210cedad08c44/vllm/v1/worker/gpu/sample/logprob.py#L105-L181)。raw 模式回答“模型原始分布如何评价 token”；processed 模式回答“实际受约束采样分布如何评价 token”。两者在 penalty、grammar 或 top-k 后可能完全不同，调用方不能只看数值而忽略 mode。

## 具体示例与 shape/状态演算

设本轮 `T=19, H=4096, R=3, V=128000`，三个紧凑 rows 映射到持久 slots `[7,2,11]`：

| row → slot | 参数 | 处理与结果 |
| --- | --- | --- |
| 0 → 7 | `temperature=0` | 不加噪声，取有限 logits 的 argmax |
| 1 → 2 | `temperature=0.7, top_k=3, seed=42, pos=100` | 仅 top 3 保持有限；随机 key 由 `(42,100,vocab_id)` 决定 |
| 2 → 11 | `top_p=0.9`，有 penalty/bad words | 先 penalty/mask，再按累计概率保留候选，最后 Gumbel-max |

`logits_indices [3]` 从 `hidden_states [19,4096]` 选出 `[3,4096]`，LM head 得 `logits [3,128000]`。只要 row 1 或 row 2 需要处理，整批建立 FP32 processed logits，约 `3×128000×4 = 1.46 MiB`。Gumbel 两级归约有 125 个 vocab blocks；局部 `int64 argmax` 与 FP32 maxima 合计约 4.4 KiB。由此可见，主要临时容量在完整 processed logits，不在归约 scratch。

对 row 1，假设 top-k 后仅 token `{5,17,900}` 有限。kernel 不需要先物化 softmax：对三者分别计算 `logit/0.7 + Gumbel(42,100,v)`，其余 `-inf + noise` 仍是 `-inf`，最终 winner 可由同一 seed/position 重放。这里“winner 是谁”依赖 kernel RNG；本文不伪造具体随机数。

## 为什么这样设计及替代方案

直接使用 `softmax(logits/T) + torch.multinomial` 更容易写 reference，但会物化完整概率或归一化状态，也更难把 RNG 与 per-request logical position 绑定。Gumbel-max 在 logit space 完成同一 categorical sampling，mask 自然由 `-inf` 表达，并适合分块归约。

代价也真实存在：持久 request-slot buffers 改善地址稳定性和每步参数装配成本，却把 slot 清理、staged write 时序变成正确性边界；而 `needs_logits_processing` 是 batch 级 fast path，只要一个 active request 需要处理，整批都承担 FP32 copy。这会形成由 batch composition 触发的性能台阶。

FlashInfer 是另一种取舍：支持时可减少 native mask/sample 的开销，但当前代码在 greedy、显式 seed、processed logprobs 下主动回退。maintainer 不应删除这些 guard，除非替代 backend 已证明逐请求语义与确定性兼容。

## 性能、并发、正确性与边界条件

- **吞吐**：大词表下 FP32 logits copy 和 top-k/top-p 扫描是 O(RV)；一个复杂请求可拖慢混合 batch。应按 processor mix 做 benchmark，而不只测全 greedy/全 sampling。
- **并发**：Sampler 状态按 ModelRunner request slot 所有；Host 写与设备消费之间必须经过 `apply_staged_writes`。slot 复用时漏写任一子状态会产生跨请求污染。
- **确定性**：当前 key 设计隔离 batch row，但依赖 logical `pos` 正确提交。抢占恢复、stale output 或回滚若让 position 重复/跳跃，会重复或跳过随机流。
- **graphability（推断）**：固定容量 UVA/device buffers 有利于 graph replay；但本篇未验证 sampler 是否处于特定 full/piecewise graph capture 边界，不能把“地址稳定”直接写成“已可捕获”。
- **兼容性**：分布一致不等于 seeded sequence 一致。未合入的 draft [PR #51367](https://github.com/vllm-project/vllm/pull/51367) 计划更换 Gumbel RNG；即使统计分布不变，也可能改变用户观察到的 seeded token 序列。它是计划材料，不是当前代码事实。

## 测试证据与未覆盖风险

[`test_gpu_sampler_flags.py`](https://github.com/vllm-project/vllm/blob/f1178f3a06fa30a0cc282376924210cedad08c44/tests/v1/worker/test_gpu_sampler_flags.py#L44-L91) 验证 default/greedy 不触发处理，而 bias、penalty、bad words、temperature、min-p、top-k/top-p 会触发；它还验证 slot 重用会重置 flag，以及 batch 判定只看 active slots。

[`test_gpu_gumbel_sample.py`](https://github.com/vllm-project/vllm/blob/f1178f3a06fa30a0cc282376924210cedad08c44/tests/v1/worker/test_gpu_gumbel_sample.py#L113-L227) 是 CUDA/Triton 测试：

- 20 万词表、50 万次 heavy-tail sampling 检查尾部概率误差；
- 400 万次近均匀采样要求词表覆盖率超过 99%，并做 chi-square bound；
- `temperature=0` 必须精确等于 argmax，`-inf` token 永不出现；
- vocab size 为 1、999、1024、4097，保护非 block 对齐的 tail mask；
- cache 测试要求保存的是 temperature 前、bit-exact 的 logits。

这些是**测试事实**，证明 kernel 的统计与边界行为；不是端到端实验。最小新增 CI guard 应固定一个显式 seed 请求，分别经历 batch row 重排、request-slot 复用、preemption/resume 与 graph replay，断言 committed token 序列和 logprobs mode 一致。另应给 native/FlashInfer 建立同参数的分布与错误语义 parity，而不是要求随机 token 逐个相同。

## 与前后章节的连接

本篇把第 09 章 Attention 输出接到了第 08 章返回事务：Attention 产生 hidden states，Sampler 产生候选 token，AsyncOutput 搬运，Scheduler 最终提交。下一章进入分布式执行边界：为什么只有 last pipeline-parallel rank 拥有 LM head/Sampler，`SamplerOutput` 如何广播，以及各 rank 怎样保持请求状态一致。

## 本篇结论、知识债与理解检查

结论是：当前 GPU v2 Sampling 的核心抽象不是一个无状态函数，而是“**稳定 request-slot 状态 + 当前 step 的 FP32 分布变换 + 以 seed/逻辑位置/vocab lane 为 key 的随机选择 + 延迟提交**”。维护时最危险的不是 Gumbel 公式，而是 state mapping、processor 顺序和 commit position 三者漂移。

仍欠缺：speculative rejection 的 RNG/position contract；PP/DP 下 output 广播；explicit seed 的端到端兼容策略；不同 GPU backend 的 native/FlashInfer parity；processed logprobs 在 grammar/penalty/top-p 组合下的 golden fixtures。

理解检查：

1. 为什么显式 seed 请求不能直接以 compact batch row 作为 RNG offset？
2. 为什么一个需要 penalty 的请求可能让整个 mixed batch 多一次 FP32 logits copy？
3. “Gumbel 分布测试通过”为什么仍不足以承诺升级后的 seeded token 序列兼容？

下一章：last PP rank sampling、`SamplerOutput` 广播与跨 rank 状态收敛。

## 课程账本增量

- 新阶段：Sampling。
- 新增主链：`hidden_states → compute_logits → grammar/processors → Gumbel-max → logprobs → AsyncOutput → Scheduler commit`。
- 新增不变量：sampling state 以 request slot 为 owner；RNG 绑定 seed/逻辑 position/vocab lane；processor 顺序与 logprobs mode 属于外部语义；加速 backend 不满足语义时必须回退。
- 最高优先知识债：seeded sequence 在重排、抢占恢复、slot 复用、graph replay及 RNG 升级下的兼容测试。


