---
layout: post
title: "vLLM 源码课程 09：Attention 的三张地址表——KV 写入、Paged 读取与编译顺序"
date: 2026-08-18 09:00:00 +0800
categories: [AI Infra, vLLM]
tags: [vllm, attention, paged-attention, kv-cache, cuda-graph]
mermaid: true
---

> 本文基于 vLLM <code>main</code> 的 [<code>c296851a</code>](https://github.com/vllm-project/vllm/commit/c296851a7d173fa89d2eefbca0243be42ae9b5e0)。源码事实固定到该 commit；本文没有运行 GPU 实验。仓库单测能够证明的结论与基于实现的性能推断会明确区分。

## 本篇在课程路线中的位置

前八篇已经闭合一轮请求事务：Scheduler 分配物理 KV blocks，ModelRunner 把计划变成设备输入，设备输出再由 Scheduler commit/rollback。现在进入 Attention 阶段。本篇只回答一个维护问题：

**ModelRunner 生成的 <code>query_start_loc</code>、<code>slot_mapping</code>、<code>block_table + seq_lens</code> 分别控制什么，它们怎样协作完成“先写当前 K/V，再按逻辑序列读取 paged KV”？**

这不是字段导览。若把三者混为“KV 地址”，代码可能不报错，却会把某请求的 K/V 写进另一请求的 page，或让 Attention 读到 padding/stale block。

## 前置知识回顾

第 07 章确认：Scheduler 只传 Host 侧 block IDs 与 token 计划；<code>BlockTables</code> 在 Worker 内维护固定地址的 device buffers，并用

<code>slot = block_id × block_size + position % block_size</code>

生成当前 step 的物理写槽。第 08 章确认：一个 step 是 plan/result transaction。Attention 位于二者之间，它消费已经 materialize 的 device metadata，但不拥有 Scheduler 的 KV allocation。

## 本篇要回答的核心问题

从第一性原理看，paged causal attention 至少要解决三个不可合并的问题：

1. packed 的 query row 属于哪个请求？
2. 本轮新算出的每个 K/V row 写到哪一个物理 slot？
3. 每个请求的逻辑历史位置从哪些物理 pages 读取，读到哪里停止？

当前实现分别用 <code>query_start_loc</code>、<code>slot_mapping</code>、<code>block_table + seq_lens</code> 回答。它们必须描述同一批请求、同一次 KV allocation，但索引空间不同。

## 组件在全局架构中的位置

~~~mermaid
flowchart LR
    S["SchedulerOutput<br/>block IDs + scheduled tokens"] --> BT["BlockTables<br/>gather_block_tables<br/>compute_slot_mappings"]
    BT --> C["CommonAttentionMetadata"]
    C --> FB["FlashAttentionMetadataBuilder"]
    FB --> FC["ForwardContext<br/>per-layer metadata + slot mapping"]
    FC --> L["Attention.forward"]
    L --> W["unified_kv_cache_update<br/>scatter K/V by slot_mapping"]
    W --> R["FlashAttentionImpl.forward<br/>read by block_table + seq_lens"]
    R --> O["packed attention output"]
~~~

owner map：

- Scheduler/KVCacheManager 拥有 logical request 与 block ownership；
- <code>BlockTables</code> 拥有 Worker 侧 persistent block-table/slot buffers；
- ModelState 构造本 step 的 Attention metadata；
- 每个 <code>Attention</code> layer 拥有对应 <code>kv_cache</code> tensor；
- backend 负责解释 metadata、写 cache、执行 attention；它不能私自改变 block allocation。

## 完整调用链

在 [<code>GPUModelRunner</code>](https://github.com/vllm-project/vllm/blob/c296851a7d173fa89d2eefbca0243be42ae9b5e0/vllm/v1/worker/gpu/model_runner.py#L1305-L1324) 中，每个 KV cache group 先调用 <code>gather_block_tables()</code> 得到紧凑 batch rows，再调用 <code>compute_slot_mappings()</code> 得到 token rows 的写地址。padding token 被写成 <code>PAD_SLOT_ID=-1</code>。

[<code>DefaultModelState.prepare_attn()</code>](https://github.com/vllm-project/vllm/blob/c296851a7d173fa89d2eefbca0243be42ae9b5e0/vllm/v1/worker/gpu/model_states/default.py#L166-L228) 将 query 起点、sequence length、block table、slot mapping 传给 [<code>build_attn_metadata()</code>](https://github.com/vllm-project/vllm/blob/c296851a7d173fa89d2eefbca0243be42ae9b5e0/vllm/v1/worker/gpu/attn_utils.py#L591-L680)。后者先构造 backend-neutral 的 <code>CommonAttentionMetadata</code>，再让每个 backend builder 生成专用 metadata；同组 layer 共用 metadata，但各 layer 仍有独立 KV cache。

执行时 [<code>Attention.forward()</code>](https://github.com/vllm-project/vllm/blob/c296851a7d173fa89d2eefbca0243be42ae9b5e0/vllm/model_executor/layers/attention/attention.py#L482-L574) 把 Q/K/V reshape 为 token-major heads。FlashAttention 声明 <code>forward_includes_kv_cache_update=False</code>，因此公共 layer 先调用 <code>unified_kv_cache_update</code>，再调用 <code>unified_attention_with_output</code>。

写路径最终进入 [<code>FlashAttentionImpl.do_kv_cache_update()</code>](https://github.com/vllm-project/vllm/blob/c296851a7d173fa89d2eefbca0243be42ae9b5e0/vllm/v1/attention/backends/flash_attn.py#L1233-L1267) 和 <code>reshape_and_cache_flash</code>。Triton kernel 对每个 token 读取一个 slot：

~~~text
slot < 0       → padding，跳过
block_id       = slot // block_size
block_offset   = slot % block_size
cache[block_id, :, block_offset, :] = K/V[token]
~~~

读路径进入 [<code>FlashAttentionImpl.forward()</code>](https://github.com/vllm-project/vllm/blob/c296851a7d173fa89d2eefbca0243be42ae9b5e0/vllm/v1/attention/backends/flash_attn.py#L970-L1231)，把 packed cache 拆为 K/V views，并把 <code>query_start_loc</code>、<code>seq_lens</code>、<code>block_table</code> 交给 varlen FlashAttention。注意：**读取历史 KV 时不消费 <code>slot_mapping</code>。**

## 关键类型、字段和接口契约

[<code>CommonAttentionMetadata</code>](https://github.com/vllm-project/vllm/blob/c296851a7d173fa89d2eefbca0243be42ae9b5e0/vllm/v1/attention/backend.py#L470-L663) 是跨 backend 的契约：

| 字段 | shape / dtype / device | 语义与调用方责任 |
| --- | --- | --- |
| <code>query_start_loc</code> | <code>[B+1]</code>，整数，device | packed query 的前缀和；必须单调，末项等于实际 query token 数 |
| <code>seq_lens</code> | <code>[B]</code>，整数，device | 每请求本轮可见 KV 长度；是 read validity boundary |
| <code>block_table_tensor</code> | <code>[B,max_blocks]</code>，整数，device | logical block index 到 physical block ID 的映射 |
| <code>slot_mapping</code> | <code>[T_padded]</code>，<code>int64</code>，device | 当前 K/V row 到 physical slot；padding 必须为负数 |
| <code>num_actual_tokens</code> | Host scalar | 区分真实 token 与 graph padding |

FlashAttention 的 Q 为 <code>[T,num_q_heads,head_size]</code>，K/V 为 <code>[T,num_kv_heads,head_size]</code>。逻辑上的 packed KV cache 为 <code>[num_blocks,num_kv_heads,block_size,2×head_size]</code>；实际 stride 可选择 NHD/HND，但 backend 的 split/view 必须保持同一地址解释。

前置条件是：每个非 padding slot 指向当前 layer 已分配且仍存活的 block；<code>seq_lens</code> 不超过 block table 的有效容量；当前 token 所在 block 已包含在 block table 中。后置条件是：当前 K/V 恰好写入一次，attention output 只观察本请求的有效历史与当前 token。

失败模式不只包括 shape/assert 错误。若 slot 合法但属于另一 request，kernel 会静默污染其 KV；若 <code>seq_lens</code> 偏大，读取也可能越过逻辑有效区。Scheduler 的 CPU-side max seq len 在 async speculative 场景可能是上界，精确 decode kernel 不能把它当每请求真实长度。

## 数据与 Buffer 生命周期

~~~mermaid
sequenceDiagram
    participant Sch as Scheduler/KVCacheManager
    participant BT as BlockTables
    participant Meta as Attention metadata
    participant KV as Layer KV cache
    participant FA as FlashAttention
    Sch->>BT: block IDs + scheduled positions
    BT->>BT: gather rows; compute physical slots
    BT->>Meta: block_table / slot_mapping
    Meta->>KV: do_kv_cache_update(K,V,slot_mapping)
    Note over Meta,KV: dummy dependency preserves write-before-read
    Meta->>FA: Q, KV views, block_table, seq_lens
    FA->>KV: paged reads for valid logical positions
    FA-->>Meta: packed output rows
~~~

<code>input_block_tables</code> 与 <code>slot_mappings</code> 是 persistent device buffers：step 间改内容、不换地址，才能被 full CUDA Graph 重放。actual batch 之后的 row/token 必须主动清零或填 <code>PAD_SLOT_ID</code>，否则 graph padding 会继承上一 step 的有效地址。

KV cache 的 lifetime 更长：由 layer 初始化并跨 step 复用；block 的逻辑 ownership 由 Scheduler 分配/释放。metadata 只借用 buffer 与 cache views，在当前 forward 有效，不获得 block ownership。

## 具体演算：两个 decode token

设 <code>block_size=16</code>，两个请求各 decode 一个 token：

- R0：<code>seq_len=18</code>，block table 为 <code>[7,3,...]</code>。新 token 位置 17 位于 logical block 1、offset 1，因此写 slot <code>3×16+1=49</code>。
- R1：<code>seq_len=33</code>，block table 为 <code>[11,5,9,...]</code>。新 token 位置 32 位于 logical block 2、offset 0，因此写 slot <code>9×16=144</code>。

于是：

~~~text
query_start_loc = [0, 1, 2]
seq_lens        = [18, 33]
slot_mapping    = [49, 144]
~~~

若 TP-local <code>num_q_heads=8</code>、<code>num_kv_heads=2</code>、<code>head_size=128</code>，则 Q 为 <code>[2,8,128]</code>，K/V 各为 <code>[2,2,128]</code>。写 kernel 把两行 K/V scatter 到 slots 49 和 144；读 kernel 对 R0 依次读取 physical blocks 7、3 中前 18 个位置，对 R1 读取 11、5、9 中前 33 个位置。

因 causal self-attention 包含当前 token，本轮 K/V 必须先写入，读才能看到位置 17/32。这里 <code>slot_mapping</code> 完成“写哪里”，<code>block_table + seq_lens</code> 完成“历史从哪里读、读多少”，<code>query_start_loc</code> 完成“输出归属谁”。

## 为什么拆分写与读，以及替代方案

历史上的 [PR #25954](https://github.com/vllm-project/vllm/pull/25954) 将 FlashAttention 的 cache update 从 forward 中拆出；当前代码是最终事实。原因不是抽象洁癖：独立写 op 让公共 Attention layer 统一 cache 更新，也让 compiler 更容易把 attention read 作为纯计算区域处理。

代价是 cache update 是隐藏副作用。公共 op 在 [<code>get_attention_context / unified ops</code>](https://github.com/vllm-project/vllm/blob/c296851a7d173fa89d2eefbca0243be42ae9b5e0/vllm/model_executor/layers/attention/attention.py#L670-L801) 中返回一个 dummy tensor，并把它喂给后续 attention op，只为让 <code>torch.compile</code> 保留 write-before-read 顺序。删掉这个看似无用的 dependency，eager 可能仍正常，compiled/full graph 却可能重排或消去副作用边。

替代方案是 backend 内融合 cache update 与 attention。它少一次 launch，天然维持顺序，对 decode 小 batch 可能更快；但会把 cache layout、slot 解释和 attention kernel 强耦合，降低跨 backend 复用与 compiler partition 自由度。当前拆分是合理默认，但应以 profile 复核：若小 batch 的 update launch 成为关键路径，或硬件提供原生 fused primitive，就应重新评估。

## 性能、并发、正确性与边界条件

- 写入量仅与本轮 token 数成正比，读取量与各请求 <code>seq_lens</code> 成正比；decode 的主要带宽压力通常在 paged read。
- physical blocks 可任意离散，避免搬移历史 KV，但 block table indirection 降低地址连续性；block size 是碎片率、页表长度和 kernel 访问粒度的共同参数。
- GQA 让多个 Q heads 共享较少 KV heads，cache 容量下降，但 backend 必须保持 TP 后 local head mapping 一致。
- full graph 依赖 fixed-address metadata buffer；同时还依赖 padding 全部无副作用。<code>PAD_SLOT_ID</code> 是 correctness contract，不只是性能占位。
- Scheduler/ModelRunner 串行更新 metadata；多个 layer 可共享 per-group metadata，但各 layer 的 cache storage 不共享。任何并行 cache update 都必须证明不同 layer 或 token 的写区间不别名。
- DCP 下非本 rank token 会被映射为 PAD，slot 公式包含 interleave/rank 语义，不能直接套用单 rank 等式。

## 测试证据与未覆盖风险

[backend correctness test](https://github.com/vllm-project/vllm/blob/c296851a7d173fa89d2eefbca0243be42ae9b5e0/tests/v1/attention/test_attention_backends.py#L120-L223) 会随机化 physical blocks，预填 context cache，并按与生产代码相同的 block/offset 规则构造 slot mapping。随后测试对 prefill、decode、mixed batches 以及多种 TP head partition、KV dtype 执行 backend；对于 split backend，明确先 <code>do_kv_cache_update()</code> 再 <code>forward()</code>，并与连续序列 reference 比较 shape、dtype、有限性和数值。

这是**仓库测试事实**：随机物理页下，写入与 paged read 的组合能得到 reference-compatible 输出。本文没有在本环境运行 GPU 测试。

未覆盖风险：

1. generic correctness test 没有直接证明 <code>torch.compile</code>/full CUDA Graph 下 dummy dependency 始终保持写读顺序；
2. 没有专门注入重复 slot、跨 request slot alias 或 stale padding slot；
3. 多 KV group、DCP interleave 与真实多进程 TP 的组合边界仍需 backend/device CI；
4. backend 各自可能选择 fused update、不同 cache layout，公共 contract 漂移时容易出现“一类 backend 通过、另一类静默错写”。

最小 CI guard 应 capture 并 replay 两个不同 batch：第二次缩小 actual tokens，故意让 padding 区保留上一轮模式；断言 cache 只有新请求的有效 slots 改变，Attention 输出与 eager reference 一致。同时加入跨请求不相交断言，保证两个 request 的 slot sets 无交集。

## 修改该区域时的影响面

1. 改 <code>CommonAttentionMetadata</code>：检查所有 backend builder、full/piecewise graph capture、KV groups、PP/TP/DCP。
2. 改 slot 公式或 block size：同步检查 Scheduler allocation、BlockTables、cache layout、prefix reuse、reshape-and-cache kernel 和 tests。
3. 改 cache update 顺序：检查 eager、<code>torch.compile</code>、CUDA Graph、custom op fake/meta implementation 与 backend fused paths。
4. 改 <code>seq_lens</code>：区分每请求精确长度与 capture 上界，检查 spec decode、chunked prefill、encoder-decoder 和 sliding window。
5. 改 padding：同时验证 slot、block-table rows、query starts、output slicing，不能只清一个 buffer。

## 与前后章节的连接

上一章的返回事务从 Attention/Sampling 之后向 Scheduler 收敛；本章补上正向执行中最关键的 device-state 边界：计划怎样变成真实 KV 写与 paged read。下一章进入 Sampling，追踪 attention hidden states 如何经过 logits processor、RNG 与 sampler kernel，成为第 08 章提交的 <code>sampled_token_ids</code>。

## 本篇结论、知识债与理解检查

核心结论：**Attention metadata 不是一张地址表，而是三个索引空间之间的联合契约。<code>slot_mapping</code> 只路由当前 K/V 写入；<code>block_table + seq_lens</code> 只定义历史 KV 读取；<code>query_start_loc</code> 定义 packed rows 的请求边界。**

新增知识债：compiled/full-graph 写读顺序缺少专门 CI；跨请求 slot alias 缺少负向测试；不同 backend 的 fused/split update 与 cache layout 漂移尚未形成统一 contract suite；DCP 与多 KV group 的写读一致性仍需独立章节。

三个检查问题：

1. 为什么 <code>slot_mapping=[49,144]</code> 不能告诉 Attention R0 的完整 18-token 历史在哪里？
2. 若 graph padding token 保留了上一 step 的有效 slot，会造成什么类型的静默破坏？
3. cache update 已经在 Python 调用顺序上位于 attention 前，为什么 compiled graph 仍需要 dummy dependency？

## 课程账本增量

- 完成：<code>SchedulerOutput → BlockTables → Common/FlashAttentionMetadata → cache scatter write → paged KV read</code>。
- 新不变量：三个 metadata 索引空间职责不可混用；当前 K/V 必须在 causal read 前可见；padding slot 必须无副作用；write metadata 与 read metadata 必须来自同一物理 allocation。
- 下一章：Sampling 的 logits processor、RNG、logprobs 与 sampler kernel。
