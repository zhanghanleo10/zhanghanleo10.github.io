# DeepSeek V4 模型结构课程账本

## 总体路线

`Tokenizer → Embedding → mHC 四流残差 → 标准 Attention → KV Cache → V4 Hybrid Compressed Attention → DeepSeek MoE → mHC Head / LM Head / Sampling → Prefill / Decode → vLLM 物理缓存与内核优化`

## 当前阶段

- 阶段 1：建立从 token ID 到下一个 token 的完整模型结构与推理状态心智模型。
- 当前主线：已完成 8 篇基础到进阶教程，并发布公开系列总览。
- 下一阶段：固定 vLLM commit，从模型语义下钻到 `deepseek_v4` 的 Attention、Compressor、Hybrid KV Cache、MoE 与 MTP 源码。

## 已完成章节

| 日期/章节 | 主题 | 主要基线 | 文章 |
| --- | --- | --- | --- |
| 2026-08-13 01 | token ID、Embedding 查表、词表并行、四流展开 | DeepSeek V4-Pro config / reference model | [Embedding]({{ '/articles/deepseek-v4-01-embedding/' | relative_url }}) |
| 2026-08-13 02 | mHC Head、RMSNorm、词表投影、logits、Softmax、采样 | DeepSeek V4-Pro `ParallelHead` | [LM Head]({{ '/articles/deepseek-v4-02-lm-head/' | relative_url }}) |
| 2026-08-13 03 | 四流残差、24 个动态路由量、Sinkhorn、`hc_pre/hc_post` | mHC paper / DeepSeek kernel | [mHC]({{ '/articles/deepseek-v4-03-mhc/' | relative_url }}) |
| 2026-08-13 04 | Q/K/V、缩放点积、因果 Mask、多头、RoPE | Attention / RoFormer | [标准 Attention]({{ '/articles/deepseek-v4-04-standard-attention/' | relative_url }}) |
| 2026-08-13 05 | Prefill、Decode、KV Cache 生命周期、PagedAttention、Prefix Cache | vLLM / PagedAttention | [KV Cache]({{ '/articles/deepseek-v4-05-kv-cache/' | relative_url }}) |
| 2026-08-13 06 | shared K=V、inverse RoPE、SWA、C4A、C128A、Indexer、五类状态 | DeepSeek reference / vLLM [`6accb779`](https://github.com/vllm-project/vllm/commit/6accb779a361c723cded3f9422b48d3fe4da0901) | [混合压缩注意力]({{ '/articles/deepseek-v4-06-hybrid-compressed-attention/' | relative_url }}) |
| 2026-08-13 07 | 384 experts、Top-6、shared expert、Hash routing、FP4、Expert Parallel | DeepSeek V4 report / reference model | [DeepSeek MoE]({{ '/articles/deepseek-v4-07-moe/' | relative_url }}) |
| 2026-08-13 08 | 完整 61 层数据流、请求状态、通信边界、MTP 定位 | DeepSeek V4 / vLLM [`6accb779`](https://github.com/vllm-project/vllm/commit/6accb779a361c723cded3f9422b48d3fe4da0901) | [完整推理链]({{ '/articles/deepseek-v4-08-end-to-end-inference/' | relative_url }}) |

公开入口：[DeepSeek V4 模型结构学习路线]({{ '/articles/deepseek-v4-model-architecture-learning-series/' | relative_url }})

## 既有专题

- [DeepSeek V4 KV Cache 与 vLLM 初学者指南]({{ '/articles/deepseek-v4-kvcache-vllm-beginner-guide/' | relative_url }})：五类缓存、Packed Slab、BlockPool、Prefix Cache 与物理页面深挖。
- [DeepSeek V4 SWA Page Run / TLoad]({{ '/articles/pto-deepseek-v4-swa-page-run-tload/' | relative_url }})：更下层的页面加载与 kernel 专题。

## 已覆盖符号与模块

- `ParallelEmbedding`
- `ParallelHead`
- `RMSNorm`
- `Block.hc_pre`
- `Block.hc_post`
- `ParallelHead.hc_head`
- `hc_split_sinkhorn`
- `Attention.wq_a/q_norm/wq_b`
- `Attention.wkv/kv_norm`
- `Compressor`
- `Indexer`
- `sparse_attn`
- `attn_sink`
- inverse RoPE
- C4A / C128A / SWA
- main compressor state / main compressed KV
- indexer compressor state / Index K
- `Gate`
- `tid2eid`
- correction bias
- `Expert`
- shared expert
- Expert Parallel dispatch / combine
- vocab-parallel logits
- Prefill / Decode / packed token layout
- MTP 的主链定位

## 已确认不变量

1. DeepSeek-V4-Pro 的词表大小是 129,280，隐藏维度 7,168，主干 Block 数 61，最大位置数 1,048,576。
2. Embedding 与 LM Head 的逻辑权重形状同为 `[129280,7168]`，但 `tie_word_embeddings=false`，参数不共享。
3. mHC 维护 4 条残差流；每个 Attention/MoE 子层只汇出一条 7,168 维工作流执行主模块，再动态写回四流。
4. `hc_pre` 每 token 产生 `4 pre + 4 post + 16 comb`；24 个量是激活，不是 checkpoint 中新增的 24 个标量参数。
5. `pre` 与 `post` 不是概率分布；`comb` 通过 20 次 Sinkhorn 迭代近似满足行列和均为 1。
6. 普通自回归 Decode 只需要当前 Q；历史 K/V 会被未来 Query 重复读取，因此跨步缓存。
7. V4 的 shared KV 是同一个 512D 向量同时作为 K 和 V，不只是普通 MQA；仅最后 64 维使用 RoPE。
8. shared K=V 后，输出的 RoPE 子空间需按当前 Query 位置 inverse RoPE，得到 query-relative 表示。
9. C4A 的长期密度约为 1/4，但单槽通常覆盖 8 token、stride 4；C128A 覆盖非重叠 128-token 窗口。
10. 压缩槽只有在窗口完全形成后才因果可见：C4 为 `i >= 4j+3`，C128 为 `i >= 128j+127`。
11. C4A 有独立的主 Compressor 和 Indexer Compressor；Index score 只决定候选，主 Attention score 决定内容权重。
12. Pro 的 C4 `index_topk=1024`；C128 没有学习式 Indexer，在 1M 上下文下最多读取 8,192 个因果可见锚点。
13. Pro 61 个主干层包含 31 个 C128A 与 30 个 C4A，没有 SWA-only 主干层；`compress_ratios` 末尾 0 属于 MTP。
14. C4A 最大有五类请求状态；C128A 没有 Indexer 的两类状态。Top-k 索引是当前 Query 的临时读取计划，不是长期 cache。
15. MoE 每层有 384 routed experts，每 token 选择 6 个，另加 1 个始终执行的 shared expert。
16. 非 Hash 层用 `Top6(s + correction_bias)` 选择专家，但最终权重从原始 `s` 中取；归一化后乘 2.5，所以 Top-6 权重和为 2.5。
17. 前 3 层由每层独立 `tid2eid` 表确定专家 IDs，但当前 hidden 仍决定选中专家的混合权重。
18. FP4 明确用于 routed expert 权重，不能把整个模型、激活和累加一概称为 FP4。
19. MoE 没有请求级历史 cache；每个新 token、每层都会重新路由并执行专家。
20. vLLM 的融合、并发、分页和量化改变执行计划与物理布局，不改变上述模型语义。

## 版本与证据边界

- 模型超参数以当前 [DeepSeek-V4-Pro config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json) 为准。
- 可读模型语义以 DeepSeek 官方 [`inference/model.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py) 与 [`inference/kernel.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py) 为准。
- 训练设计与系统结果参考 [DeepSeek-V4 技术报告](https://arxiv.org/html/2606.19348)。
- vLLM 具体 block/page/cache 数字固定在 commit [`6accb779`](https://github.com/vllm-project/vllm/commit/6accb779a361c723cded3f9422b48d3fe4da0901)，后续课程每章继续固定 commit，避免把 main 漂移当成稳定事实。
- 参考实现中的简单 `all_reduce/all_gather` 用于理解语义；生产 vLLM 的 TP/EP dispatch、融合与量化路径需按固定源码单独验证。

## 尚未解释的知识债

- DeepSeek V4 技术报告中的训练目标、数据与长上下文扩展过程。
- Q LoRA、分组 O LoRA 与 tensor-parallel shard 的精确矩阵/通信布局。
- `attn_sink` 在训练与长上下文稳定性中的作用，以及与其他 attention sink 方案的差异。
- C4 vector gate、APE、overlap transform 在 vLLM fused compressor kernel 中的具体内存布局。
- Indexer 的 Hadamard rotation、FP4 cache、打分归约与 Top-k kernel。
- `fp8_ds_mla` 的 512D 语义行如何映射为后端物理 entry 与 scale。
- Hybrid KV Cache Manager 如何注册五类 cache spec、统一 256-native-position block，并组成三个 page-size bucket。
- Prefix Cache 命中压缩边界时，Compressor state 的恢复/传输不变量。
- C4/C128 Prefill 与 Decode 的 bitwise / numerical equivalence 测试。
- MegaMoE 的 token sort、wave pipeline、FP8×FP4 GEMM、EPLB 与冗余专家布局。
- mHC 融合 kernel 如何组合 `hc_post + hc_pre + RMSNorm`，以及 FP32 路由计算的数值边界。
- LM Head 的 distributed sampling、局部 Top-k 与跨 rank 合并是否避免完整 logits materialization。
- MTP draft、target verification、接受率和 cache 更新/回滚流程。
- disaggregated Prefill/Decode 下五类状态的传输与 ownership。
- CUDA Graph、torch.compile 与动态 cache metadata 的图捕获边界。

## 下一批候选章节

1. 源码 01：`vllm/models/deepseek_v4/attention.py`，把 Query、SWA、C4/C128 metadata 与 sparse MLA 调用逐行映射。
2. 源码 02：`compressor.py`，追踪 Prefill/Decode 的 overlap state、压缩边界和 cache insertion。
3. 源码 03：Hybrid KV Cache spec、256-native-token block、三类 page bucket 与 prefix cache。
4. 源码 04：C4 Indexer 的 FP4 Index K、Top-k 与 multi-stream overlap。
5. 源码 05：mHC fused kernel，从数学四步恢复融合读写路径。
6. 源码 06：FusedMoE / MegaMoE、Expert Parallel dispatch/combine 与 FP4 权重。
7. 源码 07：Vocab-parallel LM Head、distributed sampling 与 logits 内存优化。
8. 源码 08：MTP speculative decoding 的 draft/verify/cache 状态机。
9. 性能篇：分别建立 Prefill 与 Decode 的 FLOPs、HBM bytes、网络 bytes 与 roofline 模型。
10. 验证篇：用微型张量和小 batch 构造 mHC、压缩边界、Indexer、MoE routing 的 golden tests。
