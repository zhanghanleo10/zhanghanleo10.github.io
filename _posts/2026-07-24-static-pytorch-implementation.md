---
layout: post
title: "STATIC PyTorch 实现解析：把 Trie 约束解码变成 GPU 向量化查表"
description: "从约束函数、Dense/CSR 混合索引到 VNTK 与 Beam Search 重排，逐段对齐 STATIC 官方 PyTorch 实现。"
date: 2026-07-24 00:30:00 +0800
category: "生成式推荐"
tags:
  - STATIC
  - PyTorch
  - 约束解码
  - Beam Search
  - 生成式推荐
reading_time: "约 18 分钟"
math: true
---

> 本文只分析 STATIC 的官方 PyTorch 实现，以及它与真实 GPU 推理系统的衔接方式。

STATIC 的全称是 **Sparse Transition Matrix-Accelerated Trie Index for Constrained Decoding**。它解决的不是“如何让模型预测得更准”，而是一个非常工程化的问题：

> 在生成式推荐中，怎样保证模型生成的 Semantic ID 一定属于当前允许召回的集合，同时又不让 Trie 查询拖慢 GPU 推理？

论文的答案是：保留 Trie 的语义，但不在在线阶段遍历指针结构；提前把 Trie 压平为 Dense 表和 CSR 转移表，再用 PyTorch 的张量索引一次取出所有 Beam 的合法后继。

本文对应的主要代码是：

- [`static_decoding/csr_utils.py`](https://github.com/youtube/static-constraint-decoding/blob/main/static_decoding/csr_utils.py)：离线构建 STATIC 索引；
- [`static_decoding/decoding_pt.py`](https://github.com/youtube/static-constraint-decoding/blob/main/static_decoding/decoding_pt.py)：PyTorch 约束 Beam Search 与稀疏候选提取；
- [`tests/test_pt_decoding.py`](https://github.com/youtube/static-constraint-decoding/blob/main/tests/test_pt_decoding.py)：端到端合法性测试；
- [`benchmarks/run_branch_benchmark_pt.py`](https://github.com/youtube/static-constraint-decoding/blob/main/benchmarks/run_branch_benchmark_pt.py)：PyTorch GPU Kernel 的分支规模测试。

## 1. 为什么普通 Trie 不适合 GPU

假设每个物品由固定长度为 $L$ 的 Semantic ID 表示：

$$
\mathbf{y}=(y_1,y_2,\ldots,y_L),\qquad y_t\in\mathcal{V}
$$

当前业务允许召回的物品集合记作：

$$
\mathcal{C}\subset\mathcal{V}^{L}
$$

在第 $t$ 个解码位置，约束函数为：

$$
F_t(y_{<t},y_t)
=
\mathbb{I}\left(
\exists c\in\mathcal{C},
(y_{<t},y_t)\sqsubseteq c
\right)
$$

它表达的事情很简单：把候选 token $y_t$ 接到当前前缀后，如果仍然是某个合法 SID 的前缀，就保留；否则把对应 log-probability 设为 $-\infty$。

CPU Trie 通常通过“当前节点 → 子节点指针 → 下一个节点”完成判断。问题是不同 Beam 位于不同节点，每个节点的子节点数量也不同。这会带来：

1. 不连续的随机访存；
2. CPU 与 GPU 每一步同步；
3. 数据相关的循环和分支；
4. GPU Warp 中不同线程工作量不一致。

STATIC 没有改变约束解码的数学结果，而是改变了约束索引的内存布局和执行方式。

## 2. 一个贯穿全文的例子

令词表为：

$$
\mathcal{V}=\{0,1,2,3\}
$$

合法 SID 集合为：

$$
\mathcal{C}
=
\{(1,2,1),(3,1,2),(3,1,3)\}
$$

解码时：

- 第一个 token 只能是 `1` 或 `3`；
- 前缀为 `(1)` 时，第二个 token 只能是 `2`；
- 前缀为 `(3)` 时，第二个 token 只能是 `1`；
- 前缀为 `(1,2)` 时，最后只能选 `1`；
- 前缀为 `(3,1)` 时，最后可以选 `2` 或 `3`。

STATIC 会为它构建三类数据：

| 数据 | 作用 | 示例 |
| --- | --- | --- |
| `start_mask[V]` | 限制第一个 token | 仅位置 `1`、`3` 为 `True` |
| `dense_mask[V,V]` | 快速判断前两个 token 的组合 | `(1,2)`、`(3,1)` 为合法前缀 |
| `packed_csr + indptr` | 保存更深层的合法转移 | `(3,1)` 对应后继 token `2`、`3` |

这里的关键不是把所有路径做成一个巨大的 $V^L$ 张量，而是只把很浅、访问频繁的头部做成 Dense 表，把后面高基数但高度稀疏的部分放入 CSR。

## 3. 离线阶段：`build_static_index`

官方 PyTorch 路径使用 NumPy 构建器：

```python
def build_static_index(
    fresh_sids: np.ndarray,
    vocab_size: int = 2048,
    dense_lookup_layers: int = 2,
):
    ...
```

输入 `fresh_sids` 的形状为 `[N, L]`：

- $N$：约束集合中的 SID 数；
- $L$：每条 SID 的固定长度；
- $V$：`vocab_size`。

输入必须按字典序排列。测试代码先做：

```python
sids = sids[
    np.lexsort([sids[:, i] for i in range(sid_len - 1, -1, -1)])
]
```

排序后，相同前缀会连续出现，构建器才能用相邻行差异识别新的 Trie 节点。

### 3.1 找出每一层的新前缀

```python
diff = fresh_sids[1:] != fresh_sids[:-1]
first_diff = np.full(N - 1, L, dtype=np.int8)
has_diff = diff.any(axis=1)
first_diff[has_diff] = diff[has_diff].argmax(axis=1)
```

`first_diff[i]` 表示第 `i+1` 条 SID 与前一条 SID 第一次出现差异的位置。

随后构造：

```python
is_new = np.zeros((N, L), dtype=bool)
is_new[0, :] = True
for depth in range(L):
    is_new[1:, depth] = first_diff <= depth
```

如果两条 SID 在第 $k$ 层首次不同，那么从第 $k$ 层开始，它们对应的前缀节点都是新的。这样就把逐条插入 Trie 的过程改成了数组上的批量比较。

### 3.2 为前缀节点分配 State ID

每一个唯一前缀都对应一个状态 $s$。概念上的转移矩阵为：

$$
T\in\mathbb{Z}^{S\times |\mathcal{V}|}
$$

其中：

$$
T_{s,v}=
\begin{cases}
s_{\text{next}}, & \text{当前状态存在 token }v\text{ 的转移}\\
0, & \text{不存在该转移或已到终止状态}
\end{cases}
$$

第一层状态直接采用 `token + 1`：

```python
state_ids[:, 0] = fresh_sids[:, 0].astype(np.int32) + 1
```

`0` 被留作终止或哨兵状态。更深的唯一前缀从 `vocab_size + 1` 开始顺序编号：

```python
state_ids[mask, depth] = np.arange(
    start_id, end_id, dtype=np.int32
)
state_ids[:, depth] = np.maximum.accumulate(
    state_ids[:, depth]
)
```

`maximum.accumulate` 会把新分配的状态编号传播给后续具有相同前缀的 SID。

### 3.3 收集 `(父状态, token, 子状态)` 边

```python
parent_ids = state_ids[mask, depth - 1]
token_ids = fresh_sids[mask, depth].astype(np.int32)
child_ids = (
    state_ids[mask, depth]
    if depth < L - 1
    else np.zeros_like(parent_ids, dtype=np.int32)
)
```

最后一层的 `child_ids` 为 `0`，因为完整 SID 已经生成完毕，不再需要继续转移。

### 3.4 Dense 头部

默认 `dense_lookup_layers=2`，因此：

```python
dense_shape = (vocab_size, vocab_size)
dense_mask = np.zeros(dense_shape, dtype=bool)
dense_states = np.zeros(dense_shape, dtype=np.int32)

dense_mask[first_token, second_token] = True
dense_states[first_token, second_token] = state_after_two_tokens
```

Dense 表的优点是索引简单、延迟稳定；缺点是空间复杂度为：

$$
O(V^d)
$$

因此官方实现说明实际主要支持 $d=1$ 或 $d=2$。继续增加 Dense 深度，会很快出现不可接受的内存开销。

### 3.5 CSR 稀疏尾部

CSR 使用两部分数据：

```python
indptr
packed_csr[:, 0]  # token
packed_csr[:, 1]  # next_state
```

状态 `s` 的全部合法边位于：

```python
start = indptr[s]
end = indptr[s + 1]
children = packed_csr[start:end]
```

与标准 CSR 的 `column_indices` 和 `values` 分离存储不同，官方实现把 `(token, next_state)` 放在同一行：

```python
packed_csr = np.ascontiguousarray(
    np.vstack([raw_indices, raw_data]).T
)
```

这样一次连续读取就能同时拿到合法 token 和对应的下一状态。

构建器还会统计每一层的最大分支数：

$$
K_\ell=\max_{n\in\mathcal{N}_\ell} b_n^\ell
$$

它对应 `layer_max_branches`。在线 Kernel 不按每个节点的真实分支数启动不同形状的计算，而是让同一层统一读取 $K_\ell$ 个槽位，再使用 `valid_mask` 屏蔽多读出来的部分。

最后，`packed_csr` 末尾会追加 $V$ 行 Padding：

```python
raw_indices = np.concatenate(
    [tokens, np.full(vocab_size, vocab_size, dtype=np.int32)]
)
raw_data = np.concatenate(
    [children, np.zeros(vocab_size, dtype=np.int32)]
)
```

Padding token 使用越界值 `vocab_size`，下一状态使用 `0`。它为固定长度读取提供了安全区域。

## 4. 在线阶段：PyTorch 解码主流程

`sparse_transition_torch` 可以分成四段：

1. 模型产生 log-probabilities；
2. Dense 或 CSR 路径提取合法候选；
3. 将候选分数与父 Beam 累积分数相加；
4. 全局选择新的 Top-$M$ Beam，并同步重排 token 历史和 Trie 状态。

主要张量形状如下：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `flat_logprobs` | `[B*M, V]` | 每条 Beam 的全词表分数 |
| `flat_states` | `[B*M]` | 每条 Beam 当前所在 Trie 状态 |
| 候选分数 | `[B*M, K]` | 每条 Beam 的合法后继分数 |
| `scores` | `[B, M, K]` | 加上父 Beam 累积分数 |
| `flat_scores` | `[B, M*K]` | 全局 Beam 竞争池 |
| `token_buffer` | `[B, M, L]` | 当前保留的完整生成历史 |

### 4.1 第一个 token：`start_mask`

第一步从 `[B,V]` 的模型分数开始：

```python
raw_logprobs = F.log_softmax(initial_logits[:, 0, :], dim=-1)
initial_logprobs = torch.where(
    start_mask,
    raw_logprobs,
    torch.tensor(-float("inf"), device=device),
)
top_logprobs, top_tokens = torch.topk(
    initial_logprobs, beam_size, dim=-1
)
```

随后通过：

```python
current_transition_states = top_tokens + 1
```

把第一层 token 映射到 Trie 状态编号。

### 4.2 第二个 token：Dense 快速路径

默认 $d=2$ 时，第二步直接按第一层 token 取 Dense 行：

```python
parent_tokens = (flat_states - 1).long()
masks = dense_mask[parent_tokens]
flat_logprobs = torch.where(masks, flat_logprobs, neg_inf)
```

然后先在每条 Beam 内取候选：

```python
topk_logprobs, topk_indices = torch.topk(
    flat_logprobs, tokens_per_beam, dim=-1
)
```

并用相同 token 下标取出下一状态：

```python
next_state_candidates = dense_states[
    parent_tokens.unsqueeze(1),
    topk_indices.long(),
]
```

Dense 阶段同时完成了“token 是否合法”和“合法 token 到哪个状态”两个查询。

### 4.3 更深层：CSR/VNTK 稀疏路径

从更深的位置开始：

```python
limit = max_branch_factors[step + 1]
candidates_logprobs, candidates_indices, candidates_states = (
    generate_and_apply_logprobs_mask(...)
)
```

这里的 `limit` 就是当前层最大分支数 $K_\ell$。这使输出形状只与层号相关，不再取决于每条 Beam 当前节点的真实度数。

## 5. VNTK：STATIC 的核心 PyTorch Kernel

`generate_and_apply_logprobs_mask` 是整条路径的核心。为了便于理解，令：

$$
Q=B\times M
$$

表示当前并行处理的 Beam 总数，当前层最大分支数为 $K$。

### 5.1 查找每条 CSR Row 的边界

```python
starts = csr_indptr[flat_states.long()]
actual_lens = (
    csr_indptr[flat_states.long() + 1] - starts
)
```

`starts` 和 `actual_lens` 的形状都是 `[Q]`。它们分别表示每条 Beam 当前状态对应 CSR Row 的起始位置和真实子节点数。

### 5.2 构造固定形状读取网格

```python
offsets = torch.arange(limit, device=device)
gather_indices = (
    starts.unsqueeze(1) + offsets.unsqueeze(0)
)
```

形状变化为：

$$
[Q] + [K]\longrightarrow[Q,K]
$$

即每条 Beam 都尝试读取 $K$ 个连续位置。真实只有 $k_i<K$ 个孩子的节点，也会读取到相邻数据或 Padding 区，但这些值不会进入最终候选。

```python
safe_gather_indices = gather_indices.clamp(
    max=packed_csr.size(0) - 1
)
gathered_vals = packed_csr[safe_gather_indices]
```

`gathered_vals` 的形状为 `[Q,K,2]`，最后一维分别是 token 和 next state。

### 5.3 用算术掩码消除动态分支

```python
valid_mask = (
    offsets.unsqueeze(0) < actual_lens.unsqueeze(1)
)
```

第 $i$ 条 Beam 只有前 $k_i$ 个槽位为 `True`。这一步把“循环读取不同数量的孩子”改写成“固定读取 $K$ 个槽位，再做张量比较”。

### 5.4 只 Gather 合法候选的 LogProb

```python
safe_token_ids = candidate_token_ids.long().clamp(
    max=vocab_size - 1
)
candidate_logprobs = flat_logprobs.gather(
    1, safe_token_ids
)
candidate_logprobs = torch.where(
    valid_mask,
    candidate_logprobs,
    torch.tensor(-float("inf"), device=device),
)
```

这一点很关键：PyTorch 实现没有先为每条 Beam 构造完整的 `[V]` 合法性张量，再对全词表做一次 Mask；它直接从 `[Q,V]` 中 Gather 当前状态允许的 $K$ 个 token 分数，得到 `[Q,K]` 候选集。

因此稀疏约束部分的主要在线工作量与 $QK$ 相关，而不是为 Trie 中的每个节点做遍历。论文所说的“相对约束集合规模 $|\mathcal{C}|$ 的 $O(1)$ I/O”，指的是单步只访问当前状态对应的 CSR Row，不随全部约束数量执行二分查找；它不代表 Kernel 成本与分支数 $K$ 无关。

## 6. Beam Search 如何与 Trie 状态保持一致

每条父 Beam 最多产生 $K$ 个候选：

```python
scores = (
    current_token_scores.unsqueeze(2)
    + candidates_logprobs.view(B, M, K)
)
flat_scores = scores.view(B, M * K)
```

随后在整个 `M*K` 候选池中选新的 Top-$M$：

```python
top_scores, flat_top_indices = torch.topk(
    flat_scores, beam_size, dim=-1
)
```

扁平候选编号可以恢复父 Beam：

```python
top_beam_indices = flat_top_indices // limit
```

接下来必须同步更新三类状态：

1. 新选中的 token；
2. 新 token 对应的 Trie `next_state`；
3. 新 Beam 继承的历史 token 序列。

```python
top_tokens = _gather_beams(
    flat_tokens, flat_top_indices
)
current_transition_states = _gather_beams(
    flat_next_states, flat_top_indices
)
token_buffer = _gather_beams(
    token_buffer, top_beam_indices
)
token_buffer[:, :, step + 1] = top_tokens
```

其中 `_gather_beams` 会根据输入张量的维度扩展索引：

```python
view_shape = [batch_size, new_beam_size] + [1] * (x.dim() - 2)
expand_shape = [batch_size, new_beam_size] + list(x.shape[2:])
indices = beam_indices.view(view_shape).expand(expand_shape)
return x.gather(1, indices)
```

这保证 Beam 重排之后，生成历史和 Trie 状态仍然属于同一条路径。如果只重排分数和 token、忘记重排 `current_transition_states`，下一步约束就会在错误的 Trie 节点上查询。

## 7. 如何运行官方 PyTorch 复现

```bash
git clone https://github.com/youtube/static-constraint-decoding.git
cd static-constraint-decoding
pip install -e .
python tests/test_pt_decoding.py
```

测试流程为：

1. 随机生成 `[N,L]` 的 SID；
2. 去重并按字典序排序；
3. 调用 `build_static_index`；
4. 把索引转换为 GPU/CPU 上的 PyTorch Tensor；
5. 用 `RandomModel` 执行约束 Beam Search；
6. 检查每一条输出 SID 都在原始约束集合中。

官方测试覆盖两组配置：

```python
configs = [
    (100, 50, 5, 10, 2),
    (50, 20, 8, 5, 1),
]
```

分别验证 $d=2$ 与 $d=1$ 的 Dense 配置。这个测试证明的是**生成结果合法性**，不是实际大模型的端到端性能。

## 8. PyTorch GPU Benchmark 在测什么

`run_branch_benchmark_pt.py` 单独测试：

```python
generate_and_apply_logprobs_mask
```

它使用：

- `num_sequences=1_000_000`；
- SID 长度 `L=8`；
- `batch_beam=2`；
- 分支数从 $2$ 扩展到 $262144$；
- `torch.compile(..., mode="reduce-overhead")`；
- GPU 场景下利用 CUDA Graph 降低 CPU Launch 开销；
- 每组配置执行 1000 次以统计 Kernel 时间。

因此，这个脚本反映的是 **CSR 候选提取 Kernel 随最大分支数变化的趋势**，不能直接代表真实模型的请求吞吐、首 Token 延迟或完整 Beam Search 延迟。

论文报告的 `0.033 ms/step`、`0.25%` 推理时间占比、相对 CPU Trie 的 `948×` 加速以及相对 PPV 的 `47–1033×` 加速，来自论文中的生产与实验设置，不应直接当作任意 PyTorch GPU 环境都能复现的结果。

## 9. 接入真实 PyTorch 推理系统时还缺什么

官方 `sparse_transition_torch` 是一个清晰的算法 Harness，但不是完整的生产推理引擎。

### 9.1 `RandomModel` 没有上下文和 KV Cache

示例模型只接收上一轮 token：

```python
flat_logits = model(flat_input_ids)
```

真实 Transformer 必须同时维护：

- Prompt 的 KV Cache；
- 每条 Beam 的 Decode KV；
- Beam 父子重排后的 KV 映射；
- 新请求加入和已完成请求退出后的 Batch 状态。

因此，STATIC 解决的是“候选是否合法以及下一状态是什么”，并不自动解决完整的大模型 Beam Serving。

### 9.2 Beam 重排必须传递到模型状态

代码已经通过 `top_beam_indices` 重排 `token_buffer`，真实系统还必须用同一父 Beam 映射处理 KV Cache。否则 token 历史看似正确，模型实际使用的 KV 却来自另一条父路径。

### 9.3 固定 $K_\ell$ 带来填充开销

统一使用层最大分支数能得到静态形状，但当同一层节点度数差异很大时，很多槽位只是在做 Padding 读取与 Mask。

实际部署需要观察：

$$
\frac{\text{平均分支数}}{\text{最大分支数}}
$$

如果这个比例很低，可以考虑按分支规模分桶、为不同层编译不同 Kernel，或进一步做专用 CUDA/Triton 实现。

### 9.4 索引更新是离线操作

STATIC 的优势来自“静态”。当 freshness、库存或地域约束集合变化时，需要重新生成索引并安全切换设备上的 Tensor。生产系统还要设计：

- 索引版本号；
- 后台构建与校验；
- HBM 双缓冲或分批替换；
- 请求与索引版本的一致性；
- 失败时回滚。

### 9.5 数据类型与显存

NumPy 构建器使用 `int32`。官方测试把 `packed_csr` 和 `indptr` 都转成 `torch.long`，而 Benchmark 使用 `int32` 的 `packed_csr` 和 `long` 的 `indptr`。

真实部署应根据状态数量和框架索引要求分别评估：

- token ID 是否可以保持 `int32`；
- next state 是否超过 `int32`；
- 哪些张量必须在 Gather 前临时转为 `long`；
- Dense 表与 CSR 表各自的显存占用。

## 10. 总结

STATIC 的核心可以压缩成一句话：

> 用 State ID 代替 Trie 指针，用 Dense/CSR 代替动态树结构，用固定形状 Gather 与 Mask 代替逐 Beam 分支遍历。

在官方 PyTorch 实现中，最值得抓住的是三条线：

1. `build_static_index` 把排好序的 SID 集合变成 `start_mask + dense_mask/dense_states + packed_csr/indptr`；
2. `generate_and_apply_logprobs_mask` 用 `[Q,K]` 固定读取网格一次提取所有 Beam 的合法候选；
3. `sparse_transition_torch` 在全局 Top-$M$ 之后同步重排 token 历史和 Trie State。

它非常适合作为生成式推荐约束解码的底层组件，但要接入 vLLM 或自研 PyTorch Serving，仍需把 STATIC State 与 Scheduler、Beam 生命周期、KV Cache 重排和索引热更新完整地连接起来。

## 参考资料

- [论文：Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators](https://arxiv.org/abs/2602.22647)
- [STATIC 官方代码仓库](https://github.com/youtube/static-constraint-decoding)
- [PyTorch 解码实现](https://github.com/youtube/static-constraint-decoding/blob/main/static_decoding/decoding_pt.py)
- [STATIC 索引构建实现](https://github.com/youtube/static-constraint-decoding/blob/main/static_decoding/csr_utils.py)
- [PyTorch 合法性测试](https://github.com/youtube/static-constraint-decoding/blob/main/tests/test_pt_decoding.py)
- [PyTorch GPU Benchmark](https://github.com/youtube/static-constraint-decoding/blob/main/benchmarks/run_branch_benchmark_pt.py)
