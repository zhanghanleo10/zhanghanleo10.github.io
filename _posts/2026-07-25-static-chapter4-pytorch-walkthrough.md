---
layout: post
title: "STATIC 论文第 4 章与官方 PyTorch 源码对照：从 Trie 到 CSR 与 VNTK"
description: "围绕 STATIC 第 4 章的方法设计，用同一个 SID 例子逐步推导约束函数、State ID、row_ptr、Dense/CSR 混合索引、VNTK 和 Beam 状态更新，并与官方 PyTorch 实现逐段对应。"
date: 2026-07-25 12:00:00 +0800
category: "生成式推荐"
tags:
  - STATIC
  - PyTorch
  - 约束解码
  - CSR
  - Beam Search
  - 生成式推荐
reading_time: "约 25 分钟"
math: true
---

> 本文集中分析论文 **Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators** 的第 4 章，并将其中的 Figure 1、Algorithm 1、Algorithm 2 与官方 PyTorch 实现逐段对齐。

相关源码：

- [`static_decoding/csr_utils.py`](https://github.com/youtube/static-constraint-decoding/blob/main/static_decoding/csr_utils.py)：离线构建 STATIC 索引；
- [`static_decoding/decoding_pt.py`](https://github.com/youtube/static-constraint-decoding/blob/main/static_decoding/decoding_pt.py)：PyTorch 约束 Beam Search 与 VNTK；
- [`tests/test_csr_builder.py`](https://github.com/youtube/static-constraint-decoding/blob/main/tests/test_csr_builder.py)：验证 Dense 到 CSR 的路径连通性；
- [`tests/test_pt_decoding.py`](https://github.com/youtube/static-constraint-decoding/blob/main/tests/test_pt_decoding.py)：验证端到端生成结果始终属于约束集合。

上一篇文章更偏向官方 PyTorch 工程结构，本文更关注：

> 论文第 4 章中的每一个数学对象，在源码中到底变成了什么数组、什么 Tensor，以及它们如何在一个解码 Step 内流动。

---

## 1. 第 4 章解决的核心问题

假设每个物品由固定长度为 $L$ 的 Semantic ID 表示：

$$
\mathbf{y}=(y_1,y_2,\ldots,y_L),\qquad y_t\in\mathcal{V}
$$

完整 SID 空间是 $\mathcal{V}^{L}$，但当前业务只允许模型生成其中的一个子集：

$$
\mathcal{C}\subset\mathcal{V}^{L}
$$

这个子集可以表示：

- 最近 7 天上传的视频；
- 当前地区可用的内容；
- 有库存的商品；
- 某个类别下的物品；
- 一个冷启动物品集合。

在第 $t$ 步，论文定义约束函数：

$$
F_t(y_{<t},y_t)
=
\mathbb{I}\left(
\exists c\in\mathcal{C},
(y_{<t},y_t)\sqsubseteq c
\right)
$$

它判断：

> 把候选 token $y_t$ 接到当前前缀 $y_{<t}$ 后，新的前缀是否仍能扩展为某个合法 SID。

若答案是否定的，就把该 token 的 log-probability 设为 $-\infty$。

源码没有显式实现一个 `F_t(prefix, token)` 函数。这个约束函数被编码进三类数据中：

```text
第一个 token 是否合法
    -> start_mask[token]

浅层 prefix 是否合法
    -> dense_mask[prefix tokens]

深层 next token 是否合法
    -> 当前 state 的 CSR Row 中是否存在该 token
```

这就是论文数学定义和源码数据结构之间的第一层映射。

---

## 2. 贯穿全文的最小例子

令 token 词表为：

$$
\mathcal{V}=\{0,1,2,3\}
$$

合法 SID 集合为：

$$
\mathcal{C}
=
\{(1,2,1),(3,1,2),(3,1,3)\}
$$

对应 Trie：

```text
root
├── token 1
│   └── token 2
│       └── token 1 -> terminal
└── token 3
    └── token 1
        ├── token 2 -> terminal
        └── token 3 -> terminal
```

合法后继关系为：

| 当前前缀 | 合法下一 token |
| --- | --- |
| `[]` | `{1, 3}` |
| `[1]` | `{2}` |
| `[3]` | `{1}` |
| `[1, 2]` | `{1}` |
| `[3, 1]` | `{2, 3}` |

传统 Trie 会保存 Node 对象及其 children 指针。STATIC 则会把它转换为：

```text
前缀 -> Integer State ID

每个 State 的出边
-> 一段连续的 [token_id, next_state_id]
```

---

## 3. 论文 4.2：把 Trie 变成转移矩阵

论文为每个唯一前缀分配一个整数状态 $s$，然后定义概念转移矩阵：

$$
T\in\mathbb{Z}^{S\times |\mathcal{V}|}
$$

其中：

$$
T_{s,v}
=
\begin{cases}
 s_{\text{next}}, & \text{状态 }s\text{ 存在 token }v\text{ 的转移}\\
 0, & \text{否则}
\end{cases}
$$

可以把它理解为：

```text
T[current_state, candidate_token] = next_state
```

但这个矩阵绝大多数位置为空。如果有数千万条 SID，直接保存完整 $S\times V$ 矩阵不可行，因此论文采用 CSR。

官方代码进一步做了一个重要压缩：

- 不为每个完整 SID 创建独立叶子状态；
- 最后一个 token 统一转移到 `state 0`；
- `state 0` 表示 terminal/sink。

对固定长度 SID 来说，完整 item 已经由 token history 唯一确定，因此不需要为每个叶子再保存一个不同状态。

---

## 4. `build_static_index`：源码如何识别 Trie 前缀

官方 Builder 的入口是：

```python
def build_static_index(
    fresh_sids: np.ndarray,
    vocab_size: int = 2048,
    dense_lookup_layers: int = 2,
):
    ...
```

输入形状为：

```text
fresh_sids: [N, L]
```

其中：

- $N$：合法 SID 数；
- $L$：SID 长度；
- $V$：`vocab_size`。

### 4.1 为什么输入必须按字典序排序

排序后，相同前缀的 SID 会连续出现：

```text
[1, 2, 1]
[1, 2, 4]
[1, 3, 2]
[3, 1, 2]
```

这样就不需要真的逐条插入 Python Trie，而可以通过相邻行比较识别哪些位置产生了新前缀。

源码先计算：

```python
diff = fresh_sids[1:] != fresh_sids[:-1]
```

对最小例子：

```text
row0 = [1, 2, 1]
row1 = [3, 1, 2]
row2 = [3, 1, 3]
```

相邻差异为：

```text
row1 vs row0 -> [True,  True,  True]
row2 vs row1 -> [False, False, True]
```

第一次发生差异的位置 `first_diff` 为：

```text
[0, 2]
```

随后构造 `is_new[N, L]`：

```python
is_new[1:, depth] = first_diff <= depth
```

结果：

```text
row0: [T, T, T]
row1: [T, T, T]
row2: [F, F, T]
```

含义是：

- `[3,1,3]` 不会再次创建 prefix `[3]`；
- 不会再次创建 prefix `[3,1]`；
- 只需要增加最后一条 token `3` 的边。

---

## 5. State ID 是如何分配的

官方代码对长度为 1 的前缀使用特殊映射：

```python
state_ids[:, 0] = fresh_sids[:, 0] + 1
```

因此：

```text
state 0       = terminal
prefix [0]    = state 1
prefix [1]    = state 2
prefix [2]    = state 3
prefix [3]    = state 4
```

即使某个 token 并不是合法首 token，它对应的 state 编号位置也被保留。这样做的好处是首步状态初始化非常简单：

```python
current_transition_states = top_tokens + 1
```

更深层的唯一前缀从 `vocab_size + 1` 开始连续编号。

对最小例子，最终得到：

```text
state 0        = terminal
state 2        = prefix [1]
state 4        = prefix [3]
state 5        = prefix [1,2]
state 6        = prefix [3,1]
```

完整 SID 最后一层不再创建独立 state，而是通过最后一条边转移到 0。

---

## 6. 收集 `(parent, token, child)` 边

源码按 SID 深度收集边：

```python
parent_ids = state_ids[mask, depth - 1]
token_ids = fresh_sids[mask, depth]
child_ids = (
    state_ids[mask, depth]
    if depth < L - 1
    else np.zeros_like(parent_ids)
)
```

对例子得到：

```text
state 2 -- token 2 --> state 5
state 4 -- token 1 --> state 6
state 5 -- token 1 --> state 0
state 6 -- token 2 --> state 0
state 6 -- token 3 --> state 0
```

如果沿用论文概念图，把 root 也视作一个普通 state，还会有：

```text
root -- token 1 --> prefix [1]
root -- token 3 --> prefix [3]
```

但官方实现把 root 的合法 token 单独存入 `start_mask`，没有把 root Row 放入普通 CSR 查询路径。

---

## 7. `row_ptr / indptr` 到底如何构建

这是理解 CSR 最关键的一步。

假设按 state 顺序统计出边数：

| state | 出边数 |
| ---: | ---: |
| 0 | 0 |
| 1 | 0 |
| 2 | 1 |
| 3 | 0 |
| 4 | 1 |
| 5 | 1 |
| 6 | 2 |

于是：

```text
counts = [0, 0, 1, 0, 1, 1, 2]
```

`row_ptr` 是 `counts` 的前缀和：

```text
row_ptr[0] = 0
row_ptr[i+1] = row_ptr[i] + counts[i]
```

逐步计算：

```text
row_ptr[0] = 0
row_ptr[1] = 0 + 0 = 0
row_ptr[2] = 0 + 0 = 0
row_ptr[3] = 0 + 1 = 1
row_ptr[4] = 1 + 0 = 1
row_ptr[5] = 1 + 1 = 2
row_ptr[6] = 2 + 1 = 3
row_ptr[7] = 3 + 2 = 5
```

最终：

```text
indptr = [0, 0, 0, 1, 1, 2, 3, 5]
```

所有边按父 state 连续排列：

```text
packed_csr[0] = [token 2, next_state 5]  # state 2
packed_csr[1] = [token 1, next_state 6]  # state 4
packed_csr[2] = [token 1, next_state 0]  # state 5
packed_csr[3] = [token 2, next_state 0]  # state 6
packed_csr[4] = [token 3, next_state 0]  # state 6
```

查询任意 state `s`：

```python
start = indptr[s]
end = indptr[s + 1]
edges = packed_csr[start:end]
```

例如查询 `state 6`：

```text
start = indptr[6] = 3
end   = indptr[7] = 5
```

因此：

```text
packed_csr[3:5]
=
[
  [2, 0],
  [3, 0],
]
```

这就得到 prefix `[3,1]` 的两个合法下一 token：`2`、`3`。

官方代码的构造方式正是：

```python
counts = np.bincount(parents, minlength=num_states)
indptr = np.zeros(num_states + 1, dtype=np.int32)
indptr[1:] = np.cumsum(counts)
```

所以可以把 `indptr[s]` 理解为：

> 在 state `s` 之前，所有 state 一共存放了多少条边。

---

## 8. 为什么还需要 Dense Head

纯 CSR 并不一定适合所有层。

Trie 浅层通常分支非常大：

- root 可能接近 $V$ 个合法 token；
- 第一层很多 prefix 也可能有大量孩子；
- 深层随着 prefix 变细，单个节点的分支数通常快速下降。

因此 STATIC 使用混合布局：

```text
Root           -> start_mask
前 d 层         -> Dense Lookup
更深层          -> CSR/VNTK
```

默认 $d=2$ 时：

```text
start_mask[first_token]
dense_mask[first_token, second_token]
dense_states[first_token, second_token]
```

对例子：

```text
start_mask[1] = True
start_mask[3] = True

dense_mask[1,2] = True
dense_mask[3,1] = True

dense_states[1,2] = 5
dense_states[3,1] = 6
```

第二个 token 选定后，就进入 state 5 或 state 6，第三步开始使用 CSR。

Dense 表空间随 $V^d$ 增长：

```text
V = 2048

d=1 -> 2,048 个位置
d=2 -> 4,194,304 个位置
d=3 -> 8,589,934,592 个位置
```

这也是论文和官方实现通常只使用 $d=1$ 或 $d=2$ 的原因。

---

## 9. 论文 Algorithm 1 与 `sparse_transition_torch`

Algorithm 1 可以拆成四个阶段：

```text
Phase 1: LogSoftmax
Phase 2: Constraint Candidate Extraction
Phase 3: Global Beam Top-M
Phase 4: Token / Score / State Update
```

### 9.1 初始步：Root Mask

模型先输出第一个 SID token 的分数：

```python
raw_logprobs = F.log_softmax(initial_logits[:, 0, :], dim=-1)
initial_logprobs = torch.where(start_mask, raw_logprobs, -torch.inf)
top_logprobs, top_tokens = torch.topk(
    initial_logprobs, beam_size, dim=-1
)
```

状态初始化：

```python
current_transition_states = top_tokens + 1
```

### 9.2 第二个 token：Dense 路径

当 `d_dense=2` 时，第二个 token 根据首 token 直接查 Dense Row：

```python
parent_tokens = flat_states - 1
masks = dense_mask[parent_tokens]
flat_logprobs = torch.where(masks, flat_logprobs, -torch.inf)
```

候选 token 选出后，再从 `dense_states` 获取 next state：

```python
next_state_candidates = dense_states[
    parent_tokens.unsqueeze(1),
    topk_indices,
]
```

### 9.3 深层 token：Sparse 路径

```python
limit = max_branch_factors[step + 1]

candidate_logprobs, candidate_tokens, candidate_states = (
    generate_and_apply_logprobs_mask(
        flat_logprobs,
        flat_states,
        packed_csr,
        csr_indptr,
        limit,
        vocab_size,
        device,
    )
)
```

`limit` 是当前层最大分支数，它使 Kernel 输出 Shape 在编译时可确定。

---

## 10. Algorithm 2：VNTK 如何消除动态孩子数

不同 Trie 节点的孩子数量不同：

```text
node A -> 1 child
node B -> 2 children
node C -> 17 children
```

传统代码会按真实孩子数循环。GPU/TPU 更希望固定 Shape，因此论文定义每层最大分支数：

$$
B_t=
\max_{n\in\mathcal{N}_t}\operatorname{degree}(n)
$$

任何当前节点都固定读取 $B_t$ 个槽位，再通过 Mask 清理多读部分。

### 10.1 查询 Row 边界

```python
starts = csr_indptr[flat_states]
actual_lens = csr_indptr[flat_states + 1] - starts
```

假设当前两个 Beam 分别在 state 5 和 6：

```text
state 5: start=2, actual_len=1
state 6: start=3, actual_len=2
```

### 10.2 构造固定读取网格

若当前层最大分支数 `limit=2`：

```python
offsets = torch.arange(2)  # [0, 1]
gather_indices = starts[:, None] + offsets[None, :]
```

得到：

```text
state 5 -> [2, 3]
state 6 -> [3, 4]
```

这里有一个非常重要的细节：

> state 5 实际只有一条边，但它固定读取两个位置，第二个位置 3 已经属于 state 6 的第一条真实边。

因此，正确性不能依赖多读位置是 Padding，而必须依赖：

```python
valid_mask = offsets[None, :] < actual_lens[:, None]
```

结果：

```text
state 5 -> [True, False]
state 6 -> [True, True]
```

### 10.3 一次读取 token 和 next state

官方实现把两列交错存储：

```text
packed_csr[i] = [token_id, next_state_id]
```

所以：

```python
gathered_vals = packed_csr[safe_gather_indices]
candidate_token_ids = gathered_vals[..., 0]
candidate_next_states = gathered_vals[..., 1]
```

输出 Shape：

```text
[B*M, limit, 2]
```

### 10.4 直接 Gather 合法 token 的分数

论文伪代码描述的是：

```text
合法 token list
-> Scatter 成 V 维 Mask
-> 对完整 logprobs 做 Where
```

官方 PyTorch 实现进一步优化为：

```python
candidate_logprobs = flat_logprobs.gather(
    1, safe_token_ids
)
candidate_logprobs = torch.where(
    valid_mask, candidate_logprobs, -torch.inf
)
```

也就是说，它不物化完整 `[B*M, V]` 约束 Mask，而是直接得到：

```text
candidate_logprobs: [B*M, limit]
```

后续 Beam Search 只在合法候选域上运行。

---

## 11. 一次完整 VNTK + Beam Step 手算

当前两个 Beam：

| Beam | Prefix | State | 累积分数 |
| ---: | --- | ---: | ---: |
| 0 | `[1,2]` | 5 | -0.40 |
| 1 | `[3,1]` | 6 | -0.60 |

当前层最大分支数：

```text
limit = 2
```

模型 log-probability：

```text
Beam 0:
token 1 -> -0.20
token 2 -> -0.10
token 3 -> -2.00

Beam 1:
token 1 -> -0.10
token 2 -> -0.30
token 3 -> -0.40
```

CSR 固定读取结果：

```text
Beam 0 / state 5:
slot 0 -> token 1, valid=True
slot 1 -> token 2, valid=False  # 实际读到了下一 Row

Beam 1 / state 6:
slot 0 -> token 2, valid=True
slot 1 -> token 3, valid=True
```

Mask 后的候选分数：

```text
Beam 0 -> [-0.20, -inf]
Beam 1 -> [-0.30, -0.40]
```

加上父 Beam 累积分数：

```text
[1,2,1] -> -0.40 + -0.20 = -0.60
invalid -> -inf
[3,1,2] -> -0.60 + -0.30 = -0.90
[3,1,3] -> -0.60 + -0.40 = -1.00
```

若 Beam Size 为 2，保留：

```text
[1,2,1], score=-0.60
[3,1,2], score=-0.90
```

---

## 12. Beam Top-M 后如何同步更新状态

每条父 Beam 最多产生 `limit` 个候选：

```python
scores = current_scores[:, :, None] + candidate_logprobs.view(
    batch_size, beam_size, limit
)
flat_scores = scores.view(batch_size, beam_size * limit)
```

在整个候选池中选新的 Top-M：

```python
top_scores, flat_top_indices = torch.topk(
    flat_scores, beam_size, dim=-1
)
```

展平索引编码了两部分信息：

```text
flat_index = parent_beam * limit + candidate_slot
```

恢复父 Beam：

```python
top_beam_indices = flat_top_indices // limit
```

接下来必须使用同一组选中索引更新：

```text
new token
new cumulative score
new Trie next state
new token history
```

官方代码：

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

如果只更新 token 和分数，却忘记更新 `current_transition_states`，下一步就会在错误的 CSR Row 中查询约束。

在真实 vLLM 或自研 Serving 中，还必须使用相同的父 Beam 映射更新 KV Cache 的 Block Table。

---

## 13. 论文与官方 PyTorch 实现的几个差异

### 13.1 完整 Mask vs. 候选抽取

论文 Algorithm 2 描述 Scatter 成完整词表 Mask；官方 PyTorch 直接 Gather 合法 token 的 log-probability。

后者减少了完整 `[B*M, V]` 约束张量的物化。

### 13.2 独立叶子 vs. 共享 terminal 0

论文图为了展示清晰，可以为完整 SID 绘制独立叶子；官方 Builder 把最后一条边统一指向 0。

### 13.3 论文强调端到端静态图

公开 PyTorch 主循环本身只是 `torch.inference_mode()`，没有直接对整个 `sparse_transition_torch` 使用 `torch.compile`。

仓库的 PyTorch Benchmark 会在外部针对固定 `limit` 编译 `generate_and_apply_logprobs_mask`。因此公开 PyTorch 代码更适合作为算法参考实现，论文生产系统中的完整编译与融合细节没有全部体现在这段主循环中。

### 13.4 Builder 比 Decoder 更通用

Builder 可以构造更高维 Dense 表，但当前 PyTorch Dense 路径通过：

```python
parent_tokens = flat_states - 1
```

恢复首 token，因此实际主要适配 `d_dense=1/2`。

---

## 14. 正确性不变量

对每一条存活 Beam，可以维护如下不变量：

```text
1. token history 是某个合法 SID 的前缀；
2. current_transition_state 对应该前缀的 Trie state。
```

初始步由 `start_mask` 保证第一个 token 合法。

归纳步骤中，VNTK 只返回当前 state 的 CSR Row 内存在的边，因此选中的新 token 仍然构成合法前缀，next state 也与新前缀一致。

固定解码 $L$ 步后，最后一条边到达 terminal 0，所以完整 SID 属于约束集合。

这个正确性依赖以下条件：

- Builder 输入集合正确且已经排序；
- `indptr` 和 `packed_csr` 一致；
- invalid slot 不会作为有效 Beam 被选择；
- token history 和 state 始终同步 Gather；
- 请求解码期间不切换到不兼容的 Index 版本。

---

## 15. 公开实现中需要注意的边界

### 15.1 合法候选数不足 Beam Size

无效槽分数被设为 `-inf`，但如果所有父 Beam 的有效候选总数小于 Beam Size，`topk` 仍可能返回 `-inf` filler。

生产系统需要显式支持：

- active beam count；
- dead beam；
- empty result；
- fallback 或失败语义。

### 15.2 `tokens_per_beam` 与精确 Beam Search

Dense 路径会先为每个父 Beam 取局部 `tokens_per_beam` 个候选，再执行全局 Top-M。

如果要对任意分数分布保持标准 Beam Search 精确性，一个保守条件是：

```text
tokens_per_beam >= beam_size
```

否则可能出现全局最优的多个候选都来自同一个父 Beam，却被局部截断提前丢失。

### 15.3 固定最大分支的 Padding 浪费

某层只要存在一个高分支 outlier，所有 Beam 都要按该层最大分支数读取。

可进一步探索：

- 按节点分支数分桶；
- 高分支节点走 Dense 或专用 Kernel；
- 多个 branch bucket 分别编译；
- 融合 CSR Gather、候选加分和 Top-K。

---

## 16. 复杂度如何准确理解

设：

- $B$：Batch Size；
- $M$：Beam Size；
- $V$：token 词表大小；
- $K_t$：当前层最大分支数；
- $|\mathcal{C}|$：约束 SID 数。

Sparse VNTK 的主要候选抽取工作量近似为：

$$
O(B\cdot M\cdot K_t)
$$

论文所说的相对于 $|\mathcal{C}|$ 为 $O(1)$，是指：

```text
当前 state 通过两次 indptr 读取即可定位局部 Row，
不需要在全部约束 SID 中执行 O(log |C|) 的二分查找。
```

它不代表 Kernel 与 Beam 数、分支数和词表处理完全无关。

---

## 17. 接入真实 vLLM / PyTorch Serving 时还要补什么

官方 `sparse_transition_torch` 是一个清晰的算法 Harness，但真实 Serving 还要将 STATIC State 接入：

```text
Scheduler
Beam / Sequence Group
KV Cache Manager
Model Runner
Logits Processor
Sampler / Top-K
Index Version Manager
```

最关键的状态同步为：

```text
parent beam mapping
    ├── token history
    ├── cumulative score
    ├── constraint state
    └── KV block table
```

此外，不同请求可能使用不同约束索引或位于不同 SID Step，需要按：

```text
(index_id, step, branch bucket)
```

组织 Batch，才能让 Kernel 保持较稳定的静态 Shape。

---

## 18. 总结

STATIC 第 4 章真正完成了三次转换：

```text
Trie Node Object
    -> Integer State ID

Dynamic Children Loop
    -> Fixed-size CSR Gather + Valid Mask

Full-vocabulary Constraint Mask
    -> Candidate-domain LogProb Gather
```

官方 PyTorch 实现中，最值得牢牢记住的五个变量是：

```text
indptr
packed_csr
current_transition_states
max_branch_factors / limit
flat_top_indices
```

它们分别回答：

1. 当前 state 的边从哪里开始、在哪里结束；
2. 每条边允许哪个 token、到达哪个 next state；
3. 每条 Beam 当前位于哪个 Trie 节点；
4. 当前层统一读取多少候选槽；
5. 新 Beam 来自哪个父 Beam 和哪个候选槽。

一句话总结：

> STATIC 保留了 Trie 的严格前缀约束语义，但把运行时执行从指针追踪改写为连续 CSR 读取、固定 Shape Gather、Mask 和 Beam 状态同步更新。

## 参考资料

- [论文：Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators](https://arxiv.org/abs/2602.22647)
- [STATIC 官方代码仓库](https://github.com/youtube/static-constraint-decoding)
- [`csr_utils.py`](https://github.com/youtube/static-constraint-decoding/blob/main/static_decoding/csr_utils.py)
- [`decoding_pt.py`](https://github.com/youtube/static-constraint-decoding/blob/main/static_decoding/decoding_pt.py)
- [`test_csr_builder.py`](https://github.com/youtube/static-constraint-decoding/blob/main/tests/test_csr_builder.py)
- [`test_pt_decoding.py`](https://github.com/youtube/static-constraint-decoding/blob/main/tests/test_pt_decoding.py)
