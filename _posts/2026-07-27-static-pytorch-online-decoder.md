---
layout: post
title: "STATIC PyTorch 在线解码逐行解析：VNTK、Dense/CSR 切换与 GPU Beam Search"
description: "逐段拆解 decoding_pt.py：从 Beam Gather、随机模型桩和 VNTK 稀疏候选提取，到累计分数、全局 Top-M、Trie State 重排与真实 KV Cache 接入。"
date: 2026-07-27 00:30:00 +0800
category: "生成式推荐"
tags:
  - STATIC
  - PyTorch
  - VNTK
  - Beam Search
  - CSR
  - GPU
  - 生成式推荐
reading_time: "约 32 分钟"
math: true
---

> 本文只聚焦 STATIC 的在线 PyTorch 解码文件 `static_decoding/decoding_pt.py`：离线构建好的 `start_mask + dense_states + packed_csr + indptr`，如何在每个自回归解码 Step 中变成合法候选，并与 Beam Search 的分数、Token 历史和 Trie State 保持一致。

建议先阅读：

- [STATIC 离线 Trie 静态化逐行解析]({{ '/articles/static-offline-trie-index-builder/' | relative_url }})：理解 `State ID`、`dense_states`、`packed_csr` 和 `indptr` 的来源；
- [STATIC 论文第 4 章与官方 PyTorch 源码对照]({{ '/articles/static-chapter4-pytorch-walkthrough/' | relative_url }})：理解论文 Algorithm 1、Algorithm 2 与 VNTK；
- [STATIC PyTorch 实现总览]({{ '/articles/static-pytorch-implementation/' | relative_url }})：理解整个项目的工程边界。

本文对应官方源码：

- [`static_decoding/decoding_pt.py`](https://github.com/youtube/static-constraint-decoding/blob/main/static_decoding/decoding_pt.py)
- [`static_decoding/csr_utils.py`](https://github.com/youtube/static-constraint-decoding/blob/main/static_decoding/csr_utils.py)
- [`tests/test_pt_decoding.py`](https://github.com/youtube/static-constraint-decoding/blob/main/tests/test_pt_decoding.py)
- [`benchmarks/run_branch_benchmark_pt.py`](https://github.com/youtube/static-constraint-decoding/blob/main/benchmarks/run_branch_benchmark_pt.py)

---

## 1. `decoding_pt.py` 在系统中的位置

离线 Builder 已经把合法 SID 集合编译成：

```text
start_mask
    根节点允许哪些首 token

dense_mask / dense_states
    浅层前缀是否合法，以及该前缀对应哪个 Trie State

packed_csr / csr_indptr
    深层 State 的合法后继 token 和 next_state

max_branch_factors
    每一层的最大分支数 K_l
```

`decoding_pt.py` 消费这些静态数组，执行：

```text
模型 next-token logits
        │
        ├── 第一个 token：start_mask
        ├── 浅层 token：dense_mask / dense_states
        └── 深层 token：VNTK + packed_csr / indptr
                │
                ▼
       合法候选 [logprob, token, next_state]
                │
                ▼
       累积分数 + 全局 Top-M
                │
                ▼
       同步重排 Token 历史与 Trie State
```

定义本文使用的 Shape 符号：

```text
B = batch_size
M = beam_size
L = Semantic ID 长度
V = token 词表大小
K = 当前层每条 Beam 的候选宽度
Q = B * M
```

在线过程中的核心张量如下：

| 张量 | Shape | 含义 |
| --- | --- | --- |
| `top_tokens` | `[B, M]` | 当前每条 Beam 最新生成的 token |
| `token_buffer` | `[B, M, L]` | Beam 的完整 SID 历史 |
| `current_token_scores` | `[B, M]` | Beam 的累计 log-probability |
| `current_transition_states` | `[B, M]` | Beam 当前所在的 Trie State |
| `flat_logprobs` | `[Q, V]` | 展平 Beam 后的完整词表分数 |
| `candidates_logprobs` | `[Q, K]` | 每条 Beam 的候选分数 |
| `candidates_indices` | `[Q, K]` | 候选 token ID |
| `candidates_states` | `[Q, K]` | 候选 next_state |

---

## 2. `_gather_beams`：按 Batch 重排 Beam 数据

源码：

```python
def _gather_beams(
    x: torch.Tensor,
    beam_indices: torch.Tensor,
) -> torch.Tensor:
    batch_size, new_beam_size = beam_indices.shape
    view_shape = [batch_size, new_beam_size] + [1] * (x.dim() - 2)
    expand_shape = [batch_size, new_beam_size] + list(x.shape[2:])
    indices = beam_indices.view(view_shape).expand(expand_shape)
    return x.gather(1, indices)
```

接口语义是：

```text
x            [B, M_old, ...]
beam_indices [B, M_new]
输出          [B, M_new, ...]
```

不同 Batch 可以选择不同的旧 Beam。例如：

```python
beam_indices = torch.tensor([
    [2, 0],  # Batch 0 选择旧 Beam 2 和 0
    [1, 1],  # Batch 1 两条新 Beam 都继承旧 Beam 1
])
```

假设 `x` 保存 Token 历史：

```python
x.shape == [2, 3, 4]
```

`beam_indices` 原本只有 `[B, M_new]`，但 `torch.gather` 要求索引张量与输入拥有相同维度数。因此先把索引变成：

```text
[B, M_new, 1]
```

再沿历史维扩展为：

```text
[B, M_new, L]
```

例如 Batch 0 的索引：

```text
[2, 0]
```

会扩展成：

```text
[
  [2, 2, 2, 2],
  [0, 0, 0, 0],
]
```

最后：

```python
x.gather(dim=1, index=indices)
```

沿 Beam 维取出完整历史。

`expand` 通常只创建广播视图，不像 `repeat` 那样真的复制索引数据。只要 `x` 和 `beam_indices` 位于 CUDA，这个 Gather 就在 GPU 上执行。

这个 Helper 后面有两种不同用途：

```python
top_tokens = _gather_beams(
    flat_tokens,
    flat_top_indices,
)
```

这里从 `[B, M*K]` 候选池选择具体 token；而：

```python
token_buffer = _gather_beams(
    token_buffer,
    top_beam_indices,
)
```

这里根据候选所属的父 Beam 重排历史。`flat_top_indices` 与 `top_beam_indices` 不能混用，后文会详细解释。

---

## 3. `RandomModel`：模型输出桩，不是真实 Transformer

源码：

```python
class RandomModel(nn.Module):

    def __init__(self, vocab_size: int, device: torch.device):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.to(device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.size(0)
        return torch.rand(
            batch_size,
            1,
            self.vocab_size,
            device=self.device,
        )
```

它可以理解为一个模型桩：

```text
输入  [batch, seq_len]
输出  [batch, 1, V] 随机 logits
```

它没有：

- Attention；
- 模型参数；
- Token Embedding；
- 位置编码；
- 上下文建模；
- KV Cache。

甚至不会读取 `input_ids` 的具体内容，只使用：

```python
batch_size = input_ids.size(0)
```

`torch.rand` 生成的是 `[0,1)` 范围的随机 logits，不是概率。调用端继续执行：

```python
F.log_softmax(logits[:, 0, :], dim=-1)
```

才得到 log-probability。

这个模型桩的验证目标是：

> 无论模型给出什么随机分数，STATIC 是否都能保证最终输出严格属于合法 SID 集合。

它验证约束解码正确性，不验证模型质量，也不能代表真实 Transformer 的端到端性能。

此外，`RandomModel` 没有参数或 Buffer，所以：

```python
self.to(device)
```

实际上没有多少数据可移动。真正决定输出设备的是：

```python
torch.rand(..., device=self.device)
```

如果模型以后再次执行 `model.to(other_device)`，普通 Python 属性 `self.device` 不会自动更新。更稳妥的模型桩可以让输出跟随：

```python
input_ids.device
```

---

## 4. `generate_and_apply_logprobs_mask`：PyTorch VNTK

这是 `decoding_pt.py` 中最接近论文 VNTK 的函数：

```python
@torch.inference_mode()
def generate_and_apply_logprobs_mask(
    flat_logprobs,
    flat_states,
    packed_csr,
    csr_indptr,
    limit,
    vocab_size,
    device,
):
    ...
```

输入输出 Shape：

```text
flat_logprobs        [Q, V]
flat_states          [Q]
packed_csr           [E + V, 2]
csr_indptr           [S + 2]
limit                K

candidate_logprobs   [Q, K]
candidate_token_ids  [Q, K]
candidate_next_states[Q, K]
```

它不返回完整 `[Q,V]` Mask，而是直接返回压缩后的 `[Q,K]` 候选表。

### 4.1 Phase 1：并行查询 CSR Row 边界

```python
starts = csr_indptr[flat_states.long()]
actual_lens = (
    csr_indptr[flat_states.long() + 1]
    - starts
)
```

对于 State `s`：

```text
row_start = indptr[s]
row_end   = indptr[s+1]
row_len   = row_end - row_start
```

`starts[q]` 表示第 `q` 条 Beam 当前 CSR Row 的起点；`actual_lens[q]` 表示它真实有几个合法孩子。

这两次索引直接在设备上的 `csr_indptr` 完成，不需要把 State 传回 CPU 查询 Python Trie。

### 4.2 Phase 2：为所有 Beam 构造固定 `[Q,K]` 读取网格

```python
offsets = torch.arange(limit, device=device)
gather_indices = (
    starts.unsqueeze(1)
    + offsets.unsqueeze(0)
)
```

假设：

```text
starts = [3, 5, 6]
K = 3
```

那么：

```text
gather_indices =
[
  [3, 4, 5],
  [5, 6, 7],
  [6, 7, 8],
]
```

即使三个 State 的真实分支数分别为：

```text
[2, 1, 2]
```

每条 Beam 仍然统一读取三个位置。这样 Tensor Shape 固定，不需要针对每条 Beam 写不同长度的循环。

### 4.3 防止物理越界

```python
max_idx = packed_csr.size(0) - 1
safe_gather_indices = gather_indices.clamp(max=max_idx)
```

`clamp` 只负责保证索引不超出 `packed_csr` 的物理边界。

它不判断读取位置是否仍属于当前 State。某个 State 只有一个孩子时，后两个固定读取位置可能已经落入下一条 CSR Row；这些逻辑越界稍后由 `valid_mask` 清理。

离线 Builder 还会在 `packed_csr` 末尾追加 `V` 个 Padding 条目，因此靠近数组末尾的固定宽度读取仍然拥有安全存储。

### 4.4 从 HBM 读取 `[token, next_state]`

```python
gathered_vals = packed_csr[safe_gather_indices]
candidate_token_ids = gathered_vals[..., 0]
candidate_next_states = gathered_vals[..., 1]
```

Shape：

```text
safe_gather_indices  [Q, K]
gathered_vals        [Q, K, 2]
candidate_token_ids  [Q, K]
candidate_next_states[Q, K]
```

每个位置表达一条候选转移：

```text
当前 State
    --candidate_token_ids[q,k]-->
candidate_next_states[q,k]
```

这是 STATIC 用连续数组 Gather 替代 Trie 指针跳转的核心位置。

### 4.5 Phase 3：使用算术 Mask 清理多读槽位

```python
valid_mask = (
    offsets.unsqueeze(0)
    < actual_lens.unsqueeze(1)
)
```

例如：

```text
K = 3
actual_lens = [2, 1, 2]
```

得到：

```text
[
  [T, T, F],
  [T, F, F],
  [T, T, F],
]
```

这一步把：

```text
每条 Beam 动态读取 actual_len 个孩子
```

改写成：

```text
统一读取 K 个槽位，再做固定 Shape 的 Tensor 比较
```

从而避免数据相关的 Python 分支和循环。

### 4.6 从完整词表分数中只 Gather 合法候选

```python
safe_token_ids = (
    candidate_token_ids
    .long()
    .clamp(max=vocab_size - 1)
)

candidate_logprobs = flat_logprobs.gather(
    1,
    safe_token_ids,
)
```

离线 Builder 使用 `token_id=V` 作为 Padding 哨兵，但模型分数合法索引范围只有 `[0,V-1]`。因此先把哨兵 Clamp 到 `V-1`，确保 Gather 不越界；该槽位随后仍会被 `valid_mask=False` 屏蔽。

逐元素含义：

$$
\text{candidate\_logprobs}[q,k]
=
\text{flat\_logprobs}
\left[
q,\text{safe\_token\_ids}[q,k]
\right]
$$

最后：

```python
candidate_logprobs = torch.where(
    valid_mask,
    candidate_logprobs,
    torch.tensor(-float("inf"), device=device),
)
```

无效槽位变成 `-inf`，不会在正常 Top-K 中胜出。

注意只有 `candidate_logprobs` 被 Mask。无效位置的：

```text
candidate_token_ids
candidate_next_states
```

仍可能包含下一条 CSR Row 的值或 Padding 值。后续必须依赖同一位置的 `candidate_logprobs=-inf` 排除它们。

### 4.7 三个返回值必须作为一个候选结构理解

```python
return (
    candidate_logprobs,
    candidate_token_ids,
    candidate_next_states,
)
```

相同 `[q,k]` 位置共同表示：

```text
第 q 条 Beam 的第 k 个候选：

模型本步分数 = candidate_logprobs[q,k]
生成 token   = candidate_token_ids[q,k]
进入 State   = candidate_next_states[q,k]
```

例如：

```text
candidate_logprobs[0,1]    = -0.5
candidate_token_ids[0,1]   = 6
candidate_next_states[0,1] = 15
```

表示：

```text
Beam 0 选择 token 6，本步 logprob=-0.5，Trie 转移到 state 15
```

最后一个 SID token 对应的 `next_state` 为 `0`，表示固定长度 SID 已完整生成。

### 4.8 复杂度不能误读

注释中的：

```text
O(1) relative to total constraint set size
```

表示单步只访问当前 State 的 CSR Row，不在全部约束 \(|C|\) 中做遍历或二分搜索。

VNTK 的实际工作量仍与：

$$
Q\times K
$$

相关。它不是相对 Batch、Beam 和最大分支数都为常数。

---

## 5. `sparse_transition_torch`：完整约束 Beam Search

主函数：

```python
@torch.inference_mode()
def sparse_transition_torch(
    model,
    batch_size,
    beam_size,
    tokens_per_beam,
    start_token,
    max_sample_len,
    vocab_size,
    max_branch_factors,
    packed_csr,
    csr_indptr,
    start_mask,
    dense_mask,
    dense_states,
    device,
    d_dense=2,
):
    ...
```

它维护三个必须同步的 Beam 状态：

```text
token_buffer
    已生成的 SID 历史

current_token_scores
    累计 log-probability

current_transition_states
    当前 Trie State
```

### 5.1 第一个 SID token：根节点 Mask

```python
initial_input = torch.full(
    (batch_size, 1),
    start_token,
    dtype=torch.long,
    device=device,
)

initial_logits = model(initial_input)
raw_logprobs = F.log_softmax(
    initial_logits[:, 0, :],
    dim=-1,
)
```

Shape：

```text
initial_input  [B, 1]
initial_logits [B, 1, V]
raw_logprobs   [B, V]
```

应用根节点约束：

```python
initial_logprobs = torch.where(
    start_mask,
    raw_logprobs,
    torch.tensor(-float("inf"), device=device),
)
```

`start_mask[V]` 会广播到 `[B,V]`。随后：

```python
top_logprobs, top_tokens = torch.topk(
    initial_logprobs,
    beam_size,
    dim=-1,
)
```

从根节点合法 token 中建立初始 `M` 条 Beam：

```text
top_logprobs [B, M]
top_tokens   [B, M]
```

这里隐含一个前提：合法首 token 数量足够支撑 `beam_size`。否则 `torch.topk` 仍会返回部分 `-inf` 槽位。

### 5.2 初始化 SID 历史

```python
token_buffer = torch.full(
    (batch_size, beam_size, max_sample_len),
    start_token,
    dtype=top_tokens.dtype,
    device=device,
)
token_buffer[:, :, 0] = top_tokens
```

`token_buffer` 的 Shape 为：

```text
[B, M, L]
```

`start_token` 只是初始填充值。第 `0` 个位置立即被第一个真实 SID token 覆盖，所以最终返回值不包含 BOS。

### 5.3 首 token 到 Trie State 的固定映射

```python
current_transition_states = top_tokens + 1
current_token_scores = top_logprobs
```

离线 Builder 规定：

```text
首 token T → state T+1
```

所以不需要查表即可得到第一层 State。

当前累计分数为：

$$
S_1=\log P(y_1)
$$

### 5.4 自回归循环的位置含义

```python
for step in range(max_sample_len - 1):
```

第一个 token 已经生成，因此循环只生成剩余 `L-1` 个：

```text
step=0   → 生成第 2 个 token
step=1   → 生成第 3 个 token
...
step=L-2 → 生成第 L 个 token
```

每轮先展平 Beam：

```python
flat_input_ids = top_tokens.view(
    batch_size * beam_size,
    1,
)

flat_logits = model(flat_input_ids)
flat_logprobs = F.log_softmax(
    flat_logits[:, 0, :],
    dim=-1,
)

flat_states = current_transition_states.view(
    batch_size * beam_size
)
```

Shape：

```text
flat_input_ids [Q, 1]
flat_logits    [Q, 1, V]
flat_logprobs  [Q, V]
flat_states    [Q]
```

这里把每条 Beam 当作一个独立模型输入。

### 5.5 Dense 与 CSR 的切换条件

```python
if step < d_dense - 1:
```

当 `d_dense=2`：

```text
step=0 生成第二个 token → Dense
step>=1              → CSR
```

当 `d_dense=1`：

```text
第一个 token 后立即使用 CSR
```

代码虽然接受更大的 `d_dense`，但：

```python
parent_tokens = flat_states - 1
```

只适用于第一层 `state=token+1`；多维 `dense_states` 的索引也只写了两个维度。因此在线 PyTorch 实现实际只支持 `d_dense=1/2`。

---

## 6. Dense 路径：生成第二个 token

恢复第一 token：

```python
parent_tokens = (flat_states - 1).long()
```

查询合法第二 token：

```python
masks = dense_mask[parent_tokens]
```

当：

```text
dense_mask [V, V]
parent_tokens [Q]
```

输出：

```text
masks [Q, V]
```

`masks[q,v]` 表示第 `q` 条 Beam 当前首 token 后，第二 token `v` 是否构成合法二元前缀。

应用 Mask：

```python
flat_logprobs = torch.where(
    masks,
    flat_logprobs,
    torch.tensor(-float("inf"), device=device),
)
```

每条父 Beam 选择 `tokens_per_beam` 个候选：

```python
topk_logprobs, topk_indices = torch.topk(
    flat_logprobs,
    tokens_per_beam,
    dim=-1,
)
```

其中：

```text
topk_indices [Q, K]
```

直接就是第二个 token ID。

根据前两个 token 查询 State：

```python
next_state_candidates = dense_states[
    parent_tokens.unsqueeze(1),
    topk_indices.long(),
]
```

高级索引在 `[Q,1]` 和 `[Q,K]` 之间广播，等价于逐候选查询：

```python
dense_states[first_token, second_token]
```

输出是读取二元前缀后到达的 Trie State：

```text
next_state_candidates [Q, K]
```

最后统一接口：

```python
limit = tokens_per_beam

candidates_logprobs = topk_logprobs
candidates_indices = topk_indices
candidates_states = next_state_candidates
```

---

## 7. CSR 路径：调用 VNTK

```python
limit = max_branch_factors[step + 1]
```

`step+1` 对应当前要生成的 token 位置。例如：

```text
step=1 → 生成第三 token
limit=max_branch_factors[2]
```

随后：

```python
(
    candidates_logprobs,
    candidates_indices,
    candidates_states,
) = generate_and_apply_logprobs_mask(
    flat_logprobs,
    flat_states,
    packed_csr,
    csr_indptr,
    limit,
    vocab_size,
    device,
)
```

Dense 和 CSR 分支最终都输出相同协议：

```text
candidates_logprobs [Q, K]
candidates_indices  [Q, K]
candidates_states   [Q, K]
```

后面的 Beam Search 不需要知道候选来自 Dense 还是 CSR。

---

## 8. 累计分数：从 `[B,M,K]` 构造全局候选池

```python
scores = (
    current_token_scores.unsqueeze(2)
    + candidates_logprobs.view(
        batch_size,
        beam_size,
        limit,
    )
)
```

Shape：

```text
current_token_scores.unsqueeze(2) [B, M, 1]
candidates_logprobs.view(...)     [B, M, K]
scores                            [B, M, K]
```

计算：

$$
S_{\text{new}}
=
S_{\text{parent}}
+
\log P(y_t\mid y_{<t})
$$

然后：

```python
flat_scores = scores.view(
    batch_size,
    beam_size * limit,
)
```

把每个 Batch 的候选展平为：

```text
父 Beam 0 的 K 个候选
父 Beam 1 的 K 个候选
...
父 Beam M-1 的 K 个候选
```

Shape 从：

```text
[B, M, K]
```

变成：

```text
[B, M*K]
```

---

## 9. 全局 Top-M：允许父 Beam 被复制或淘汰

```python
top_scores, flat_top_indices = torch.topk(
    flat_scores,
    beam_size,
    dim=-1,
)
```

从整个 `M*K` 候选池中选择新的 `M` 条 Beam。

它不是“每个父 Beam 留一个孩子”。可能发生：

```text
父 Beam 0 的两个孩子都进入新 Top-M
父 Beam 1 没有任何孩子进入新 Top-M
```

这正是标准 Beam Search 的全局竞争。

### 9.1 扁平候选索引编码

候选池的线性索引满足：

$$
\text{flat index}
=
\text{parent beam}\times K
+
\text{child slot}
$$

因此：

```python
top_beam_indices = flat_top_indices // limit
```

可以恢复候选来自哪条父 Beam。

例如：

```text
K=3
flat_top_index=7
```

则：

```text
parent_beam = 7 // 3 = 2
child_slot  = 7 % 3 = 1
```

代码不显式计算 `child_slot`，因为 token 和 next_state 也会展平，直接用 `flat_top_indices` Gather 即可。

---

## 10. 为什么 Token/State 与历史使用不同索引

先展平候选属性：

```python
flat_tokens = candidates_indices.view(
    batch_size,
    beam_size * limit,
)

flat_next_states = candidates_states.view(
    batch_size,
    beam_size * limit,
)
```

选择具体候选 token：

```python
top_tokens = _gather_beams(
    flat_tokens,
    flat_top_indices,
)
```

选择同一候选的 next_state：

```python
current_transition_states = _gather_beams(
    flat_next_states,
    flat_top_indices,
)
```

二者都使用 `flat_top_indices`，因为它们是 `M*K` 候选池的属性。

历史缓冲区却只有 `M` 条父 Beam：

```python
token_buffer = _gather_beams(
    token_buffer,
    top_beam_indices,
)
```

因此必须使用：

```text
top_beam_indices = flat_top_indices // K
```

先找到候选所属的父 Beam，再继承历史。

最后写入新 token：

```python
token_buffer[:, :, step + 1] = top_tokens
current_token_scores = top_scores
```

这个顺序维护了一个关键不变量：

```text
token_buffer[b,m]
current_token_scores[b,m]
current_transition_states[b,m]
```

永远描述同一条 Beam 路径。

如果只重排 Token 历史，却忘记重排 Trie State，下一步会在错误的 CSR Row 查询合法后继；如果真实模型忘记同步重排 KV Cache，模型上下文也会来自错误的父 Beam。

---

## 11. 哪些部分运行在 GPU

VNTK 是算法模块名，不等于“只有 VNTK 才在 GPU 上运行”。

只要输入 Tensor 位于 CUDA，以下操作都会在 GPU 上执行：

| 模块 | 主要操作 | 是否属于 VNTK |
| --- | --- | --- |
| 模型前向 | Transformer / `RandomModel` | 否 |
| 概率归一化 | `F.log_softmax` | 否 |
| Dense 约束 | Dense Tensor 索引、`where`、`topk` | 否 |
| CSR 边界查询 | `indptr[state]` | 是 |
| 固定宽度读取 | `packed_csr[gather_indices]` | 是 |
| 稀疏候选分数 | `flat_logprobs.gather`、`where` | 是 |
| 累计分数 | Broadcast Add | 否，属于 Beam Search |
| 全局选择 | `torch.topk` | 否，属于 Beam Search |
| Beam 重排 | `torch.gather` | 否，属于 Beam Search |

例如：

```python
scores = parent_scores + candidate_scores
```

是 GPU Elementwise Kernel；

```python
torch.topk(flat_scores, beam_size)
```

是 CUDA Top-K；

```python
_gather_beams(token_buffer, top_beam_indices)
```

最终是 CUDA Gather。

而：

```python
view()
unsqueeze()
```

通常只修改 Shape/Stride 元数据，不需要单独计算 Kernel。

必须保证所有相关 Tensor 真正在 CUDA：

```python
assert flat_logprobs.is_cuda
assert flat_states.is_cuda
assert packed_csr.is_cuda
assert csr_indptr.is_cuda
assert token_buffer.is_cuda
```

只设置：

```python
device = torch.device("cuda")
```

不会自动移动已经存在的输入 Tensor。

### 11.1 `inference_mode` 不等于 Kernel 编译

```python
@torch.inference_mode()
```

只关闭梯度，不会自动把多个操作融合成一个 GPU Kernel。

普通 CUDA 执行仍可能产生：

```text
Index Kernel
Add Kernel
Clamp Kernel
Gather Kernel
Where Kernel
Top-K Kernel
```

官方 Benchmark 另外使用：

```python
compiled_fn = torch.compile(
    target_func,
    mode="reduce-overhead",
)
```

让 Inductor 针对固定 `K` 尝试减少中间张量、融合可融合操作，并通过 CUDA Graph 降低 CPU Launch 开销。

`torch.topk` 是复杂的选择算子，通常仍可能形成独立 Kernel 或融合边界。因此“逻辑上的 VNTK 单元”不应简单理解为物理上必然只有一个 CUDA Kernel。

---

## 12. 最小端到端合法性实现

下面使用官方 Builder、`RandomModel` 和 PyTorch Decoder 构造一个可以在 CPU 或 CUDA 上运行的小例子：

```python
import numpy as np
import torch

from static_decoding.csr_utils import build_static_index
from static_decoding.decoding_pt import RandomModel
from static_decoding.decoding_pt import sparse_transition_torch


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

vocab_size = 10
sids = np.array(
    [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 6, 7],
        [1, 2, 8, 9, 0],
        [2, 3, 4, 5, 6],
    ],
    dtype=np.int32,
)

# Builder 依赖字典序
L = sids.shape[1]
sids = sids[
    np.lexsort([
        sids[:, i]
        for i in range(L - 1, -1, -1)
    ])
]

(
    packed_csr,
    indptr,
    max_branch_factors,
    start_mask,
    dense_mask,
    dense_states,
) = build_static_index(
    sids,
    vocab_size=vocab_size,
    dense_lookup_layers=2,
)

packed_csr_t = torch.tensor(
    packed_csr,
    dtype=torch.long,
    device=device,
)
indptr_t = torch.tensor(
    indptr,
    dtype=torch.long,
    device=device,
)
start_mask_t = torch.tensor(
    start_mask,
    dtype=torch.bool,
    device=device,
)
dense_mask_t = torch.tensor(
    dense_mask,
    dtype=torch.bool,
    device=device,
)
dense_states_t = torch.tensor(
    dense_states,
    dtype=torch.long,
    device=device,
)

model = RandomModel(vocab_size, device)

outputs = sparse_transition_torch(
    model=model,
    batch_size=2,
    beam_size=2,
    tokens_per_beam=3,
    start_token=0,
    max_sample_len=L,
    vocab_size=vocab_size,
    max_branch_factors=max_branch_factors,
    packed_csr=packed_csr_t,
    csr_indptr=indptr_t,
    start_mask=start_mask_t,
    dense_mask=dense_mask_t,
    dense_states=dense_states_t,
    device=device,
    d_dense=2,
)

valid_set = {
    tuple(row)
    for row in sids
}

for decoded_sid in outputs.cpu().numpy().reshape(-1, L):
    assert tuple(decoded_sid) in valid_set

print(outputs)
```

由于模型 logits 是随机的，具体输出 SID 不固定；但每一条输出都必须属于 `valid_set`。

这个例子验证的是：

```text
随机模型分数
    + Dense/CSR 约束
    + Beam Search 重排
    → 最终 SID 合法
```

它不验证真实模型效果或生产延迟。

---

## 13. 正确性依赖的几个不变量

### 13.1 Trie State 必须与 Token 历史一致

对任意：

```text
token_buffer[b,m,:t]
```

`current_transition_states[b,m]` 必须是这段前缀对应的 State。

### 13.2 候选三元组必须使用同一位置

```text
candidate_logprobs[q,k]
candidate_token_ids[q,k]
candidate_next_states[q,k]
```

必须共同移动，不能分别排序。

### 13.3 历史从父 Beam 继承

具体候选位置使用：

```python
flat_top_indices
```

父历史位置使用：

```python
flat_top_indices // K
```

### 13.4 `-inf` 候选不能成为有效 Beam

如果某个 Batch 的合法候选总数小于 `beam_size`，`torch.topk` 可能被迫返回 `-inf` 位置。生产实现需要：

- 动态收缩 Beam；
- 维护 finished/dead Beam；
- 或保证约束图每一步都有足够候选。

---

## 14. 接入真实 Transformer 还缺什么

### 14.1 完整上下文或 KV Cache

当前循环只传入：

```python
flat_input_ids = top_tokens.view(B * M, 1)
```

`RandomModel` 不关心上下文，所以没有问题。真实 Transformer 必须同时维护：

```text
Prompt KV Cache
每条 Beam 的 Decode KV Cache
位置索引
Attention Metadata
```

### 14.2 KV Cache 必须跟随父 Beam 重排

Token 历史使用：

```python
top_beam_indices
```

重排，真实 KV Cache 必须使用相同映射：

```python
kv_cache = reorder_kv_cache(
    kv_cache,
    top_beam_indices,
)
```

否则表面上的 token history 正确，模型内部 Attention 上下文却来自另一条 Beam。

### 14.3 固定长度与结束状态

当前实现假设 SID 长度固定为 `L`：

```python
for step in range(L - 1):
```

`state=0` 只表示最后一个 token 后终止，不支持不同长度序列、中途 EOS 或可选后缀。

### 14.4 索引热更新

STATIC 索引是离线构建的。真实 freshness、库存、地域集合变化时，还需要：

- 后台构建新索引；
- 数据合法性校验；
- HBM 上传；
- 双 Buffer 或版本化切换；
- 请求与索引版本一致性；
- 失败回滚。

### 14.5 返回值不足

当前函数只返回：

```python
token_buffer
```

真实系统通常还需要：

```text
最终 Beam 分数
完成状态
父 Beam 映射
最终 Trie State
请求级元数据
```

---

## 15. 性能分析时要分清三层

### 第一层：VNTK Kernel

只测：

```python
generate_and_apply_logprobs_mask
```

它回答：

> CSR 候选提取随 `Q`、`K`、数据类型和设备带宽如何变化？

### 第二层：约束 Beam Search

包括：

```text
VNTK
累计分数
Top-K
Token/State Gather
历史重排
```

这时 `torch.topk` 和 Beam Gather 可能占据明显开销。

### 第三层：真实端到端 Serving

还包括：

```text
模型前向
KV Cache
Scheduler
Continuous Batching
请求队列
通信
索引版本管理
```

官方 `run_branch_benchmark_pt.py` 主要属于第一层，不能直接代表请求吞吐、首 Token 延迟或完整 Beam Search 延迟。

---

## 16. 总结

`decoding_pt.py` 可以压缩成三条状态流：

```text
模型分数流：
logits → log_softmax → 合法候选 → 累积分数 → Top-M

Token 流：
candidate token → 选择具体候选 → 继承父历史 → 追加新 token

Trie State 流：
current_state → Dense/CSR next_state → 与 token 同步选择
```

其中：

1. `_gather_beams` 解决每个 Batch 独立的 Beam 重排；
2. `RandomModel` 是 next-token logits 模型桩，用来隔离真实 Transformer；
3. `generate_and_apply_logprobs_mask` 是 PyTorch VNTK，用固定 `[Q,K]` Gather 替代 Trie 指针遍历；
4. `sparse_transition_torch` 把 Dense/CSR 候选统一成 `[logprob, token, next_state]`；
5. 全局 Top-M 后，具体候选用 `flat_top_indices`，父历史用 `flat_top_indices // K`；
6. VNTK 和 Beam Search 都可以在 GPU 上运行，但它们是不同算法模块；
7. 真实 Serving 还必须把同一父 Beam 映射传递给 KV Cache、Scheduler 和请求状态。

STATIC 的关键不只是“CSR 查询很快”，而是把约束状态、模型候选和 Beam 生命周期全部改写成适合加速器批量执行的静态 Tensor 数据流。
