---
layout: post
title: "STATIC 离线 Trie 静态化逐行解析：从 SID 矩阵到 Dense + CSR 索引"
description: "逐行拆解 build_static_index：输入 Shape、前缀去重、State ID、边收集、Dense 头部、CSR row_ptr、静态分支宽度与安全 Padding。"
date: 2026-07-26 20:00:00 +0800
category: "生成式推荐"
tags:
  - STATIC
  - Trie
  - CSR
  - 约束解码
  - NumPy
  - 生成式推荐
reading_time: "约 30 分钟"
---

> 本文只聚焦 STATIC 的离线索引构建函数 `build_static_index`：它如何在不创建 Python Trie 节点的情况下，把一个合法 Semantic ID 集合编译成加速器可以直接消费的 Dense + CSR 静态数组。

对应源码：

- [`static_decoding/csr_utils.py`](https://github.com/youtube/static-constraint-decoding/blob/main/static_decoding/csr_utils.py)
- [`tests/test_csr_builder.py`](https://github.com/youtube/static-constraint-decoding/blob/main/tests/test_csr_builder.py)
- [`static_decoding/decoding_pt.py`](https://github.com/youtube/static-constraint-decoding/blob/main/static_decoding/decoding_pt.py)
- [`static_decoding/decoding_jax.py`](https://github.com/youtube/static-constraint-decoding/blob/main/static_decoding/decoding_jax.py)

STATIC 的全称是 **Sparse Transition-Accelerated Trie Index for Constrained Decoding**。它的核心不是把普通 Trie 序列化，而是把 Trie 的语义编译成纯数组：

```text
合法 SID 集合 [N, L]
        │
        ├── start_mask：根节点允许的首 token
        │
        ├── dense_mask / dense_states：浅层直接寻址表
        │
        └── packed_csr / indptr：深层稀疏状态转移
```

在线解码时不再访问 Python Node、dict 或 children 指针，而是在 GPU/TPU 上直接做 Tensor 索引。

---

## 1. 输入 Shape 与贯穿全文的例子

函数入口为：

```python
def build_static_index(
    fresh_sids: np.ndarray,
    vocab_size: int = 2048,
    dense_lookup_layers: int = 2,
):
    ...
```

核心输入：

```text
fresh_sids.shape = [N, L]
```

- `N`：合法 SID 的数量；
- `L`：每条 SID 的固定长度；
- `fresh_sids[n, l]`：第 `n` 条 SID 在第 `l` 层的 token ID；
- `vocab_size = V`：每层 token 的取值范围为 `[0, V-1]`；
- `dense_lookup_layers = d`：前多少个 token 使用 Dense 直接寻址。

本文使用：

```python
fresh_sids = np.array([
    [1, 2, 3, 4, 5],  # A
    [1, 2, 3, 6, 7],  # B
    [1, 2, 8, 9, 0],  # C
    [2, 3, 4, 5, 6],  # D
], dtype=np.int32)

N = 4
L = 5
vocab_size = 10
dense_lookup_layers = 2
```

对应 Trie：

```text
root
├── 1
│   └── 2
│       ├── 3
│       │   ├── 4 → 5
│       │   └── 6 → 7
│       └── 8 → 9 → 0
└── 2
    └── 3 → 4 → 5 → 6
```

这里有几个必须满足、但 Builder 没有完整检查的条件：

```text
fresh_sids 是非空二维数组
所有 SID 等长
0 <= fresh_sids[n, l] < vocab_size
1 <= dense_lookup_layers < L
fresh_sids 已按行做字典序排序
```

最后一条尤其重要：后续的前缀去重和边排序都依赖字典序。

如果各层词表大小不同，`fresh_sids` 仍然是 `[N,L]`；只不过第 `l` 列满足：

```text
0 <= fresh_sids[:, l] < vocab_sizes[l]
```

相应接口更适合写成：

```python
vocab_sizes: tuple[int, ...]  # shape [L]
```

但官方实现目前使用统一的 `vocab_size`。

---

## 2. `start_mask`：编码 Trie 根节点

```python
start_mask = np.zeros(vocab_size, dtype=bool)
start_mask[np.unique(fresh_sids[:, 0])] = True
```

首先读取第一列：

```python
fresh_sids[:, 0]
# [1, 1, 1, 2]
```

去重：

```python
np.unique(fresh_sids[:, 0])
# [1, 2]
```

所以：

```python
start_mask
# [False, True, True, False, False,
#  False, False, False, False, False]
```

它表达根节点的出边：

```text
root
├── token 1
└── token 2
```

在线生成第一个 token 时：

```python
initial_logprobs = torch.where(
    start_mask,
    raw_logprobs,
    -float("inf"),
)
```

非法首 token 的分数直接变为 `-inf`。

`start_mask` 只保存合法性，不保存下一状态。第一层状态使用固定映射：

```text
token t → state t+1
```

如果各层词表不同，这里的 shape 应该是：

```text
start_mask.shape = [V0]
```

---

## 3. 比较相邻 SID：找到第一次分叉位置

```python
diff = fresh_sids[1:] != fresh_sids[:-1]
first_diff = np.full(N - 1, L, dtype=np.int8)
has_diff = diff.any(axis=1)
first_diff[has_diff] = diff[has_diff].argmax(axis=1)
```

### 3.1 `diff`

两个切片分别是：

```text
fresh_sids[1:]  → 第 1 到 N-1 行
fresh_sids[:-1] → 第 0 到 N-2 行
```

所以代码比较的是：

```text
B vs A
C vs B
D vs C
```

结果 shape：

```text
diff.shape = [N-1, L] = [3, 5]
```

具体结果：

```text
B vs A: [F, F, F, T, T]
C vs B: [F, F, T, T, T]
D vs C: [T, T, T, T, T]
```

### 3.2 `first_diff`

`first_diff` 初始全部填 `L`：

```python
first_diff = [5, 5, 5]
```

合法列下标最大为 `L-1`，所以 `L` 是“完全相同”的哨兵值。

`has_diff` 判断每一对相邻 SID 是否至少有一个位置不同：

```python
has_diff
# [True, True, True]
```

布尔数组中 `True=1`，因此：

```python
diff[has_diff].argmax(axis=1)
```

返回每一行第一个 `True` 的位置：

```python
first_diff
# [3, 2, 0]
```

含义：

| 相邻 SID | `first_diff` | 解释 |
| --- | ---: | --- |
| B vs A | 3 | 前 3 个 token 相同 |
| C vs B | 2 | 前 2 个 token 相同 |
| D vs C | 0 | 第一个 token 就不同 |

为什么不能直接对所有行 `argmax`？因为全 `False` 的数组也会返回 0：

```python
np.array([False, False]).argmax()
# 0
```

重复 SID 会因此被误判为第 0 列发生变化，所以必须先使用 `has_diff`。

实现还有一个边界：`first_diff` 使用 `int8`，只能安全表达不超过 127 的 `L`。更稳妥的类型是 `np.int32`。

---

## 4. `is_new`：标记唯一 Trie 前缀

```python
is_new = np.zeros((N, L), dtype=bool)
is_new[0, :] = True

for depth in range(L):
    is_new[1:, depth] = first_diff <= depth
```

定义：

```text
is_new[row, depth] =
第 row 条 SID 的前 depth+1 个 token
是否构成首次出现的 Trie 前缀
```

第一条 SID 的所有前缀都是首次出现：

```text
[1]
[1,2]
[1,2,3]
[1,2,3,4]
[1,2,3,4,5]
```

因此：

```python
is_new[0, :] = True
```

后续使用：

```python
first_diff <= depth
```

如果第一次变化发生在位置 `k`：

- `depth < k`：仍与上一条 SID 共享前缀；
- `depth >= k`：从该位置开始的所有更长前缀都是新的。

本例结果：

```text
A: T T T T T
B: F F F T T
C: F F T T T
D: T T T T T
```

逐行解释：

```text
A 的所有前缀都是新的
B 共享 [1]、[1,2]、[1,2,3]
C 共享 [1]、[1,2]
D 从第一个 token 开始就是新路径
```

字典序保证相同前缀的 SID 连续出现，因此只与上一行比较就足以确定该前缀是否首次出现。若未排序，同一前缀可能隔行重复，代码会错误地重复创建节点。

---

## 5. State ID：把 Trie 节点变成整数

### 5.1 第一层状态

```python
state_ids = np.zeros((N, L - 1), dtype=np.int32)
state_ids[:, 0] = fresh_sids[:, 0].astype(np.int32) + 1
```

`state_ids[n, depth]` 表示消费完前 `depth+1` 个 token 后所在的状态。

为什么 shape 是 `[N,L-1]`？

```text
完整 SID 有 L 个 token
前 L-1 个 token 后需要保存普通状态
最后一个 token 直接转移到 terminal state 0
```

状态编号约定：

```text
state 0       → terminal
token 0       → state 1
token 1       → state 2
...
token V-1     → state V
```

本例第一列：

```python
state_ids[:, 0]
# [2, 2, 2, 3]
```

初始矩阵：

```text
A: [2, 0, 0, 0]
B: [2, 0, 0, 0]
C: [2, 0, 0, 0]
D: [3, 0, 0, 0]
```

第一层预留了完整的 `1...V`。即使某个首 token 不合法，其 State ID 位置仍被占用。这会产生少量空洞，但换来了无需查表的 `token+1` 映射。

### 5.2 深层状态的起始位置

```python
depth_id_ranges = []
current_offset = vocab_size + 1
```

本例 `V=10`：

```text
state 0     → terminal
state 1~10  → 第一层 token
state 11    → 下一个可用 ID
```

`depth_id_ranges` 记录每个深度的 State ID 左闭右开区间：

```text
[start_id, end_id)
```

它主要用于后续按层统计最大分支数。

### 5.3 为深层唯一前缀分配状态

```python
for depth in range(1, L - 1):
    mask = is_new[:, depth]
    num_new = np.sum(mask)
    start_id = current_offset
    end_id = current_offset + num_new

    depth_id_ranges.append((start_id, end_id))

    state_ids[mask, depth] = np.arange(
        start_id, end_id, dtype=np.int32
    )
    state_ids[:, depth] = np.maximum.accumulate(
        state_ids[:, depth]
    )
    current_offset += num_new
```

循环范围：

```text
depth = 1 ... L-2
```

- `depth=0` 已使用 `token+1`；
- `depth=L-1` 是最后一个 token，不创建普通状态。

#### `depth=1`

```python
mask = [True, False, False, True]
num_new = 2
start_id = 11
end_id = 13
```

先只给新前缀赋值：

```text
[11, 0, 0, 12]
```

再执行：

```python
np.maximum.accumulate(...)
```

得到：

```text
[11, 11, 11, 12]
```

于是：

```text
[1,2] → state 11
[2,3] → state 12
```

共享 `[1,2]` 的 A、B、C 都获得 state 11。

#### `depth=2`

```text
mask：          [T, F, T, T]
新 ID：         [13, 0, 14, 15]
maximum 后：    [13,13,14,15]
```

对应：

```text
[1,2,3] → state 13
[1,2,8] → state 14
[2,3,4] → state 15
```

#### `depth=3`

四个长度为 4 的前缀都不同：

```text
[1,2,3,4] → state 16
[1,2,3,6] → state 17
[1,2,8,9] → state 18
[2,3,4,5] → state 19
```

最终：

```python
state_ids = [
    [2, 11, 13, 16],
    [2, 11, 13, 17],
    [2, 11, 14, 18],
    [3, 12, 15, 19],
]
```

`maximum.accumulate` 正确工作的前提仍然是排序：同一前缀的行连续排列，新 ID 写在分组首行，并向下传播到下一个新前缀。

此时：

```python
num_states = current_offset
# 20
```

合法 State ID 为 `0...19`，所以 `num_states` 是状态数和 ID 上界，不是最大 State ID。

---

## 6. 收集 `(parent, token) → child` 边

```python
all_parents, all_tokens, all_children = [], [], []

for depth in range(1, L):
    mask = is_new[:, depth]
    parent_ids = state_ids[mask, depth - 1]
    token_ids = fresh_sids[mask, depth].astype(np.int32)

    child_ids = (
        state_ids[mask, depth]
        if depth < L - 1
        else np.zeros_like(parent_ids, dtype=np.int32)
    )

    all_parents.append(parent_ids)
    all_tokens.append(token_ids)
    all_children.append(child_ids)
```

每条边表示：

```text
(当前 State, 新生成 token) → 下一 State
```

循环从 `depth=1` 开始，因为根节点和第一个 token 已由 `start_mask`、`token+1` 处理。

### 6.1 第二个 token

```text
parent_ids = [2, 3]
token_ids  = [2, 3]
child_ids  = [11,12]
```

对应：

```text
(state 2, token 2) → state 11
(state 3, token 3) → state 12
```

### 6.2 第三个 token

```text
parent_ids = [11,11,12]
token_ids  = [3, 8, 4]
child_ids  = [13,14,15]
```

对应：

```text
state 11
├── token 3 → state 13
└── token 8 → state 14

state 12
└── token 4 → state 15
```

### 6.3 第四个 token

```text
parent_ids = [13,13,14,15]
token_ids  = [4, 6, 9, 5]
child_ids  = [16,17,18,19]
```

### 6.4 最后一个 token

当 `depth == L-1`，不存在 `state_ids[:,depth]`，所有合法末边进入 terminal：

```text
(state 16, token 5) → state 0
(state 17, token 7) → state 0
(state 18, token 0) → state 0
(state 19, token 6) → state 0
```

为什么使用当前深度的 `is_new` 作为 mask？因为每个唯一当前前缀恰好对应一条唯一入边。共享前缀不应该重复写入相同转移。

---

## 7. Dense Specialization：把热前缀做成直接寻址表

```python
dense_shape = tuple(
    [vocab_size] * dense_lookup_layers
)
dense_mask = np.zeros(dense_shape, dtype=bool)
dense_states = np.zeros(dense_shape, dtype=np.int32)
```

若 `V=10,d=2`：

```text
dense_shape        = [10,10]
dense_mask.shape   = [10,10]
dense_states.shape = [10,10]
```

坐标：

```python
dense_mask[token0, token1]
```

表示长度为 2 的前缀是否合法；

```python
dense_states[token0, token1]
```

表示消费这两个 token 后到达的 State ID。

它不是创建 `d` 张表，而是创建一张 `d` 维表：

```text
d=1 → table[token0]
d=2 → table[token0, token1]
d=3 → table[token0, token1, token2]
```

### 7.1 构造多维索引

```python
indices = tuple(
    fresh_sids[:, i].astype(np.int32)
    for i in range(dense_lookup_layers)
)
```

本例 `d=2`：

```python
indices = (
    np.array([1,1,1,2]),
    np.array([2,2,2,3]),
)
```

这是 NumPy 的逐位置高级索引：

```text
第 0 项 → [1,2]
第 1 项 → [1,2]
第 2 项 → [1,2]
第 3 项 → [2,3]
```

不是两个数组的笛卡尔积。

### 7.2 查出消费 `d` 个 token 后的 State

```python
final_dense_ids = state_ids[
    :, dense_lookup_layers - 1
]
```

`state_ids` 第 0 列表示消费 1 个 token 后的状态，所以消费 `d` 个 token 对应第 `d-1` 列。

本例：

```python
final_dense_ids
# [11,11,11,12]
```

填表：

```python
dense_mask[indices] = True
dense_states[indices] = final_dense_ids
```

最终有效位置：

```text
dense_mask[1,2]   = True
dense_states[1,2] = 11

dense_mask[2,3]   = True
dense_states[2,3] = 12
```

共享前缀会重复赋相同的值，结果仍然正确。

Dense 的查询复杂度是直接寻址 O(1)，但内存复杂度为：

```text
dense_mask   ≈ 1 × V^d bytes
dense_states ≈ 4 × V^d bytes
总计         ≈ 5 × V^d bytes
```

当 `V=2048`：

```text
d=1 → 约 10 KiB
d=2 → 约 20 MiB
d=3 → 约 40 GiB
```

因此实际通常只使用 `d=1` 或 `d=2`。

如果每层词表不同，数学上的 Dense shape 应为：

```python
dense_shape = tuple(
    vocab_sizes[:dense_lookup_layers]
)
```

例如 `vocab_sizes=(100,256,64)`、`d=2` 时为 `[100,256]`。但为了让 JAX/GPU 每步 shape 固定，工程上也可以统一 padding 到 `Vmax`，再使用 layer mask 屏蔽当前层不存在的 token。

---

## 8. 将按层边列表展平

```python
parents = np.concatenate(all_parents)
tokens = np.concatenate(all_tokens)
children = np.concatenate(all_children)
```

本例：

```python
parents = [
    2,3,
    11,11,12,
    13,13,14,15,
    16,17,18,19,
]

tokens = [
    2,3,
    3,8,4,
    4,6,9,5,
    5,7,0,6,
]

children = [
    11,12,
    13,14,15,
    16,17,18,19,
    0,0,0,0,
]
```

三个数组相同位置描述一条边：

```text
(parents[i], tokens[i]) → children[i]
```

在标准 CSR 概念里：

```text
parents  → row
tokens   → column index
children → value
```

逻辑上等价于：

```python
transition[parent_state, token] = child_state
```

但只保存真实存在的边。

当前实现后面不会显式按 `parents` 排序，所以相同 parent 的边必须已经连续。这个性质来自：

1. 输入 SID 按字典序排列；
2. State ID 按深度、前缀出现顺序递增；
3. 边按深度依次追加。

`np.bincount` 只负责计数，不会替代码重新排列边。

---

## 9. `counts` 与 `indptr`：构造 CSR Row Pointer

中间数组完成使命后：

```python
del state_ids, is_new
gc.collect()
```

它们的 shape 分别是 `[N,L-1]` 和 `[N,L]`，大数据集下主动释放可以降低离线构建峰值内存。

### 9.1 统计每个 State 的出边数

```python
counts = np.bincount(
    parents,
    minlength=num_states,
)
```

本例：

```text
state 0  → 0
state 1  → 0
state 2  → 1
state 3  → 1
state 4~10 → 0
state 11 → 2
state 12 → 1
state 13 → 2
state 14 → 1
state 15 → 1
state 16~19 → 各 1
```

因此：

```text
sum(counts) = 13 = 真实边数 E
```

`minlength=num_states` 确保未使用或无出边的 State 也有对应的 0。

### 9.2 前缀和得到 `indptr`

```python
indptr = np.zeros(num_states + 1, dtype=np.int32)
indptr[1:] = np.cumsum(counts)
```

定义：

```text
state s 的边区间：
[indptr[s], indptr[s+1])
```

本例：

```python
indptr = [
    0,0,0,1,2,
    2,2,2,2,2,2,2,
    4,5,7,8,9,
    10,11,12,13,
]
```

查询 state 11：

```python
start = indptr[11]  # 2
end = indptr[12]    # 4

tokens[2:4]
# [3,8]

children[2:4]
# [13,14]
```

所以：

```text
state 11
├── token 3 → state 13
└── token 8 → state 14
```

无出边 state 满足：

```text
indptr[s] == indptr[s+1]
```

例如 terminal state 0 的区间为 `[0,0)`。

---

## 10. `layer_max_branches`：为静态 Shape 提供 K

Trie 每个节点的子节点数量不同，但 JAX/XLA、GPU 编译更喜欢固定 shape。STATIC 为每个深度计算：

```text
K_l = 该层所有 State 中的最大出边数
```

在线时每个 State 都读取 `K_l` 个槽位，短行用 mask 屏蔽多读出的部分。

### 10.1 根节点

```python
layer_max_branches = [
    np.sum(start_mask)
]
```

本例合法首 token 为 `{1,2}`：

```text
layer_max_branches = [2]
```

### 10.2 第一层 State

```python
l0_counts = counts[1:vocab_size + 1]
layer_max_branches.append(
    int(l0_counts.max())
    if len(l0_counts) > 0
    else 0
)
```

State ID `1...V` 恰好对应第一层 token。它们的最大出边数决定第二个 token 的候选宽度。

本例 state 2、3 都只有一条边：

```text
layer_max_branches = [2,1]
```

### 10.3 更深层 State

```python
for start_id, end_id in depth_id_ranges:
    if start_id < len(counts):
        layer_counts = counts[start_id:end_id]
        layer_max_branches.append(
            int(layer_counts.max())
            if len(layer_counts) > 0
            else 0
        )
    else:
        layer_max_branches.append(0)
```

本例：

| State 范围 | 前缀长度 | `layer_counts` | 最大值 |
| --- | ---: | --- | ---: |
| `[11,13)` | 2 | `[2,1]` | 2 |
| `[13,16)` | 3 | `[2,1,1]` | 2 |
| `[16,20)` | 4 | `[1,1,1,1]` | 1 |

最终：

```python
layer_max_branches
# [2,1,2,2,1]
```

对齐关系：

| 索引 | 要生成的位置 | 最大候选数 |
| ---: | --- | ---: |
| 0 | 第 1 个 token | 2 |
| 1 | 第 2 个 token | 1 |
| 2 | 第 3 个 token | 2 |
| 3 | 第 4 个 token | 2 |
| 4 | 第 5 个 token | 1 |

防御性补齐：

```python
while len(layer_max_branches) < L:
    layer_max_branches.append(1)
```

填 1 而不是 0，是为了避免产生零宽候选 Tensor。不过在当前正常流程中：

```text
2 个初始层级 + (L-2) 个 depth range = L
```

所以这个循环通常不会执行。

---

## 11. 安全 Padding 与最终打包

### 11.1 在真实边后追加 Padding

```python
raw_indices = np.concatenate([
    tokens,
    np.full(
        vocab_size,
        vocab_size,
        dtype=np.int32,
    ),
])

raw_data = np.concatenate([
    children,
    np.zeros(vocab_size, dtype=np.int32),
])

indptr = np.append(
    indptr,
    indptr[-1] + vocab_size,
)
```

合法 token 是 `0...V-1`，因此使用：

```text
padding token = V
padding next_state = 0
```

每个 State 在线统一读取 `K_l` 个位置。若最后一个真实 CSR Row 比 `K_l` 短，固定宽度 gather 可能越过真实边末尾；追加 `V` 行 padding 提供安全区域，因为合法节点最多只有 `V` 个不同 token 出边。

本例：

```text
真实边数 E = 13
padding 数  = 10
最终行数    = 23
```

扩展后的最后两个指针：

```text
indptr[-2] = 13
indptr[-1] = 23
```

这相当于增加一个不会由合法路径到达的虚拟 padding state。

当前 PyTorch kernel 还会 clamp gather index，JAX 使用 `mode="fill"`，所以 padding 与 kernel 越界保护存在一定重复，但它让底层数组自身也具备完整安全尾区。

### 11.2 打包 `[token, next_state]`

```python
packed_csr = np.ascontiguousarray(
    np.vstack([raw_indices, raw_data]).T
)
```

`vstack` 后：

```text
shape = [2, E+V]

[
  [token0, token1, ...],
  [state0, state1, ...],
]
```

转置后：

```text
shape = [E+V, 2]

[
  [token0, next_state0],
  [token1, next_state1],
  ...
]
```

于是：

```python
packed_csr[i, 0]  # token
packed_csr[i, 1]  # next state
```

`.T` 通常产生非 C-contiguous 视图，`np.ascontiguousarray` 保证物理内存按行连续：

```text
token0, state0, token1, state1, ...
```

一次 gather 就可以同时读取 token 和下一状态。

`parents` 不需要返回，因为 parent 已编码进 `indptr`：

```python
edges = packed_csr[
    indptr[state]:indptr[state + 1]
]
```

---

## 12. 六个返回值的完整契约

```python
return (
    packed_csr,
    indptr,
    tuple(layer_max_branches),
    start_mask,
    dense_mask,
    dense_states,
)
```

本例 `N=4,L=5,V=10,d=2,E=13,num_states=20`：

| 返回值 | Shape | 含义 |
| --- | --- | --- |
| `packed_csr` | `[23,2]` | 每行 `[token,next_state]` |
| `indptr` | `[22]` | 每个 State 的 CSR 区间指针，加一个 padding row |
| `layer_max_branches` | 长度 5 的 tuple | 每个 token 位置的最大分支宽度 |
| `start_mask` | `[10]` | 合法首 token |
| `dense_mask` | `[10,10]` | 合法二元前缀 |
| `dense_states` | `[10,10]` | 二元前缀到 State 的映射 |

具体元数据：

```python
layer_max_branches
# (2,1,2,2,1)
```

转换为 tuple 的一个重要原因是 JAX JIT 的静态参数需要可哈希对象，list 不可哈希。

最终合法路径可以这样手工遍历：

```python
sid = [1,2,3,6,7]

# Dense 入口
state = dense_states[1,2]
# 11

# CSR：token 3
packed_csr[indptr[11]:indptr[12]]
# 包含 [3,13]、[8,14]
state = 13

# CSR：token 6
packed_csr[indptr[13]:indptr[14]]
# 包含 [4,16]、[6,17]
state = 17

# CSR：token 7
packed_csr[indptr[17]:indptr[18]]
# 包含 [7,0]
state = 0
```

到达 `state 0`，说明完整 SID 合法终止。

---

## 13. 离线与在线的交接

默认 `d=2` 时，在线流程是：

```text
第 1 个 token
  start_mask
      ↓ token+1
第一层 State
      ↓ dense_mask[token0] 约束第 2 个 token
  dense_states[token0,token1]
      ↓
长度为 2 的前缀 State
      ↓ packed_csr[indptr[state]:indptr[state+1]]
第 3...L 个 token
      ↓
terminal state 0
```

CSR kernel 不需要建立 `[B*M,V]` 的完整 Trie mask，而是直接取当前 State 的 `K_l` 个合法 token：

```text
flat_states:      [B*M]
gather_indices:   [B*M,K_l]
candidate_tokens: [B*M,K_l]
next_states:      [B*M,K_l]
```

其开销与当前层最大分支数 `K_l` 有关，而与约束集合总 SID 数 `N` 无关。这才是 STATIC 静态化真正想换取的在线特性。

---

## 14. 代码实现边界

### 14.1 Builder 支持的 `d` 比 Decoder 更宽

Builder 可以构造 `d=3` 甚至更深的 Dense Tensor，但官方 PyTorch/JAX Decoder 的 Dense 路径实际只可靠支持 `d=1/2`。

Decoder 使用：

```python
parent_tokens = flat_states - 1
```

只有第一层 State 满足 `state=token+1`。进入更深层后，State ID 已经是全局前缀编号，不能再恢复成单个 token。

### 14.2 CSR 实际包含 Dense 头部边

文档把 CSR 描述为 sparse tail，但当前 Edge Collection 从 `depth=1` 开始收集全部后续边。因此默认 `d=2` 时，第二个 token 的边既存在于 Dense 表，也存在于 CSR。

这是内存上的重复，不影响正确性。

### 14.3 这还不是完整的磁盘索引格式

`build_static_index` 只返回 NumPy 数组，没有提供：

- `np.savez` 或 mmap 封装；
- schema/version；
- vocab 与 SID 数据校验；
- checksum；
- 原子切换和线上热更新。

所以“离线”表示构建发生在解码前，并不代表仓库已经实现了生产级磁盘持久化。

### 14.4 固定长度假设

最后 token 统一转移到 state 0，因此当前结构围绕固定长度 SID 设计。若支持不同长度，需要显式 EOS 边或终止标记，否则某条短 SID 的 terminal 与长 SID 的中间前缀无法完整表达。

### 14.5 输入校验不足

更稳健的 Builder 应主动检查：

```text
N > 0
L >= 2
1 <= d < L
token 不为负
token 小于对应层词表大小
输入已排序
State 和边数量未超过 int32
```

尤其是负 token：NumPy 会把它解释为从末尾开始的负索引，不一定立刻报错，可能静默污染 `start_mask` 或 Dense 表。

---

## 15. 一句话总结

`build_static_index` 可以看成一个小型 Trie 编译器：

```text
排序 SID
→ 相邻比较识别唯一前缀
→ 前缀映射成整数 State
→ 唯一前缀映射成状态转移边
→ 热前缀写入 Dense 直接寻址表
→ 其余状态转移组织成 CSR
→ 用每层最大分支数固定在线 Tensor Shape
→ 用安全 Padding 支持无分支循环的批量 gather
```

它保留了 Trie 的约束语义，却移除了在线阶段的 Python 对象、指针追踪和 CPU/GPU 往返，为后续向量化约束 Beam Search 提供了静态、连续、可编译的内存布局。
