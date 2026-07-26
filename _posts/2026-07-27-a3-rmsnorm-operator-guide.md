---
layout: post
title: "怎样写一个 A3 算子：以 RMSNorm 为例"
description: "从 PyTorch 参考实现出发，用 PyPTO 编写、编译和验证 Ascend A3 RMSNorm，并沿生成的 PTO IR 解释 Tile 切分、FP32 归约、广播、A3 指令映射与性能调优。"
date: 2026-07-27 10:00:00 +0800
category: "NPU · MegaKernel"
series: "MegaKernel"
tags:
  - NPU
  - Ascend A3
  - RMSNorm
  - PyPTO
  - PTO ISA
  - Kernel
reading_time: "约 18 分钟"
math: true
---

> 本文基于 `hw-native-sys` 的 [`pypto-lib`](https://github.com/hw-native-sys/pypto-lib/tree/45be52c113abbb2f255e1fcdec2e5fd154cfb196)、[`pypto`](https://github.com/hw-native-sys/pypto/tree/d64380cba7b10785f9bc4f64e324b49d964c053b)、[`PTOAS`](https://github.com/hw-native-sys/PTOAS/tree/26212b29e4a1286ac3facc03f87ce47c9110be90) 和 [`pto-isa`](https://github.com/hw-native-sys/pto-isa/tree/439faf48d7ab5c36a7bf72bc99d29c213a315550) 代码。检查日期为 **2026-07-27**。本文做了代码与生成物分析，没有在 A3 实机上执行性能测试。

## 1. 最短答案

如果目标是给 PTO 软件栈写一个 A3 RMSNorm，推荐路径是：

```text
PyTorch golden
    ↓
pypto-lib 中编写 @pl.jit
    ↓
先跑 -p a2a3sim
    ↓
再跑 -p a2a3 -d 0
    ↓
检查 passes_dump / ptoas / report
    ↓
用 PYPTO_BENCH + L2 swimlane 做性能迭代
```

不要一开始就手写 PTO C++，也不要直接复制 PTOAS 仓里现有的 RMSNorm PTODSL SIMT 示例：后者当前明确使用 `@pto.jit(target="a5")`，是 A5 显式 SIMT 路径，不是 A3 的现成模板。

另一个命名陷阱是：**PyPTO 对 A3 使用的平台参数是 `a2a3`，对应内部 `Ascend910B` backend。** 因此实机命令写 `-p a2a3`，不是 `-p a3`。底层 `pto-isa` 单测脚本才使用 `-v a3`。

## 2. 先固定 RMSNorm 的数学和合同

对输入 $x\in\mathbb{R}^{M\times H}$、权重 $\gamma\in\mathbb{R}^{H}$，RMSNorm 是：

$$
y_{i,j} =
x_{i,j}
\cdot
\frac{1}{\sqrt{
\frac{1}{H}\sum_{k=0}^{H-1}x_{i,k}^{2}
+\epsilon
}}
\cdot
\gamma_j
$$

写 Kernel 前先固定四件事：

| 项目 | 示例选择 | 原因 |
| --- | --- | --- |
| 归约轴 | 最后一维 `HIDDEN` | 每一行独立归一化 |
| 输入/输出 | BF16 或 FP32 | 取决于模型合同 |
| 累加类型 | FP32 | 避免平方和在低精度下快速累积误差 |
| `eps` | 与模型配置一致 | `1e-5` 与 `1e-6` 不能随意互换 |

最简单的 PyTorch golden：

```python
def golden_rms_norm(x, gamma, eps=1e-6):
    x_fp32 = x.float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + eps)
    return (x_fp32 * inv_rms * gamma.float()).to(x.dtype)
```

golden 不是附属代码。它定义数值语义、dtype 转换、舍入位置和容差，是后续判断 Kernel 是否正确的唯一基准。

## 3. 推荐的 PyPTO 实现

仓库已经提供了可直接运行的教学实现：

- [`examples/intermediate/rms_norm.py`](https://github.com/hw-native-sys/pypto-lib/blob/45be52c113abbb2f255e1fcdec2e5fd154cfb196/examples/intermediate/rms_norm.py)
- [`examples/kernels/07_normalization.py`](https://github.com/hw-native-sys/pypto/blob/d64380cba7b10785f9bc4f64e324b49d964c053b/examples/kernels/07_normalization.py)

下面保留第一份实现的核心结构：

```python
import pypto.language as pl

ROWS = 512
HIDDEN = 512
ROW_TILE = 64
HIDDEN_TILE = 64
EPS = 1e-6


@pl.jit
def rms_norm(
    x: pl.Tensor[[ROWS, HIDDEN], pl.FP32],
    gamma: pl.Tensor[[1, HIDDEN], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, HIDDEN], pl.FP32]],
):
    # 不同行之间没有数据依赖，可以分发到不同 core-group。
    for r in pl.parallel(0, ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_norm_rows"):
            # 每行一个平方和；用 [1, ROW_TILE] 保持 RowMajor。
            sq_sum = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)

            # 第一遍：分块读取 hidden，累计 sum(x^2)。
            for hb in pl.range(HIDDEN // HIDDEN_TILE):
                h0 = hb * HIDDEN_TILE
                x_tile = x[r : r + ROW_TILE, h0 : h0 + HIDDEN_TILE]
                sq = pl.mul(x_tile, x_tile)
                row_sq = pl.reshape(pl.row_sum(sq), [1, ROW_TILE])
                sq_sum = pl.add(sq_sum, row_sq)

            inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / HIDDEN), EPS))
            inv_rms = pl.reshape(inv_rms, [ROW_TILE, 1])

            # 第二遍：重新读取 x，乘每行 inv_rms 和每列 gamma。
            for hb in pl.range(HIDDEN // HIDDEN_TILE):
                h0 = hb * HIDDEN_TILE
                x_tile = x[r : r + ROW_TILE, h0 : h0 + HIDDEN_TILE]
                gamma_tile = gamma[:, h0 : h0 + HIDDEN_TILE]
                normed = pl.row_expand_mul(x_tile, inv_rms)
                normed = pl.col_expand_mul(normed, gamma_tile)
                y[r : r + ROW_TILE, h0 : h0 + HIDDEN_TILE] = normed

    return y
```

这段代码的重点不是 API 数量，而是三个层次：

```text
pl.parallel                 跨 core-group 的行块并行
  └─ pl.at(CORE_GROUP)      一个 InCore 区域
       └─ pl.range          单个区域内带累加依赖的顺序循环
```

### 为什么是两遍 hidden

第一遍结束前，`inv_rms` 还不知道；第二遍才能计算输出。若把所有 `x` 常驻 UB，可以避免第二次 GM 读取，但通常会显著增加 UB 占用，限制 `ROW_TILE × HIDDEN`。对于较大的 hidden size，两遍流式读取是更稳妥的起点。

### 为什么外层用 `pl.parallel`

不同行的 RMSNorm 完全独立。使用 `pl.range` 会人为制造串行依赖，使任务集中在少数执行通道。`pl.parallel(0, ROWS, ROW_TILE)` 表明每个行块可独立调度。

### 为什么归约循环先用 `pl.range`

`sq_sum` 是循环携带的累加状态，语义上必须按 hidden chunk 合并。先用 `pl.range` 保证实现清晰正确；确认生成 IR 和数值后，可参考模型 Kernel 改成：

```python
for kb in pl.pipeline(HIDDEN // HIDDEN_TILE, stage=2):
    ...
```

真实模型实现已经使用 `pl.pipeline(..., stage=2/4)` 隐藏 load 与 vector 计算延迟，例如：

- [`Qwen3 JIT RMSNorm`](https://github.com/hw-native-sys/pypto/blob/d64380cba7b10785f9bc4f64e324b49d964c053b/examples/models/qwen3_jit/kernels/rmsnorm.py)
- [`DeepSeek V4 RMSNorm`](https://github.com/hw-native-sys/pypto-lib/blob/45be52c113abbb2f255e1fcdec2e5fd154cfb196/models/deepseek/v4-flash/rmsnorm.py)

## 4. 面向真实模型时怎样改成 BF16

教学示例使用 FP32 输入输出。真实 LLM 常见合同是：

```text
x       BF16 [T, H]
gamma   BF16/FP32 [H] 或 [1, H]
y       BF16 [T, H]
acc     FP32
```

核心改动是“加载后升到 FP32，写回前降到 BF16”：

```python
x_chunk = pl.cast(
    x[t0 : t0 + T_TILE, h0 : h0 + H_TILE],
    target_type=pl.FP32,
)

sq_sum = pl.add(
    sq_sum,
    pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T_TILE]),
)

gamma_chunk = pl.cast(
    pl.reshape(gamma[h0 : h0 + H_TILE], [1, H_TILE]),
    target_type=pl.FP32,
)

out_fp32 = pl.col_expand_mul(
    pl.row_expand_mul(x_chunk, inv_rms),
    gamma_chunk,
)

y[t0 : t0 + T_TILE, h0 : h0 + H_TILE] = pl.cast(
    out_fp32,
    target_type=pl.BF16,
    mode="rint",
)
```

需要保证 Kernel 和 golden 的转换位置一致。若 golden 在最后才转 BF16，而 Kernel 中途已经多次 BF16 round-trip，即使公式相同也会出现系统性误差。

## 5. A3 最终会看到哪些 PTO 指令

PTOAS 仓库保存了一份由 PyPTO 生成的 A3 Qwen3 RMSNorm：

- [`Qwen3DecodeA3/rmsnorm.pto`](https://github.com/hw-native-sys/PTOAS/blob/26212b29e4a1286ac3facc03f87ce47c9110be90/test/samples/Qwen3DecodeA3/rmsnorm.pto)

文件开头已经明确：

```text
module attributes {pto.target_arch = "a2a3"} {
  func.func @rmsnorm(...)
    attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
```

也就是说 RMSNorm 被降低为 A3 的 vector Kernel。主要映射关系是：

| PyPTO | PTO IR / ISA | RMSNorm 中的作用 |
| --- | --- | --- |
| Tensor slice | `tload` / `tstore` | GM 与 vector buffer 间搬运 |
| `pl.cast` | `tcvt` | BF16 ↔ FP32 |
| `pl.mul(x, x)` | `tmul` | 平方 |
| `pl.row_sum` | `trowsum` | hidden 维归约 |
| 标量乘/加 | `tmuls` / `tadds` | 除以 H、加 eps |
| `pl.rsqrt` | `trsqrt` | 倒数平方根 |
| `pl.sqrt` + `pl.recip` | `tsqrt` + `trecip` | 另一种倒数平方根表达 |
| `pl.row_expand_mul` | `trowexpandmul` | 每行乘自己的 `inv_rms` |
| `pl.col_expand_mul` | `tcolexpandmul` | 每列乘 gamma |

实际选 `trsqrt` 还是 `tsqrt + trecip`，取决于源程序和编译 Pass；不要只凭 Python API 名字推断最终指令，应该检查本次构建生成的 `.pto`。

### 一个 A3 特有的 UB 注意点

在 A2/A3 上，`TROWEXPANDMUL` 的每行标量模式需要把 ColMajor 的 `[ROW_TILE, 1]` 广播为 32 字节块。底层实现会使用临时广播缓冲；不显式传 tmp 的接口使用内部最多约 8 KB 缓冲。

因此 RMSNorm 看起来只有 `x_tile + sq_sum + inv_rms`，实际 UB 预算还要包含广播临时空间。调大 `ROW_TILE` 或 `HIDDEN_TILE` 时，必须以编译后的内存报告为准。

## 6. 加上 TensorSpec、golden 和运行入口

在 `pypto-lib` 中，建议让同一个文件同时具备编译、执行和数值验证能力：

```python
def build_tensor_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("x", [ROWS, HIDDEN], torch.float32,
                   init_value=torch.randn),
        TensorSpec("gamma", [1, HIDDEN], torch.float32,
                   init_value=torch.randn),
        TensorSpec("y", [ROWS, HIDDEN], torch.float32,
                   is_output=True),
    ]


def golden_fn(tensors):
    x = tensors["x"]
    gamma = tensors["gamma"]
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    tensors["y"][:] = x * inv * gamma
```

入口函数：

```python
from golden import run_jit

result = run_jit(
    fn=rms_norm,
    specs=build_tensor_specs(),
    golden_fn=golden_fn,
    runtime_cfg=dict(
        platform=args.platform,
        device_id=args.device,
        enable_l2_swimlane=args.enable_l2_swimlane,
    ),
    rtol=1e-2,
    atol=1e-2,
)
```

`y` 必须写成 `pl.Out[...]`。如果 orchestration 入口只写普通 `pl.Tensor`，运行时会把它当输入，不执行正确的 device-to-host copy-back，最终可能看到全零输出。

## 7. 从模拟器到 A3 实机的命令

先安装固定版本的 pypto、simpler 和 PTOAS，并准备 CANN 环境。进入 `pypto-lib`：

```bash
# 先验证编译链、生成物和基本数值
python examples/intermediate/rms_norm.py -p a2a3sim

# 再运行 A3 实机，device 0
python examples/intermediate/rms_norm.py -p a2a3 -d 0
```

`a2a3` 和 `a2a3sim` 都映射到 `BackendType.Ascend910B`，代码生成时写入：

```text
pto.target_arch = "a2a3"
```

两者的区别是 simpler 选择模拟平台还是真实设备平台。

### 构建后重点看什么

```text
build_output/<ProgramName>_<timestamp>/
├── passes_dump/       每个 PyPTO Pass 后的 IR
├── ptoas/             生成的 .pto 与 PTOAS 中间产物
├── kernels/aiv/       RMSNorm vector kernel C++ wrapper
├── orchestration/     调度 Kernel 的 AICPU C++
├── kernel_config.py   Kernel 类型和运行时 ID
├── report/            内存分配、调度和编译报告
└── dfx_outputs/       swimlane、PMU、依赖图等
```

建议按这个顺序排错：

1. **PyPTO verifier 报错**：看 `passes_dump/` 最后一个正常 Pass 和第一个异常 Pass。
2. **PTOAS 报错**：定位 `ptoas/*.pto` 中具体 op、shape 和 layout。
3. **模拟器通过、实机失败**：检查设备日志、CANN 与固定依赖版本。
4. **结果不一致**：保存输入和 golden，固定同一批数据重放。
5. **偶发错误或卡死**：打开参数 dump 和依赖图，排查错误依赖与同步。

## 8. A3 上怎样调 RMSNorm 性能

### 8.1 先测端到端时间

```bash
PYPTO_BENCH=1 \
PYPTO_BENCH_ROUNDS=20 \
PYPTO_BENCH_WARMUP=5 \
python examples/intermediate/rms_norm.py -p a2a3 -d 0
```

先记录 `effective_us`，再改 tile。没有基线的“优化”无法判断是否真的生效。

### 8.2 先调两个 tile

RMSNorm 的第一组旋钮通常只有：

```text
ROW_TILE       每个 Kernel 处理多少行
HIDDEN_TILE    每次向量归约处理多少列
```

调参时同时观察：

- `ROW_TILE` 太小：Kernel 数量多，AICPU 调度占比高；
- `ROW_TILE` 太大：vector UB 压力大，并行块数下降；
- `HIDDEN_TILE` 太小：load 和归约次数过多；
- `HIDDEN_TILE` 太大：工作 Tile、临时 Tile 和广播缓冲可能超 UB；
- 两者都必须满足 shape、对齐和当前模板的整除条件。

A3/910C 的 vector buffer 报告上限为 192 KB。查看：

```text
report/memory_after_AllocateMemoryAddr.txt
```

不要只按 `ROW_TILE × HIDDEN_TILE × sizeof(dtype)` 估算；`row_sum` 临时 Tile、FP32 转换结果、流水级和广播 tmp 都会进入实际占用。

### 8.3 看 L2 swimlane

```bash
python examples/intermediate/rms_norm.py \
  -p a2a3 -d 0 --enable-l2-swimlane
```

打开：

```text
dfx_outputs/merged_swimlane_<timestamp>.json
```

重点看：

- AICPU 连续忙、AIV 大量空洞：Kernel 太碎；
- 只有一个 AIV 长尾：行块过大或并行度不足；
- RMSNorm 前后有独立的 cast、residual、projection Kernel：考虑跨 `pl.at` 融合；
- 单独 RMSNorm 明显短于约 50 μs：继续优化单 Kernel 的收益可能不如减少一次 AICPU hand-off。

在完整模型中，RMSNorm 常常应该和前后的 vector/cube 工作合到更大的 InCore 区域，而不是永远作为独立 Kernel 追求局部峰值。

### 8.4 从 `pl.range` 升级到 `pl.pipeline`

正确性稳定后，用 `pl.pipeline(stage=2)` 起步。增加 stage 会提高并行在途数据，也会增加 buffer 占用。每次变化都要同时检查：

```text
数值误差
memory_after_AllocateMemoryAddr.txt
effective_us
PMU / kernel insight
```

如果 stage 增加后 UB 溢出或编译器拆分了 Tile，性能可能反而下降。

## 9. 什么时候需要下沉到 PTO ISA C++

只有出现以下情况，才值得从 PyPTO 下沉：

- PyPTO/PTOAS 生成的指令序列无法表达目标流水；
- 必须手工安排 UB 地址、event 和 MTE/V 同步；
- 需要稳定复用一段高度调优的 A3 指令组合；
- 通过生成 `.pto` 和 C++ wrapper 已经确认瓶颈在 InCore，而不是调度、数据布局或模型融合。

手写版本的基本指令链仍然是：

```text
TLOAD
→ TCVT(BF16→FP32)
→ TMUL(x, x)
→ TROWSUM
→ TMULS(1/H)
→ TADDS(eps)
→ TRSQRT 或 TSQRT + TRECIP
→ TROWEXPANDMUL
→ TCOLEXPANDMUL
→ TCVT(FP32→BF16)
→ TSTORE
```

但一旦进入手写 C++，就要自己负责 Tile 类型、RowMajor/ColMajor、UB 地址、临时缓冲、同步事件和尾块。对于第一个 A3 算子，这通常不是成本最低的入口。

## 10. 这个模板暂时不包含什么

当前教学模板有明确边界：

- `ROWS`、`HIDDEN` 是静态 shape；
- 假设 `ROWS % ROW_TILE == 0`；
- 假设 `HIDDEN % HIDDEN_TILE == 0`；
- 没有处理尾块 `valid_shape`；
- 没有动态 token 数；
- 没有把 RMSNorm 与 residual、QKV projection 融合；
- 没有量化或分布式切分；
- 模拟器正确不代表 A3 性能已经合理。

动态 token 数可以参考 DeepSeek V4 的 `pl.dynamic`、`bind_dynamic` 和 `pl.spmd` 用法；尾块则需要显式处理 valid shape 或 padding，不能简单去掉整除断言。

## 11. 提交前检查清单

- [ ] 数学公式、归约轴和 `eps` 与模型一致；
- [ ] BF16 输入先升到 FP32 再平方和归约；
- [ ] 输出入口声明为 `pl.Out`；
- [ ] 行块用 `pl.parallel` 或 `pl.spmd` 表达独立并行；
- [ ] hidden 累加先保证正确，再尝试 `pl.pipeline`；
- [ ] `ROW_TILE`、`HIDDEN_TILE` 满足整除、布局和 UB 限制；
- [ ] `-p a2a3sim` 通过；
- [ ] `-p a2a3 -d 0` 与 PyTorch golden 通过；
- [ ] 检查生成 `.pto` 确认 `target_arch="a2a3"`；
- [ ] 检查 `memory_after_AllocateMemoryAddr.txt`；
- [ ] 用 `PYPTO_BENCH` 记录优化前后数据；
- [ ] 用 L2 swimlane 判断该优化单 Kernel，还是与邻接 Kernel 融合。

---

对第一个 A3 算子而言，最重要的不是立刻写出手工指令，而是建立一条可重复的闭环：**golden 定义语义，PyPTO 表达并行和 Tile，PTOAS 暴露真实指令，simpler 执行并记录调度，最后用精度和性能数据共同决定是否继续下沉。** RMSNorm 足够小，能看清整条链；同时又包含归约、广播、精度提升和两遍访存，是学习 A3 vector Kernel 的合适起点。
