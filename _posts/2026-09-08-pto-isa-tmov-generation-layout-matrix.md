---
layout: post
title: "PTO 全栈课程 29：同名 Tile，不同代际——TMOV 的 ND/NZ/ZN 合法矩阵"
description: "从 A2/A3 与 A5 的真实 TMOV 实现、Tile alias 和 NPU 测试出发，建立 source/destination location、layout 与物理重排的代际合法矩阵。"
date: 2026-09-08 09:15:00 +0800
category: "PTO · 全栈精讲"
series: "PTO 全栈课程"
tags: [PTO, pto-isa, TMOV, ND, NZ, ZN, Ascend]
reading_time: 25
mermaid: true
---

## 本篇在 PTO 课程路线中的位置

课程 07 已经解释 A2/A3 GEMM 中的 `Mat→Left/Right`：逻辑值不变，但数据从 L1 进入 L0A/L0B，并按 Cube operand role 重排。那一章回答“为什么普通 Mat 不能直接喂给 `TMATMUL`”。

本篇不是重复它，而是处理随后出现的**代际漂移**：到了 A5，`TMOV` 不仅扩展了 TileType 组合，还为 Vector local memory 中的 `ND→NZ`、`ND→ZN` 增加了显式重排；更隐蔽的是，同一个 `TileLeft` C++ 别名在 A2/A3 和 A5 上代表不同的物理布局。课程位置是：

```text
Tile layout 基础 → A2/A3 Mat→Left/Right → 本篇代际合法矩阵 → layout-changing ops 边界
```

源码基线是 pto-isa [`5a4f74cb`](https://github.com/hw-native-sys/pto-isa/commit/5a4f74cbf627d4aac2e0ce10d5e0d8b118343265)。本文以已合入实现和直接测试为事实来源；硬件周期与 bank 行为没有公开 trace 时只作推断。

## 前置知识

- `shape` 描述逻辑矩阵，layout 描述其物理线性化方式；shape 相同不等于可按字节复制。
- `Vec`、`Mat`、`Left`、`Right` 不只是标签，也对应 Vector、L1、L0A、L0B 等不同消费位置。
- `ND` 是普通 row-major；`NZ/ZN` 把矩阵拆成 fractal，并分别改变外层 block order 或 inner-fractal orientation。
- `valid shape` 限定本次指令定义的语义域，capacity 仍是编译期资源盒子。

## 今日两个核心问题

1. 给定 source/destination 的 TileType、layout、shape 与 dtype，A2/A3 和 A5 的 `TMOV` 到底会选择哪条实现？
2. 为什么 `TileLeft<T,M,K>` 在两个代际上源码名字相同，却不能据此推断物理格式相同？

## PTO 全栈中的位置

`TMOV` 位于“local Tile 已存在”和“下游算子按角色消费”之间。上游可以是 `TLOAD`、Vector compute 或 Accumulator；下游可以是 `TMATMUL`、`TSTORE` 或另一个 Vector op。

~~~mermaid
flowchart LR
    GM[GlobalTensor ND in GM] -->|TLOAD, MTE2| VND[Vec ND in UB]
    VND -->|A5 TMOV ND to NZ| VNZ[Vec NZ in UB]
    VND -->|A5 TMOV ND to ZN| VZN[Vec ZN in UB]
    VNZ -->|TSTORE or later role move| OUT1[packed consumer]
    VZN -->|TSTORE or later role move| OUT2[packed consumer]
    MAT[Mat in L1] -->|TMOV to Left| LEFT[Left in L0A]
    MAT -->|TMOV to Right| RIGHT[Right in L0B]
    LEFT --> MM[TMATMUL]
    RIGHT --> MM
~~~

这里最重要的边界是：`TMOV` 不是抽象的“随便转 layout”。它是由 target-specific 模板重载枚举出来的一组合法迁移；未命中语义分支，不应靠“模板能实例化”猜测得到某种重排。

## 概念和精确语义

先建立最小矩阵。表中的“普通 copy”表示保持线性/逐行搬运语义，不等于物理格式转换。

| 迁移 | A2/A3 | A5 | 关键结果 |
| --- | --- | --- | --- |
| `Vec ND → Vec ND` | 普通 `Vec→Vec` copy | 普通 `Vec→Vec` copy | 逻辑与物理次序均不改 |
| `Vec ND → Vec NZ` | **无显式重排分支** | 显式 `TMovND2NZ` | A5 按 fractal block 重排 |
| `Vec ND → Vec ZN` | **无显式重排分支** | 显式 `TMovND2ZN` | A5 还做 inner-fractal 转置式采集 |
| `Mat → Left` | 支持 | 支持 | A2/A3 的 `TileLeft` 是 ZZ；A5 是 NZ |
| `Mat → Right` | 支持 | 支持 | 两代 `TileRight` 都是 ZN |
| `Acc → Vec`、`Vec → Mat` | 不在主分派矩阵 | 支持 | A5 扩大了 location transition 集合 |
| `Mat → ScaleLeft/ScaleRight` | 不支持 | 支持 | A5 增加量化角色 Tile |

公共后置条件仍是：destination 的有效逻辑元素与 source 对应元素相等。但若 destination layout 不同，“相等”发生在逻辑坐标，不是相同 byte offset。

A5 的公共检查还要求 source/destination dtype 相同，并把可识别 source layout 限定为 row-major、NZ 或 ZN；`TMovToLeft` 要求 destination 为 NZ，`TMovToRight` 要求 destination 为 ZN。对应实现见 [`a5/TMov.hpp` 的 CommonCheck 与 role dispatch](https://github.com/hw-native-sys/pto-isa/blob/5a4f74cbf627d4aac2e0ce10d5e0d8b118343265/include/pto/npu/a5/TMov.hpp#L238-L252) 和 [`TMOV_TILE_IMPL`](https://github.com/hw-native-sys/pto-isa/blob/5a4f74cbf627d4aac2e0ce10d5e0d8b118343265/include/pto/npu/a5/TMov.hpp#L748-L801)。

## 真实文件、类型、API 或指令逐段解读

### 1. A2/A3：分派首先看 TileType

[`a2a3/TMov.hpp`](https://github.com/hw-native-sys/pto-isa/blob/5a4f74cbf627d4aac2e0ce10d5e0d8b118343265/include/pto/npu/a2a3/TMov.hpp#L137-L166) 先静态要求 source/destination capacity shape 相同，然后按类型对分派：

```text
Mat -> Left/Right/Bias/Scaling
Vec -> Vec
Acc -> Mat
```

`Mat→Left/Right` 会进入 `TMovToLeft/TMovToRight`，由 MTE1 语义完成 L1 到 Cube operand memory 的 role transfer。`Vec→Vec` 则是 Vector local copy。源码没有 A5 那种“检测 destination 是 NZ/ZN，改走专用 pack kernel”的分支。因此，在 A2/A3 上把两个不同 layout 的 Vec 类型塞进 generic path，即使表面 template 条件没有及早报错，也不能把结果宣称为 ND→NZ/ZN。

这揭示一个 verifier 规则：**语义合法性不能只用 `SrcType==Vec && DstType==Vec` 表示，还需要 target-aware layout transition 枚举。**

### 2. A5：先识别 role，再识别 Vec layout transition

A5 的 [`TMOV_TILE_IMPL`](https://github.com/hw-native-sys/pto-isa/blob/5a4f74cbf627d4aac2e0ce10d5e0d8b118343265/include/pto/npu/a5/TMov.hpp#L748-L801) 扩充为：

- `Mat→Left/Right/ScaleLeft/ScaleRight/Bias/Scaling`；
- `Acc→Vec/Mat`；
- `Vec→Mat`；
- `Vec→Vec` 内部再区分 ordinary copy、`ND→NZ`、`ND→ZN`。

`ND→NZ` 的实现使用 `vlds`/`vsstb` 组织分块 load/store，并检查 source valid rows 不超过 destination valid rows，见 [`TMovND2NZ`](https://github.com/hw-native-sys/pto-isa/blob/5a4f74cbf627d4aac2e0ce10d5e0d8b118343265/include/pto/npu/a5/TMov.hpp#L501-L548)。`ND→ZN` 通过 `vgather2` 重排 inner fractal，且要求：

```text
K0 = 32 bytes / sizeof(T)
Rows % K0 == 0
Cols % 16 == 0
```

实现见 [`TMovND2ZN`](https://github.com/hw-native-sys/pto-isa/blob/5a4f74cbf627d4aac2e0ce10d5e0d8b118343265/include/pto/npu/a5/TMov.hpp#L550-L644)。这些不是性能建议，而是构造合法物理盒子的静态前置条件。

### 3. 同名 `TileLeft` 的代际漂移

真正容易埋雷的地方在 [`pto_tile.hpp`](https://github.com/hw-native-sys/pto-isa/blob/5a4f74cbf627d4aac2e0ce10d5e0d8b118343265/include/pto/common/pto_tile.hpp#L1748-L1769)：

- A2/A3：`TileLeft` 的 `BLayout=RowMajor, SLayout=RowMajor`，即本课程记作 ZZ；
- A5：`TileLeft` 的 `BLayout=ColMajor, SLayout=RowMajor`，即 NZ；
- [`TileRight`](https://github.com/hw-native-sys/pto-isa/blob/5a4f74cbf627d4aac2e0ce10d5e0d8b118343265/include/pto/common/pto_tile.hpp#L1772-L1780) 在两代都是 ZN。

因此，`TileLeft<half,32,64>` 是 portable source spelling，却不是 portable physical ABI。生成器、序列化测试和跨 target golden 必须解析 target 后的完整类型，而不能只比较 alias 名称。

## 对象、Tile 与 Buffer 生命周期

以 A5 ND→ZN 直接测试为例：

~~~mermaid
sequenceDiagram
    participant GM as GlobalTensor ND in GM
    participant Src as srcTile Vec ND in UB
    participant Dst as znTile Vec ZN in UB
    participant Raw as ndTile raw view of same dst UB
    GM->>Src: TLOAD writes logical ND elements
    Note over Src: wait MTE2 to V before vector read
    Src->>Dst: TMOV ND to ZN writes packed bytes
    Note over Dst,Raw: two descriptors alias one UB address
    Note over Dst: wait V to MTE3 before external read
    Raw->>GM: TSTORE exposes physical packed bytes
~~~

`GlobalTensor` 只持有 GM descriptor，不拥有 allocation。`TLOAD` 在 source UB 形成 ND Tile；MTE2→V event 之后 `TMOV` 才能读取它。destination ZN Tile 拥有另一段 UB 写权限，完成后由 V→MTE3 event 交给 `TSTORE`。

测试中 `znTile` 与 `ndTile` 绑定同一个 destination UB 地址：前者是结构化写视图，后者是为了把物理字节原样写回 GM 的线性读视图。这个 alias 是测试观察手段，不意味着两种 layout 同时拥有独立 storage；同步完成前用 raw view 读取就是 RAW hazard。

## 端到端调用链或指令链

A5 两个直接 kernel 的真实链条是：

```text
GM ND GlobalTensor
  -> TLOAD(src Vec ND, GM)
  -> TASSIGN event MTE2 to V
  -> TMOV(dst Vec NZ or ZN, src Vec ND)
  -> TASSIGN event V to MTE3
  -> TSTORE(GM output, raw destination view)
```

可点击证据：[`tmov_nd2nz_kernel.cpp`](https://github.com/hw-native-sys/pto-isa/blob/5a4f74cbf627d4aac2e0ce10d5e0d8b118343265/tests/npu/a5/src/st/testcase/tmov_nd2nz/tmov_nd2nz_kernel.cpp) 与 [`tmov_nd2zn_kernel.cpp`](https://github.com/hw-native-sys/pto-isa/blob/5a4f74cbf627d4aac2e0ce10d5e0d8b118343265/tests/npu/a5/src/st/testcase/tmov_nd2zn/tmov_nd2zn_kernel.cpp)。它同时证明 `TMOV` 的数据依赖不是隐式全局 barrier：两侧 pipeline handoff 仍由调用者显式建立。

## 具体 shape、Tile 和状态演算

取 `half` 矩阵 `S[K,N]=S[32,64]`，共 2048 个元素、4096 B。观察逻辑元素 `S[17,50]`。

### ND→NZ

`C0=N0=16`，物理盒子可写作 `[C1=4,N1=2,16,16]`：

```text
D[c1,n1,n0,c0] = S[n1*16+n0, c1*16+c0]
```

对 `S[17,50]`：`n1=1,n0=1,c1=3,c0=2`，在 destination 中的线性 offset 为：

```text
(((3*2+1)*16+1)*16+2) = 1810
```

### ND→ZN

`K0=32/sizeof(half)=16`，物理盒子为 `[K1=2,N1=4,16,16]`：

```text
D[k1,n1,j,i] = S[k1*K0+i, n1*16+j]
```

同一元素得到 `k1=1,i=1,n1=3,j=2`，线性 offset 为：

```text
(((1*4+3)*16+2)*16+1) = 1825
```

源 ND offset 是 `17*64+50=1138`。三个 offset 不同，但逻辑坐标和值相同。这就是为什么 `memcpy`、NZ 和 ZN 不能互换。

## 为什么这样设计及替代方案

### 方案 A：运行时 local `TMOV`

优点是布局决策靠近 consumer，source ND Tile 还能被其他 Vector 算子复用；缺点是额外占一份 destination UB，并消耗 Vector load/store 或 gather 指令。对短 kernel，重排可能直接落在延迟关键路径。

### 方案 B：GM 中预先保存 NZ/ZN

模型权重可离线预打包，kernel 直接 `TLOAD` 匹配布局，避免每次 dispatch 重排。代价是保存多份格式、增加转换工具与版本 ABI，还可能不适合动态 activation。

### 方案 C：让 `TLOAD` 完成跨布局搬运

若 target 的 DMA 路径支持对应 GM→local 转换，可省掉中间 ND UB Tile；但 source global layout、对齐和 burst contract更严格，也减少了“先做 Vector compute，再决定 Left/Right role”的复用自由度。

选择原则不是“哪条指令更高级”，而是比较：重排频次、source reuse、UB 峰值、Vector/MTE overlap、预处理成本和跨 target 维护成本。

## 访存、计算、流水、并行和硬件约束

- `32×64×half` 的一次 local repack 至少读取并写入 4096 B 有效 payload；若不能与其他 pipe overlap，这就是额外 local traffic。
- A5 ND→NZ 的代码事实是 vector load/store 组合；ND→ZN 的代码事实是 `vgather2` 加 store。由此可以**推断** ZN 路径有更高的索引/指令压力，但没有 device trace 不能声称实际更慢。
- destination capacity、fractal divisibility 与 dtype 在编译期限定；valid region 仍必须保证 source 已初始化范围覆盖 destination 将读取的范围。
- source 与 destination 若别名，只有实现明确允许并证明读写顺序时才安全；当前直接测试使用独立 source/destination UB。
- `TMOV` 完成不等于下游 pipe 可见，MTE2→V 与 V→MTE3/M 的 handoff 仍需 event 或编译器生成的等价同步。

## 测试证据与未覆盖风险

### 已有测试事实

`tmov_nd2nz` 的实际 NPU case 只覆盖 `hifloat8` 的 `32×32`、`32×64`、`64×64`，并逐字节比较 golden。golden 将 ND reshape 后按 `[c1,n1,n0,c0]` 排列，见 [`gen_data.py`](https://github.com/hw-native-sys/pto-isa/blob/5a4f74cbf627d4aac2e0ce10d5e0d8b118343265/tests/npu/a5/src/st/testcase/tmov_nd2nz/gen_data.py)。

`tmov_nd2zn` 覆盖 `hifloat8`、`half`、32-bit 类型的 `32×32`、`32×64`、`64×64`、`128×128`，同样比较完整物理字节；golden 固定 `K0=32/sizeof(T)` 与 `[k1,n1,j,i]` 映射，见 [`ND→ZN gen_data.py`](https://github.com/hw-native-sys/pto-isa/blob/5a4f74cbf627d4aac2e0ce10d5e0d8b118343265/tests/npu/a5/src/st/testcase/tmov_nd2zn/gen_data.py)。

直接相关的已合入提交 [`5e98634a`](https://github.com/hw-native-sys/pto-isa/commit/5e98634aadfef35468b9bf8ae96fe8e2d531973c) 增加 A5 SIMD ND→ZN 支持及测试；提交说明记录“全部 ST 通过”是提交者报告，不等于本文重新上板测量。

### 仍未覆盖

- 没有针对非法 TileType pair、dtype mismatch、`Rows%K0!=0`、`Cols%16!=0` 的系统 negative compile matrix。
- ND→NZ 的实际 case dtype/shape 覆盖明显窄于实现宣称集合。
- ND→ZN 未覆盖非满 valid region、tail 与 poison padding，不能证明 partial Tile 不越读。
- 没有 A2/A3 “误把 generic Vec copy 当 repack”的反例 golden。
- 缺少 `TMOV→TMATMUL` 的 A5 consumer E2E、UB 峰值与 Vector/MTE/Cube stall trace。

## 与前后章节的连接

- 向前连接课程 02：layout 是类型 contract，不是注释。
- 修正课程 07 的可移植边界：A2/A3 `TileLeft=ZZ` 结论不能外推到 A5；`TileRight=ZN` 才在这两个代际保持一致。
- 连接课程 15–19：如果 compiler 只看同名 alias 或相同 shape，PlanMemory 与 InsertSync 都无法补救错误的 layout semantic。
- 下一章将区分 `TMOV`、`TTRANS`、`TRESHAPE`：一个改变物理表示，一个改变逻辑坐标关系或 view contract，不能只按输出 shape 判断等价。

## 本篇结论、知识债、理解检查与下一章

结论只有三条：

1. `TMOV` 的合法性键是 `(target, SrcTileType, DstTileType, src layout, dst layout, shape, dtype)`，不能缩成“都是 Vec”或“shape 相同”。
2. A5 的 ND→NZ/ZN 是显式物理重排；A2/A3 generic `Vec→Vec` 没有相同语义分支。
3. `TileLeft` 是 target-conditioned source alias：A2/A3 为 ZZ，A5 为 NZ；跨 target 工具必须比较展开后的完整 layout。

知识债：A5 partial-valid/tail、非法组合 negative tests、A2/A3 误用反例、真实 stall/UB 峰值，以及 `TMOV→TMATMUL` consumer contract。

三个理解检查问题：

1. `S[17,50]` 在 NZ 与 ZN 中 offset 不同，为什么数学结果仍能相同？
2. A2/A3 上 `Vec ND→Vec NZ` 若碰巧能编译，为什么仍不能据此认定合法？
3. 一个跨 target cache key 若只包含 `TileLeft<half,32,64>` 字符串，会发生哪类 ABI 错配？

下一章：**`TTRANS/TRESHAPE` 与 `TMOV` 的边界——逻辑坐标、view 与物理重排分别由谁负责。**

## 课程账本增量

- 日期：2026-09-08；章节：29。
- 源码基线：pto-isa [`5a4f74cb`](https://github.com/hw-native-sys/pto-isa/commit/5a4f74cbf627d4aac2e0ce10d5e0d8b118343265)。
- 新覆盖：A2/A3 与 A5 `TMov.hpp`、`pto_tile.hpp` 的 target-conditioned aliases、A5 `tmov_nd2nz/tmov_nd2zn` kernel 与 golden、提交 `5e98634a`。
- 新不变量：layout-changing move 必须命中 target-specific semantic branch；同名 alias 不构成跨 target physical ABI；destination 的逻辑有效域必须被 source 初始化域覆盖；pipeline handoff 仍需同步。
- 下一章：`TTRANS/TRESHAPE` 与 `TMOV` 的职责边界。
