# per-op profiler：不猜瓶颈

> 状态：**已实现并定版**，实测零开销
> 来源：移植自 [smolmaxxing](https://github.com/) 的 `GGML_OPPROF`（MIT）
> 唯一平台改动：`clock_gettime(CLOCK_MONOTONIC)` → `ggml_time_us()`（MSVC 没有前者）

---

## 0. 一句话

`GGML_OPPROF=1` 让 thread 0 给每个节点计时（含节点后的屏障，也就是最慢线程的耗时），
退出时打一张 per-op 表。**换任何模型，十分钟就知道钱花在哪**，不用猜、不用读源码。

---

## 1. 两个级别

| 变量 | 作用 |
|---|---|
| `GGML_OPPROF=1` | 退出时打 per-op 汇总表：调用数、总时间、占比、每线程耗时 |
| `GGML_OPPROF=2` | 再加：把**一张**图逐节点 dump 出来（图号由 `GGML_OPPROF_GRAPH` 指定，默认 12） |

`=2` 才是找不均衡的工具。它输出每节点的**每线程耗时**和**启动偏斜**，
`max/mean` 一眼可见。

```
[optrace] 12 MUL_MAT_ID blk.3.ffn_gate_inp-0 | 892.1 1204.7 903.3 ... | cpu 0 4 8 ...
              ↑图号 ↑op        ↑节点名          ↑每线程耗时(us)            ↑逻辑 CPU
```

---

## 2. 三个实现细节（都是踩出来的）

### 2.1 必须存原始下标 `prof_n`

图计算里 fusion 会推进 `node_n`，如果用推进后的下标做统计，**账会记到错的 op 上**。
所以在循环开头存下原始 `prof_n`，trace 汇总用它。

症状很隐蔽：表能打出来、总数也对，但某些 op 的时间被算到邻居头上，
于是你会去优化一个根本不热的算子。

### 2.2 计时必须包住屏障

只计 `compute_forward` 本身得到的是「本线程干了多久」，而关键路径是「最慢的线程干了多久」。
把节点后的屏障一起计进去，才等于关键路径上的耗时。

同时另存一份 `g_opprof_c`（纯 compute，不含屏障等待），
两者之差就是**这个节点上的等待时间**，也就是不均衡的直接度量。

### 2.3 MSVC 没有 `clock_gettime`

改成 `(double) ggml_time_us() * 1e-6`。精度足够（微秒级），且不需要引 `<time.h>` 的平台分支。

---

## 3. 开销

| | tok/s |
|---|---|
| 不开 | 19.26 |
| `GGML_OPPROF=1` | 19.29 |

在噪声内。所以**可以长期留着**，不需要为了测速重新编译一份干净的。

（这也是它比"临时插桩"有价值的地方：临时插桩每次都要重编，而且改完就走样了。）

---

## 4. 怎么读这张表

### 4.1 先看占比，再看 max/mean

占比只告诉你"这里花时间多"，`max/mean` 才告诉你"这里有没有浪费"。

```
MUL_MAT_ID   46.4%   max/mean 1.173   ← 花时间多，而且有 17% 白等
MUL_MAT:q8_0 44.6%   max/mean 1.09    ← 花时间多，但基本均衡
ADD_ID        3.1%   max/mean 1.02    ← 小 op，均衡
```

**一个 op 占比高不等于该优化它**。`MUL_MAT:q8_0` 占 44.6%，但它是带宽受限的，
而且已经跑到 62~66 GB/s，没有可回收空间。真正能拿的是那个 1.173。

### 4.2 用 p90 而不是 max 判断"是不是离群点"

`max` 会被单个离群线程带偏。把 `max` 换成 `p90` 重算一遍：

```
按 max 算：浪费 1,321 μs
按 p90 算：浪费   936 μs   ← 385 μs 是单次离群，不可预测也不可回收
```

如果两个数差很多，说明相当一部分"浪费"是噪声，别去追。

### 4.3 按逻辑 CPU 聚合，找硬件事实

把同节点内的耗时按 `us / mean` 归一化，再按逻辑 CPU 聚合：

```
cpu 2~9     rel 1.07 ~ 1.11   ← 系统性慢 8~10%，稳定
cpu 其余    rel 0.90 ~ 0.99
```

这是**这台机器的永久事实**，不是某次测量的噪声。知道它之后：
- 不会再幻想"多核就一定快"
- 也不会去做"避开慢核"这种傻事（见下）

⚠️ **别猜逻辑编号到物理核的映射**。这台机器上 `0-5 / 6-11` 是 SMT 兄弟，
不是"先 6 个核再 6 个兄弟"。用 `-C 0xFF555` 之类的方式强行绑核，
实测 13.9 → 7.0 tok/s 腰斩。要绑核先实测映射，别按直觉排。

### 4.4 用 MUL_MAT_ID 分派计数器确认走的是哪条路径

`g_opprof_mmid_gen`（generic 路径，`ggml-cpu.c`）和 `g_opprof_mmid_repack`（repack 路径，
`repack.cpp`）在表里会打出来：

```
MUL_MAT_ID calls: generic=0 repack=648
```

**这条数据曾救过一次大返工**：改 `ggml-cpu.c` 里的通用实现之前看到 `generic=0`，
说明实际执行的是 repack 路径，改错地方等于白干。

---

## 5. 用法

```bash
# 打表
GGML_OPPROF=1 ./build/bin/llama-server -m model.gguf ...

# 单图逐节点（默认第 12 张图，通常正好是第一个 decode step）
GGML_OPPROF=2 ./build/bin/llama-bench -m model.gguf -p 0 -n 1 -t 18 2>&1 | tail -60

# 指定图号
GGML_OPPROF=2 GGML_OPPROF_GRAPH=40 ./build/bin/llama-server ...
```

配合 `specs/mul-mat-id-load-balance.md` 的分析脚本思路：
按 op 算 `max/mean` 浪费、按节点名聚合、按逻辑 CPU 归因。
