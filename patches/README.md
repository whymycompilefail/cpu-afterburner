# patch 说明

```
cpu-afterburner.patch     20 files changed, 1428 insertions(+), 194 deletions(-)   112 KB
```

## 基线

| | |
|---|---|
| 上游 commit | `5096f2a42468daff7ff38d97b48bd17432196e98` |
| 相对的上游版本 | 2026-10 中旬 |
| patch 相对谁生成 | fork 内的 `c649f60` |

`c649f60` 是把 fork 从独立仓库 flatten 进上层项目时产生的提交。它跟上游 `5096f2a`
**内容完全一致、只是目录层级不同**（`git diff --summary -M 5096f2a c649f60` 只有 rename 行）。
所以 patch 可以直接打在 `5096f2a` 上，不需要先复刻 flatten。

**别打在别的 commit 上**。上游 `ggml/src/ggml-cpu/` 每周都在重构，打不上时 `git apply`
报的位置会跟你说"某个不相干的函数对不上"，很容易误判成 patch 坏了。

## 20 个文件，按主题分组

### A. MUL_MAT_ID 负载切分（decode +5.7% / prefill +18.3%）

| 文件 | 改动 |
|---|---|
| `ggml/src/ggml-cpu/repack.cpp` | `+129` 抢块调度：`MMID_DYNAMIC` 编译期开关、块网格对齐 `NB_COLS=8`、前缀和、跨专家领活 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | generic / repack 两条路径的分派计数器（`g_opprof_mmid_*`） |

spec：`specs/mul-mat-id-load-balance.md`

> **MoE 模型才会走到这里**。dense 模型的 `MUL_MAT_ID` 调用数是 0，不会有任何变化。

### B. per-op profiler

| 文件 | 改动 |
|---|---|
| `ggml/src/ggml-cpu/ggml-cpu.c` | `+60` `GGML_OPPROF=1` 打表 / `=2` 单图逐节点 trace，含每线程耗时与启动偏斜 |

spec：`specs/op-profiler.md`

零开销（实测 19.29 vs 19.26 tok/s）。移植自 smolmaxxing，唯一平台改动是
`clock_gettime` → `ggml_time_us()`（MSVC 没有前者）。

### C. 自适应 flash attention（Spec 01）

| 文件 | 改动 |
|---|---|
| `include/llama.h` | `fa_kv_threshold` / `fa_max_tokens` 参数 |
| `common/common.h`、`common/arg.cpp`、`common/common.cpp` | `-fakv` / `--fa-kv-threshold` 命令行 |
| `src/llama-context.cpp`、`src/llama-cparams.h`、`src/llama-graph.cpp`、`src/llama-model.cpp` | 每个 decode step 按 KV 长度选 FA / legacy |

spec：`specs/fa-auto-adaptive.md`

一句话：CPU 上 FA 的退化速度比 legacy 快得多，所以 `n_kv <= 阈值` 用 FA，超了切 legacy。
运行时开关：`-fakv N`（0 = 一直 FA）。

### D. GQA 循环重排（Spec 02）

| 文件 | 改动 |
|---|---|
| `ggml/src/ggml-cpu/ops.cpp` | f16 attention kernel 里交换循环顺序：每个 K/V 行只读一次，再分给该 kv 组的所有 query head |

spec：`specs/fa-gqa-loop-reorder.md`

本机实测 KV 流量倍数：Qwen3-0.6B（GQA ratio 2）≈ 2.2，Qwen3-4B / glm-edge-4b（ratio 4）≈ 3.4。
重排后倍数降到 1。只对单行 decode 生效（`neq1 == 1 && neq3 == 1`），其余回落原 kernel。
运行时开关：`LLAMA_FA_GQA=0`。

### E. 量化 V 的 legacy attention（Spec 03）

| 文件 | 改动 |
|---|---|
| `ggml/include/ggml.h`、`ggml/src/ggml.c` | 新算子 `GGML_OP_MUL_MAT_TQ`（**追加在枚举尾部**，插中间会平移所有值） |
| `ggml/src/ggml-cpu/ops.h`、`ops.cpp` | kernel：decode 走 partials、prefill 走直接写 |
| `ggml/src/ggml-cpu/common.h` | `GGML_MUL_MAT_TQ_MAX_N 16`：两种模式的分界，wsize planner 与 kernel 共用同一个宏 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | wsize 规划 |

spec：`specs/legacy-quant-v-op.md`

### F. TTS 图窗口优化（GEN_WAV）

| 文件 | 改动 |
|---|---|
| `tools/mtmd/clip.cpp` | 图按**真实帧数**构建；尾部窗口不再付整个 attention 窗口的代价，也不需要 code-0 padding 和后续裁剪 |
| `tools/mtmd/models/models.h` | `clip_graph_qwen3tts_gen` 增加 `n_frames` 成员 |
| `tools/mtmd/models/qwen3tts-gen.cpp` | 用真实 `n_frames` 建张量，加 `GGML_ASSERT(n_frames > 0 && n_frames <= wav_tfm_swa)` |

spec：`specs/tts-gen-wav-window.md`

这是纯算力优化：短尾窗口原来要算满 `wav_tfm_swa` 帧，现在只算需要的。

### G. TTS 控制行与线程透传

| 文件 | 改动 |
|---|---|
| `tools/tts/tts.cpp` | `+215` `@spk=<path>` 切换 speaker 参考、`@seed=<n>` 重播种 sampler 与音频码 RNG；sampler 生命周期归本处管理；`n_threads` 透传给 mtmd |

spec：`specs/tts-resident-loop-and-control.md`

`@seed` 的用途是**让失败可复现**：固定 seed 会永远以同样方式失败，换 seed 重试才有意义。
`n_threads` 透传是性能修复——不透传的话 TTS 的 mtmd 侧拿不到你指定的线程数。

### H. 模型转换脚本

| 文件 | 改动 |
|---|---|
| `conversion/base.py` | `+5` 补一个 deepseek-coder-1.3b-base 的 tokenizer 哈希（与 6.7b 同构但 vocab 32000，哈希不同，上游列表没登记） |

## 活跃开关一览

| 开关 | 类型 | 作用 |
|---|---|---|
| `MMID_DYNAMIC` | **编译期**（`repack.cpp` 顶部） | 0 = stock 静态切分，1 = 抢块（默认 1） |
| `MMID_DYNAMIC_CHUNKS` | 编译期 | 每线程切几块（默认 4；实测 2/4/8 在噪声内，收益来自跨专家领活而非切得细） |
| `LLAMA_FA_GQA=0` | 运行期 | 关掉 Spec 02 的 GQA 重排 |
| `-fakv N` | 运行期 | Spec 01 的 FA / legacy 分界（0 = 一直 FA） |
| `GGML_OPPROF=1` / `=2` | 运行期 | 打 per-op 表 / 单图逐节点 trace |

⚠️ `MMID_DYNAMIC` **必须是编译期宏**：把同一段 stock 循环原样包进运行期 `if`（else 为空）
就掉 26%（19.26 → 14.32 tok/s）。`gemv<>` 是巨型模板，MSVC 一旦见它落在分支内就不再生成
同样的代码。这条踩过，见 `docs/pitfalls.md`。

## 这个 patch 不包含什么

- **`specs/quant-dot-efficiency.md` 描述的东西没有实现**。那份文档自述"草案，待审阅，
  未写任何生产代码"，只做了测量。放在 `specs/` 里是为了保留"量化点积是每元素指令成本
  受限"这个结论的来历，不代表仓库里有对应代码。
- **不含权重，不含题池，不含构建产物**。
- **不含上游代码**。这个仓库只有 patch + 脚本，源码由 `scripts/setup.sh` 重建。
