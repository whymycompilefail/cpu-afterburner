# cpu-afterburner

**一台 CPU 上跑 MoE 推理和 TTS 的 llama.cpp，改到没有免费的性能可捡为止。**

加力燃烧不换发动机——往尾喷管里再注一次油，让同样的引擎多出推力。
这个仓库干的是同一件事：**不换模型、不改架构、不动量化方案**，只改执行层
（切分、循环顺序、图宽度），把已有的算力榨干。

---

## 这是什么

**不是一份 fork 源码，是一个 patch。**

```
patches/cpu-afterburner.patch      20 files, +1428 / -194,  112 KB
scripts/setup.sh                   clone 上游 -> 退到基线 -> 打 patch -> 编译
```

`setup.sh` 会 clone 官方 llama.cpp、退到 `5096f2a`、打上 patch、编出
`llama-server` / `llama-bench` / `llama-tts`。

为什么不直接发源码：llama.cpp 有 3,600 多个文件，全搬进仓库既臃肿又没法 review。
patch 的好处是**改动永远可见**——你随时能看到"他到底动了什么"，而不是在一堆
vendor 代码里比对。

## 快速开始

```bash
bash scripts/setup.sh            # 默认建在 ./llama.cpp（已 gitignore）
# 或者建到公共位置，给多个项目共用：
DEST=/d/llama.cpp bash scripts/setup.sh
```

Windows 上必须在 MSVC 环境里跑（`INCLUDE` / `LIB` 要有值），否则会挂在
`fatal error C1083: 无法打开 stdbool.h`——那个报错看起来像源码问题，其实是环境变量没设。

基线是 `5096f2a`，**别换成别的 commit**。上游 `ggml/src/ggml-cpu/` 每周都在重构。

---

## 改了什么

| 主题 | 效果 | spec |
|---|---|---|
| **MUL_MAT_ID 负载切分** | decode **+5.7%** / prefill **+18.3%** | [mul-mat-id-load-balance](specs/mul-mat-id-load-balance.md) |
| **per-op profiler** | 零开销，随时能看钱花在哪 | [op-profiler](specs/op-profiler.md) |
| 自适应 flash attention | 按 KV 长度在 FA / legacy 间切 | [fa-auto-adaptive](specs/fa-auto-adaptive.md) |
| GQA 循环重排 | KV 流量倍数 2.2~3.4 → 1 | [fa-gqa-loop-reorder](specs/fa-gqa-loop-reorder.md) |
| 量化 V 的 legacy attention | 新算子 `GGML_OP_MUL_MAT_TQ` | [legacy-quant-v-op](specs/legacy-quant-v-op.md) |
| TTS 图按真实帧数构建 | 尾窗不再为 padding 付算力 | [tts-gen-wav-window](specs/tts-gen-wav-window.md) |
| TTS 常驻循环 + 控制行 | 去掉每次调用的启动税 | [tts-resident-loop-and-control](specs/tts-resident-loop-and-control.md) |

这里没有一项是"换个更快的库"或者"用更好的量化"。每一项都是**测量指出来之后才动的**，
而且每项都留下了可以被别人重跑的对照。

---

## 唯一一个值得看的数字，和它的三个版本

`MUL_MAT_ID` 那项，同一批交替 A/B：

| 口径 | 数字 |
|---|---|
| `tg128` @t=14，短上下文 bench | 18.82 → 19.90（**+5.7%**，4/4 轮单向） |
| `pp512` @t=14 | 49.55 → 58.64（**+18.3%**，4/4 轮单向） |
| 各自最优线程数（stock@16 vs dyn@18） | decode **+8.0%**，prefill **+22.9%** |

然后是端到端：54 题批处理，墙钟 **683 s → 638 s（−6.6%）**。
但同一批的**真实请求 decode 是 18.46 → 18.34，持平**。

**这三件事不是一回事，不能互相替代。** 短上下文 bench 测的是纯 decode；
真实请求里长上下文的 KV 读取把 decode 收益摊平了；墙钟那 45 秒几乎全来自 prefill。

所以准确的说法是：**这笔收益主要在 prefill，不在 decode。**
如果只想要"生成快一点"，这个 patch 给你的比你期望的少。

---

## 为什么会失衡（机制，不是玄学）

MoE 的 decode 里，每个 token 路由到不同专家。stock 的实现把 `(专家, 行)` 空间
静态切成连续片段、一线程一段——但**每个专家分到的 token 数本来就不均匀**。
分到小专家的线程几微秒就干完，然后站在屏障上等；分到大专家的线程跑到底。

profiler 给出的证据：`MUL_MAT_ID` 的 `max/mean = 1.173`（关键路径白等 17%），
而同一个线程池上 head 那个 615 MB 的单节点是 **1.002**——**不是线程池坏了，是切分策略不适配**。

改成抢块后 `1.030`。收益来自**跨专家领活**而不是切得更细：
粒度 2 / 4 / 8 三档全在噪声内。

---

## 被推翻的结论

这个仓库的技术判断里，有一半是"想错了然后实测打回来"的。列在这里，免得别人重走：

| 曾经的结论 | 现状 |
|---|---|
| "decode 每 token 只能拿到约 31 GB/s，等的是内存" | **错**。head 单 op 实测 **62~66 GB/s**。瓶颈是**每元素解包指令成本**，不是带宽——省 47% 字节只换来 6% 速度就是证据 |
| "减字节 = 线性提速" | **错**。实得带宽随 B/elem 从 41.8 掉到 22.8，压缩反而降低效率 |
| "把 head 量化到 Q4 能省 28% 每 token 字节" | 只值 **+1.2% ~ +6.4%**。Q4_0 解包要 `shuffle_epi8` 查表，省下的字节被指令吃回去 |
| "动态切分比静态慢 5%" | **错**。那是运行期 `if` 包装的产物（−26%），不是动态切分本身 |
| "切得越细越好" | **错**。2/4/8 噪声内，收益来源是跨专家领活 |
| "应该避开那 8 个慢核" | **错**。只用 12 个快核是最差配置，比 18 线程低 13%。慢核照样贡献带宽 |
| "多加点线程总没错" | **错**。stock 下 t=22 掉 17%；抢块后最优点才上移到 18 |

`specs/quant-dot-efficiency.md` 里那份"q5_0 追上 q8_0 就能 +25%"是**草案、从未实现**。
留在那里是因为它记录了"每元素指令成本才是瓶颈"这个结论的来历，不代表仓库有对应代码。

---

## 和其他仓库的关系

这个仓库是**唯一真源**，另外三个项目都从这里取 llama.cpp：

| 项目 | 用途 |
|---|---|
| [cpu-only-ai-tuber](https://github.com/whymycompilefail/cpu-only-ai-tuber) | 本地虚拟主播（LLM + TTS 主链路） |
| AlterVox | 全 CPU 语音管线（ASR → 情感 → TTS） |
| [cpu-intern](https://github.com/whymycompilefail/cpu-intern) | 代码修复流水线（本地小模型 + 云端升级） |

早期这三个仓库各自收编了一份 llama.cpp，其中两份几乎全同（99% 的文件是同一个上游），
只有 4 个文件版本不同，**且差异全是旧的那份落后**。现在收拢到这里，谁也不许再扁平化一份进去。

用法：把 `DEST` 指到同一个位置，几个项目共用一份构建产物。

---

## 不含什么

- **不含权重**。模型自己去 HuggingFace 找。
- **不含上游代码**。由 `setup.sh` 重建。
- **不含构建产物**。
- **不含那个"51 题的分数"**。那属于 cpu-intern，且依赖不公开的题池，
  在这里引用它等于不可复现的主张。

## License

MIT。移植部分（profiler）来自 smolmaxxing（同为 MIT），已在对应 spec 中标注。
