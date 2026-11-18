#!/usr/bin/env bash
# 重建我们这一版 llama.cpp：clone 上游 → 退到基线 → 打 patch → 编译。
#
#   bash scripts/setup.sh                    # 建在 <repo>/llama.cpp（已 gitignore）
#   DEST=/d/llama.cpp bash scripts/setup.sh  # 建在别处，供多个项目共用
#   SKIP_BUILD=1 bash scripts/setup.sh       # 只打 patch，不编译
#
# patch 必须打在 **e351231** 上。上游每周几十个提交，ggml 那块重构频繁，
# 打不上时 git apply 会指到一个跟你改动无关的地方。
#
# 这个 commit 不是随手挑的：拿 patch 里 20 个文件的前像 blob，在官方 master 历史里
# 反查，只有它（2026-09-25）能让这 20 个文件与本 patch 的基线逐字节一致。
# 早先文档写的是 `5096f2a`，那是个**错误**：它是我们自己建的快照 commit
# （author baseline@local、无 parent、工作区在 llama.cpp-master/ 下），官方仓库里
# 没有这个对象 —— `git fetch upstream 5096f2a` 会报 "not our ref"。
# 别用"日期最接近"的 commit 顶替：一天之差就能让 17/20 个文件被上游改过。
# 自己验基线的办法见 patches/README.md「怎么自己验基线」。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN=e351231c4f4cdd89c88e696c46d0eb718c9e0ab5
DEST="${DEST:-$HERE/llama.cpp}"
PATCH="$HERE/patches/cpu-afterburner.patch"

[ -f "$PATCH" ] || { echo "找不到 patch: $PATCH" >&2; exit 2; }

if [ -d "$DEST/.git" ]; then
    echo "=== 复用已有克隆：$DEST"
    cd "$DEST"
    git fetch --tags origin || true
    # 已经打过 patch 的树直接重来一遍会失败（上下文已变），所以先丢弃本地改动。
    git checkout -f "$PIN" -- . 2>/dev/null || true
else
    echo "=== clone 到 $DEST"
    # --filter=blob:none 只按需取 blob；checkout 目标 commit 时会补齐。
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$DEST"
    cd "$DEST"
fi

echo "=== 退到基线 $PIN"
git checkout -f "$PIN"

echo "=== 打 patch"
git apply --check "$PATCH"
git apply "$PATCH"
echo "    ok"

if [ "${SKIP_BUILD:-0}" = "1" ]; then
    echo "=== SKIP_BUILD=1，跳过编译"
    exit 0
fi

echo "=== 配置 & 编译"
# Windows 上最大的坑：MSVC 的 INCLUDE / LIB 只在「x64 Native Tools Command Prompt」
# 或 vcvars64.bat 里才有。在普通 shell 里直接 cmake 会挂在
#   ggml.h(211) fatal error C1083: 无法打开 stdbool.h
# 报错位置看起来像源码问题，其实是环境变量没设。
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    if [ -z "${INCLUDE:-}" ]; then
      echo "错误：INCLUDE 为空 —— 当前不是 MSVC 环境。" >&2
      echo "     请在「x64 Native Tools Command Prompt for VS 2022」里运行，或先执行：" >&2
      echo '       call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"' >&2
      exit 2
    fi
    ;;
esac

# GGML_NATIVE=ON 是故意的：让编译器按本机 CPU 生成指令。关掉能跑但会慢一截，
# 然后你会以为是 patch 没用。GGML_AVX_VNNI 保持默认关（见 docs/pitfalls.md）。
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
    -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF
# llama-tts 一定要编：TTS 那 4 个文件的改动只有这个 target 会走到。
cmake --build build --target llama-server llama-bench llama-tts -j

echo
echo "=== 完成。产物在 $DEST/build/bin/"
echo "    llama-server  —— LLM 推理"
echo "    llama-tts     —— Qwen3-TTS 合成（@spk= / @seed= 控制行）"
echo
echo "    想看瓶颈：GGML_OPPROF=1 ./llama-server ... （退出时打 per-op 表）"
