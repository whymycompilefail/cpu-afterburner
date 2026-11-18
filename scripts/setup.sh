#!/usr/bin/env bash
# 编译本仓库自带的 llama.cpp。
#
#   bash scripts/setup.sh                      # 构建目录 llama.cpp/build
#   BUILD=/d/build-cpu bash scripts/setup.sh    # 构建目录放别处（多个项目共用）
#
# 源码就在 llama.cpp/ 里：**不用联网，不用 clone 上游，不用打 patch**。
# 这棵树 = 上游 e351231（2026-09-25）+ patches/cpu-afterburner.patch，逐字节等价。
# 想自己核这件事（而不是信这句话），跑 scripts/verify-tree.sh。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$HERE/llama.cpp"
BUILD="${BUILD:-$SRC/build}"

[ -f "$SRC/CMakeLists.txt" ] || {
    echo "找不到源码树: $SRC" >&2
    echo "它应该随仓库一起来。如果你是下载 zip 或用了 sparse-checkout，确认 llama.cpp/ 在里面。" >&2
    exit 2
}

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

echo "=== 配置 ($SRC -> $BUILD)"
# GGML_NATIVE=ON 是故意的：让编译器按本机 CPU 生成指令。关掉能跑但会慢一截，
# 然后你会以为是 patch 没用。GGML_AVX_VNNI 保持默认关（见 docs/pitfalls.md）。
cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
    -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF

echo "=== 编译"
# llama-tts 一定要编：TTS 那 4 个文件的改动只有这个 target 会走到。
cmake --build "$BUILD" --target llama-server llama-bench llama-tts -j

echo
echo "=== 完成。产物在 $BUILD/bin/"
echo "    llama-server  —— LLM 推理"
echo "    llama-tts     —— Qwen3-TTS 合成（@spk= / @seed= 控制行）"
echo
echo "    想看瓶颈：GGML_OPPROF=1 ./llama-server ... （退出时打 per-op 表）"
