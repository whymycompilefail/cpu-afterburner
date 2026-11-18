#!/usr/bin/env bash
# 校验 llama.cpp/ 这棵树确实等于「上游 e351231 + patches/cpu-afterburner.patch」。
#
#   bash scripts/verify-tree.sh
#
# 三项检查，任意一项不过就退出码 1：
#   A. 与上游基线不同的文件，恰好是 patch 里那 20 个（不多不少）
#   B. 这 20 个文件的 blob 哈希 == patch 里写的后像哈希
#   C. 基线里有、我们没有的，只允许是 docs/ 与 models/（这两个目录是故意不跟的）
#
# 判据用 blob 哈希而不是 diff，是为了绕开行尾：这台机器系统级
# core.autocrlf=true，git apply 会把补丁写成 CRLF，看起来"改了很多行"。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=e351231c4f4cdd89c88e696c46d0eb718c9e0ab5
PATCH="$HERE/patches/cpu-afterburner.patch"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/verify-tree.XXXXXX")"
trap 'rm -rf "$WORK" 2>/dev/null || true' EXIT

[ -f "$PATCH" ] || { echo "找不到 patch: $PATCH" >&2; exit 2; }
git -C "$HERE" ls-files --error-unmatch llama.cpp/CMakeLists.txt >/dev/null 2>&1 || {
    echo "llama.cpp/ 没有被 git 跟踪 —— 检查 .gitignore" >&2; exit 2; }

echo "=== 取上游基线 $BASE（只取树，不取文件内容）"
git -C "$WORK" init -q
# --filter=blob:none：校验只比对 blob 哈希，不需要真的下载那 60MB 内容。
git -C "$WORK" fetch -q --depth=1 --filter=blob:none \
    https://github.com/ggml-org/llama.cpp "$BASE"
git -C "$WORK" ls-tree -r FETCH_HEAD | awk '{print $4"\t"$3}' | sort > "$WORK/base.txt"

echo "=== 取 patch 声明的 20 个后像"
awk '/^diff --git/{f=substr($3,3)} /^index /{h=$2; i=index(h,".."); print f"\t"substr(h,i+2)}' \
    "$PATCH" | sort > "$WORK/post.txt"

echo "=== 取本仓库索引里的 blob"
git -C "$HERE" ls-files -s llama.cpp |
    awk '{p=$4; sub(/^llama.cpp\//,"",p); print p"\t"$2}' | sort > "$WORK/ours.txt"

join -t$'\t' "$WORK/base.txt" "$WORK/ours.txt" |
    awk -F'\t' '$2!=$3{print $1}' | sort > "$WORK/changed.txt"

fail=0

echo
echo "--- A. 与基线不同的文件 ---"
n_changed=$(wc -l < "$WORK/changed.txt")
n_post=$(wc -l < "$WORK/post.txt")
if [ "$n_changed" -eq "$n_post" ] &&
   [ -z "$(comm -3 <(cut -f1 "$WORK/changed.txt") <(cut -f1 "$WORK/post.txt"))" ]; then
    echo "    ok：$n_changed 个，与 patch 的文件清单完全一致"
else
    echo "    失败：我们改了 $n_changed 个，patch 声明 $n_post 个" >&2
    echo "    只在树里变了的:" >&2
    comm -23 <(cut -f1 "$WORK/changed.txt") <(cut -f1 "$WORK/post.txt") | sed 's/^/      /' >&2
    echo "    只在 patch 里的:" >&2
    comm -13 <(cut -f1 "$WORK/changed.txt") <(cut -f1 "$WORK/post.txt") | sed 's/^/      /' >&2
    fail=1
fi

echo "--- B. 这 20 个是否等于 patch 后像 ---"
bad=$(awk -F'\t' 'NR==FNR{post[$1]=$2; next} ($1 in post) && substr($2,1,length(post[$1]))!=post[$1]{print $1}' \
        "$WORK/post.txt" "$WORK/ours.txt")
if [ -z "$bad" ]; then
    echo "    ok：$n_post/$n_post 对上"
else
    echo "    失败：" >&2; echo "$bad" | sed 's/^/      /' >&2; fail=1
fi

echo "--- C. 基线有、我们没有的（只允许 docs/ 与 models/）---"
comm -23 <(cut -f1 "$WORK/base.txt") <(cut -f1 "$WORK/ours.txt") > "$WORK/missing.txt"
unexpected=$(grep -vE '^(docs|models)/' "$WORK/missing.txt" || true)
echo "    缺失 $(wc -l < "$WORK/missing.txt") 个（docs $(grep -c '^docs/' "$WORK/missing.txt" || true) / models $(grep -c '^models/' "$WORK/missing.txt" || true)）"
if [ -n "$unexpected" ]; then
    echo "    失败：这些不在 docs/ 或 models/ 里却缺了：" >&2
    echo "$unexpected" | sed 's/^/      /' >&2; fail=1
else
    echo "    ok：没有意外缺失"
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "=== 通过：这棵树 = 上游 $BASE + patch，逐字节。"
else
    echo "=== 不通过（见上）。" >&2
    exit 1
fi
