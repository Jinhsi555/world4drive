#!/usr/bin/env bash
# 改进：
# 1. 自动定位工程根目录，避免硬编码导致指向错误 exp 目录。
# 2. 使用严格模式 + nullglob，防止通配为空或错误字符串进入循环。
# 3. 跳过已存在且已正确指向的链接；使用 ln -sfn 避免链式/残留链接。
# 4. 清理 tb_runs 中的失效(断裂)符号链接，避免 TensorBoard 递归异常。
# 5. 防止出现自指/循环：比较 realpath 后再决定是否创建。
# 6. 允许通过环境变量 NAVSIM_EXP_ROOT 覆盖默认路径。

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$( cd -- "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
# 若外部未设置 NAVSIM_EXP_ROOT，则默认使用脚本所在目录上一层的 exp
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$ROOT_DIR/exp}"
TB_RUNS_DIR="$NAVSIM_EXP_ROOT/tb_runs"

mkdir -p "$TB_RUNS_DIR"

# 删除 tb_runs 中已经失效的符号链接（指向不存在的目标）
find "$TB_RUNS_DIR" -maxdepth 1 -xtype l -print -exec rm -f {} + 2>/dev/null || true

# 遍历 exp 下名称包含 agent 的一级目录
for d in "$NAVSIM_EXP_ROOT"/*agent*/ ; do
  [ -d "$d" ] || continue
  # 保险：如果 tb_runs 名字里恰巧含 agent（当前不含），跳过自身
  [ "$(basename "$d")" = "tb_runs" ] && continue

  bn="$(basename "$d")"
  link="$TB_RUNS_DIR/$bn"
  target_real="$(readlink -f "$d")"

  # 已存在且指向正确 -> 跳过
  if [ -L "$link" ]; then
    link_real="$(readlink -f "$link" || true)"
    if [ "$link_real" = "$target_real" ]; then
      continue
    fi
  fi

  # 如果目标 realpath 与 link realpath 完全一致且 link 是一个真实目录（不是 symlink），避免自指
  if [ -d "$link" ] && [ ! -L "$link" ]; then
    link_real="$(readlink -f "$link" || true)"
    if [ "$link_real" = "$target_real" ]; then
      echo "[跳过] 发现同名真实目录而非符号链接：$link (避免自指)" >&2
      continue
    fi
  fi

  ln -sfn "$d" "$link"
  echo "[链接] $link -> $d"

done

echo "TensorBoard 日志目录: $TB_RUNS_DIR"
exec tensorboard --logdir "$TB_RUNS_DIR" --host 0.0.0.0

###################################### 旧版本备份 ##############################
# NAVSIM_EXP_ROOT="$HOME/Driving/navsim_workspace/exp"
# mkdir -p $NAVSIM_EXP_ROOT/tb_runs
# for d in $NAVSIM_EXP_ROOT/*agent*/; do
#   ln -s "$d" "$NAVSIM_EXP_ROOT/tb_runs/$(basename "$d")"
# done
# tensorboard --logdir $NAVSIM_EXP_ROOT/tb_runs --host 0.0.0.0