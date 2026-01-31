############################### original eval #####################################
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/datasets/maps"
export NAVSIM_EXP_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp"
export NAVSIM_DEVKIT_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive"
export OPENSCENE_DATA_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/datasets"

export PYTHONPATH=/vepfs-mlp2/c20250502/haoce/wlb/world4drive/worldmirror:$PYTHONPATH

SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles
split=navtest
agent=transfuser_agent
dir=training_w4d_agent_4mode_navtrain_all_v15_dino_vision_query/3_views_refine_temporal_wm_bs1024/navtest_eval_one_stage_train_all
metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache"
cd ${NAVSIM_DEVKIT_ROOT}
ckpt_dir=/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp/training_w4d_agent_4mode_navtrain_all_v15_dino_vision_query/3_views_refine_temporal_wm_bs1024/2026.01.31.07.43.22/lightning_logs/version_0/checkpoints

# 并行相关配置 (可通过环境变量覆盖)
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}   # 逗号分隔 GPU id 列表
MAX_PROCS=${MAX_PROCS:-8}                 # 期望最大并发任务数（含等待调度）
SKIP_EXISTING=${SKIP_EXISTING:-1}         # =1 如果已经有对应 pkl 则跳过
LOG_DIR_SUFFIX=${LOG_DIR_SUFFIX:-eval_logs_${split}}

# 解析 GPU 列表
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
GPU_COUNT=${#GPU_ARRAY[@]}
if [ "$MAX_PROCS" -gt 0 ] && [ "$MAX_PROCS" -lt "$GPU_COUNT" ]; then
  CONCURRENCY=$MAX_PROCS
else
  CONCURRENCY=$GPU_COUNT
fi

# 若用户设置 MAX_PROCS 大于 GPU 数，仍然限制为 GPU 数，避免同 GPU 多任务争抢
PROCS_LIMIT=$CONCURRENCY

echo "使用 GPU: ${GPU_ARRAY[*]} (最大并发: $PROCS_LIMIT)"

# 日志目录
LOG_DIR=${NAVSIM_EXP_ROOT}/${dir}/${LOG_DIR_SUFFIX}
mkdir -p "$LOG_DIR"

echo "日志输出目录: $LOG_DIR"

## 重命名ckpt文件 (只在需要时执行)
cd ${ckpt_dir}
for file in epoch=*-step=*.ckpt; do
    [ -e "$file" ] || { echo "[INFO] 未找到需要重命名的原始 ckpt 文件"; break; }
    epoch=$(echo "$file" | sed -E 's/.*epoch=([0-9]+)-step=.*/\1/')
    if [ -z "$epoch" ]; then
        echo "[WARN] 无法解析 epoch (跳过): $file"
        continue
    fi
    new_filename="epoch${epoch}.ckpt"
    if [ -e "$new_filename" ]; then
        if [ "$(readlink -f "$file")" = "$(readlink -f "$new_filename")" ]; then
            echo "[SKIP] 已重命名: $file -> $new_filename"
        else
            echo "[SKIP] 目标已存在，避免覆盖: $new_filename (源: $file)"
        fi
        continue
    fi
    mv "$file" "$new_filename"
    echo "[OK] $file -> $new_filename"
done

################################ 动态并行评测所有ckpt #####################################
cd ${NAVSIM_DEVKIT_ROOT}

CKPTS=( $(ls ${ckpt_dir}/epoch*.ckpt 2>/dev/null | sort -V -r) )
if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "未找到 checkpoint: ${ckpt_dir}/epoch*.ckpt"
  exit 1
fi

echo "共找到 ${#CKPTS[@]} 个 ckpt"

# 使用关联数组跟踪 GPU->PID (需要 bash 支持关联数组)
# 若 /bin/bash 不是 4+ 可能会出问题；此项目环境一般是 4+/5+。
# 动态调度策略：
#   1. 始终保持每块 GPU 最多 1 个进程
#   2. 有空闲 GPU 且仍有待评测 ckpt 时立即启动新任务
#   3. 利用 wait -n (若可用) 实时回收；否则轮询退化

declare -A GPU_PIDS

# 检测是否支持 wait -n
support_waitn=0
if wait -n 2>/dev/null; then
  support_waitn=1
fi
# 上面会立即返回（因为没有子进程），可能给出非零状态；我们换个更稳当的检测方式
# 重新判定：
if bash -c 'help wait' 2>/dev/null | grep -q -- "-n"; then
  support_waitn=1
fi

prune_finished() {
  for g in "${!GPU_PIDS[@]}"; do
    pid=${GPU_PIDS[$g]}
    if ! kill -0 "$pid" 2>/dev/null; then
      unset GPU_PIDS[$g]
      echo "[回收] GPU $g 上的进程 $pid 已结束"
    fi
  done
}

launch_eval() {
  local ckpt=$1
  local epoch=$2
  local gpu=$3
  local padded_epoch=$(printf "%02d" "$epoch")
  local experiment_name="${dir}/test-${padded_epoch}ep-${split}"
  local subscore_path=${NAVSIM_EXP_ROOT}/${dir}/epoch${epoch}_${split}.pkl
  local log_file=${LOG_DIR}/epoch${epoch}.log

  if [ "$SKIP_EXISTING" = "1" ]; then
    local experiment_dir="${NAVSIM_EXP_ROOT}/${dir}/test-${padded_epoch}ep-${split}"
    local latest_run_dir=""
    local csv_file=""
    if [ -d "$experiment_dir" ]; then
      latest_run_dir=$(find "$experiment_dir" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort | tail -n 1)
      if [ -n "$latest_run_dir" ]; then
        csv_file=$(find "$latest_run_dir" -maxdepth 1 -type f -name '*.csv' -print -quit 2>/dev/null)
        if [ -n "$csv_file" ]; then
          echo "[跳过] epoch ${epoch} 最新目录 $(basename "$latest_run_dir") 已含 .csv 结果: $(basename "$csv_file")"
          return 1
        fi
      fi
    fi
  fi

  echo "[启动] epoch ${epoch} -> GPU ${gpu}; log: $(basename "$log_file")"
  (
      export CUDA_VISIBLE_DEVICES=$gpu
      export SUBSCORE_PATH=$subscore_path
      echo "==== 开始评测 epoch ${epoch} (GPU ${gpu}) 时间: $(date '+%F %T') ==== "
      python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage_gpu.py \
          agent=$agent \
          agent.checkpoint_path=${ckpt} \
          trainer.params.precision='32' \
          trainer.params.accelerator=gpu \
          +trainer.params.devices=1 \
          trainer.params.num_nodes=1 \
          experiment_name=${experiment_name} \
          cache_path="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp/dino_feature_cache_test" \
          metric_cache_path=${metric_cache_path} \
          train_test_split=${split} \
          agent.config.model_version='refine_temporal_short_step' \
          agent.config.num_mode=4 \
          agent.config.num_frames=4 \
          agent.config.use_wm=True \
          +agent.config.avg_mode=True \
          agent.config.use_cmd_embed=False \
          traffic_agents=non_reactive \
          worker.threads_per_node=14 \
          dataloader.params.batch_size=32 \


      status=$?
      echo "==== 结束评测 epoch ${epoch} (GPU ${gpu}) 退出码: $status 时间: $(date '+%F %T') ==== "
      exit $status
  ) > "$log_file" 2>&1 &
  local pid=$!
  GPU_PIDS[$gpu]=$pid
  return 0
}

# 主调度循环
for ckpt in "${CKPTS[@]}"; do
  epoch=$(basename "$ckpt" | grep -o 'epoch[0-9]\+' | grep -o '[0-9]\+')

  while :; do
    prune_finished

    # 如果当前运行数 < 限制，则尝试找空闲 GPU
    if [ ${#GPU_PIDS[@]} -lt $PROCS_LIMIT ]; then
      free_gpu=""
      for g in "${GPU_ARRAY[@]}"; do
        if [ -z "${GPU_PIDS[$g]}" ]; then
          free_gpu=$g
          break
        fi
      done
      if [ -n "$free_gpu" ]; then
        launch_eval "$ckpt" "$epoch" "$free_gpu"
        break  # 进入处理下一个 ckpt
      fi
    fi

    # 没有空闲 GPU 或达到并发上限 -> 等待任一任务结束
    if [ $support_waitn -eq 1 ]; then
      wait -n 2>/dev/null || true
    else
      # 退化：等待最早一个 PID 完成（简单轮询）
      sleep 2
    fi
  done

done

# 等所有剩余任务完成
if [ ${#GPU_PIDS[@]} -gt 0 ]; then
  echo "等待剩余 ${#GPU_PIDS[@]} 个任务..."
  for g in "${!GPU_PIDS[@]}"; do
    pid=${GPU_PIDS[$g]}
    if kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
    fi
  done
fi

echo "全部评测完成。日志目录: $LOG_DIR"