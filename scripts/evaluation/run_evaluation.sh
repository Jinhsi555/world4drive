############################### original eval #####################################
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OPENBLAS_CORETYPE=Haswell
export HYDRA_FULL_ERROR=1

export NAVSIM_DEVKIT_ROOT="$HOME/Driving/navsim_workspace/world4drive"
export NAVSIM_EXP_ROOT="$HOME/Driving/navsim_workspace/exp"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/Driving/navsim_workspace/dataset/maps"
export OPENSCENE_DATA_ROOT="$HOME/Driving/navsim_workspace/dataset"

SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles
split=navhard
agent=transfuser_agent
dir=training_w4d_agent
metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache_${split}_two_stage"
cd ${NAVSIM_DEVKIT_ROOT}
ckpt_dir=/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/training_w4d_agent/2025.09.26.22.30.31/lightning_logs/version_0/checkpoints
## 重命名ckpt文件
cd ${ckpt_dir}
for file in epoch=*-step=*.ckpt; do
    [ -e "$file" ] || { echo "[INFO] 未找到需要重命名的原始 ckpt 文件"; break; }
    # 提取 epoch= 后的完整数字（单/多位均可）
    epoch=$(echo "$file" | sed -E 's/.*epoch=([0-9]+)-step=.*/\1/')
    if [ -z "$epoch" ]; then
        echo "[WARN] 无法解析 epoch (跳过): $file"
        continue
    fi
    new_filename="epoch${epoch}.ckpt"
    if [ -e "$new_filename" ]; then
        # 若已存在同名文件则跳过，避免覆盖
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

################################ 循环评测所有ckpt #####################################
for ckpt in $(ls ${ckpt_dir}/epoch*.ckpt | sort -V -r); do
    # 从文件名中提取epoch数字
    epoch=$(basename $ckpt | grep -o 'epoch[0-9]\+' | grep -o '[0-9]\+')
    padded_epoch=$(printf "%02d" $epoch)
    
    experiment_name="${dir}/test-${padded_epoch}ep-${split}"
    
    export SUBSCORE_PATH=${NAVSIM_EXP_ROOT}/${dir}/epoch${epoch}_${split}.pkl; # save path for the scores

    echo "评测 checkpoint: $ckpt (epoch $epoch)"
    
    python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2.py \
        agent=$agent \
        agent.checkpoint_path=${ckpt} \
        trainer.params.precision=32 \
        experiment_name=${experiment_name} \
        +cache_path=null \
        metric_cache_path=${metric_cache_path} \
        train_test_split=${split}_two_stage \
        synthetic_sensor_path=${SYNTHETIC_SENSOR_PATH} \
        synthetic_scenes_path=${SYNTHETIC_SCENES_PATH} 
done


############################### 单次评测 #####################################
# ckpt=
# experiment_name=
# python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2.py \
#         agent=$agent \
#         agent.checkpoint_path=${ckpt} \
#         trainer.params.precision=32 \
#         experiment_name=${experiment_name} \
#         +cache_path=null \
#         metric_cache_path=${metric_cache_path} \
#         train_test_split=${split}_two_stage \
#         synthetic_sensor_path=${SYNTHETIC_SENSOR_PATH} \
#         synthetic_scenes_path=${SYNTHETIC_SCENES_PATH} 