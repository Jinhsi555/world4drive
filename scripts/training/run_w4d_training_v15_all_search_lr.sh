#!/bin/bash
# lr_search.sh - 学习率搜索脚本

# 环境变量（保持不变）
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/datasets/maps"
export NAVSIM_EXP_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp"
export NAVSIM_DEVKIT_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive"
export OPENSCENE_DATA_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/datasets"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_TIMEOUT=36000
export PYTHONPATH=/vepfs-mlp2/c20250502/haoce/wlb/world4drive/worldmirror:$PYTHONPATH

# 固定参数
config="all_navtrain_training"
TRAIN_TEST_SPLIT=navtrain
experiment_name="training_w4d_agent_4mode_navtrain_all_v15_dino_vision_query_lr_search_bs512"

# 学习率搜索列表
LEARNING_RATES=("2e-4" "5e-4" "8e-4" "1e-3")

# 训练脚本路径
SCRIPT=$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py

# 公共配置
BASE_CONFIG="agent=transfuser_agent \
dataloader.params.batch_size=16 \
dataloader.params.num_workers=12 \
train_test_split=$TRAIN_TEST_SPLIT \
use_cache_without_dataset=True \
force_cache_computation=False \
cache_path=$NAVSIM_EXP_ROOT/dino_feature_cache \
agent.config.model_version=15 \
agent.config.num_mode=4 \
agent.config.traj_cmd_loss_weight=0 \
agent.config.use_cmd_embed=False \
trainer.params.num_nodes=$MLP_WORKER_NUM \
trainer.params.devices=8 \
worker.threads_per_node=14"

# 遍历学习率
for lr in "${LEARNING_RATES[@]}"; do
    echo "========================================"
    echo "开始训练，学习率: $lr"
    echo "========================================"
    
    full_config="$BASE_CONFIG agent.lr=$lr experiment_name='$experiment_name/$lr'"
    
    torchrun \
        --nnodes=$MLP_WORKER_NUM \
        --nproc_per_node=8 \
        --node_rank=$MLP_ROLE_INDEX \
        --master_addr=$MLP_WORKER_0_HOST \
        --master_port=$MLP_WORKER_0_PORT \
        $SCRIPT \
        --config-name ${config} \
        $full_config
    
    echo "完成学习率: $lr"
    echo ""
done

echo "所有学习率训练完成！"
