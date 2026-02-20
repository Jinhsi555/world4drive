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

config="all_navtrain_training" # this config uses the entire navtrain dataset for training
TRAIN_TEST_SPLIT=navtrain
experiment_name=training_w4d_agent_4mode_navtrain_all_dino_geometry_only/256_50_epoch_lr3e_4_wm_post_training_dino_small

# 分阶段训练调度说明 (wm_loss_weight=0.6, geometry_loss_weight=0.6):
# Epoch 0~19:   WM=0,     Geo=0.6  → 阶段1: 纯 geometry + traj（20 个 epoch 充分蒸馏几何）
# Epoch 20~49:  WM 0→0.6, Geo 0.6→0.06 → 阶段2: WM 逐步引入，geometry 同步衰减
# Epoch 50~100: WM=0.6,   Geo=0.06 → 阶段3: WM 主导，geometry 仅正则化

torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --nproc_per_node=8 \
    --node_rank=$MLP_ROLE_INDEX \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_post_training.py \
    --config-name ${config} \
    agent=transfuser_agent \
    dataloader.params.batch_size=8 \
    dataloader.params.num_workers=12 \
    experiment_name=$experiment_name \
    train_test_split=$TRAIN_TEST_SPLIT \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    cache_path=$NAVSIM_EXP_ROOT/feature_cache_navtrain \
    agent.checkpoint_path="'/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp/training_w4d_agent_4mode_navtrain_all_dino_geometry_only/256_50_epoch_lr3e_4_geometry_only_dino_small/2026.02.19.10.37.10/lightning_logs/version_0/checkpoints/epoch=49-step=20200.ckpt'" \
    agent.config.model_version='dino_small_wm_post_training' \
    agent.config.num_mode=4 \
    agent.config.traj_cmd_loss_weight=0 \
    agent.config.use_cmd_embed=False \
    agent.config.use_wm=True \
    agent.config.num_frames=4 \
    agent.config.num_scene_query_token=16 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.wm_loss_weight=0.6 \
    agent.config.geometry_loss_weight=0.0 \
    agent.config.use_staged_loss=False \
    agent.config.curriculum_start_epoch=100 \
    agent.config.curriculum_end_epoch=100 \
    agent.lr=3e-4 \
    trainer.params.num_nodes=$MLP_WORKER_NUM \
    trainer.params.devices=8 \
    worker.threads_per_node=14 \
    trainer.params.precision='bf16-mixed' \
    trainer.params.accumulate_grad_batches=1 \
    trainer.params.max_epochs=50 \
