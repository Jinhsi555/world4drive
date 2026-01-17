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

TRAIN_TEST_SPLIT=navtrain
experiment_name=training_w4d_agent_3mode_with_cmd_embed

torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --nproc_per_node=8 \
    --node_rank=$MLP_ROLE_INDEX \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=transfuser_agent \
    dataloader.params.batch_size=32 \
    experiment_name=$experiment_name \
    train_test_split=$TRAIN_TEST_SPLIT \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    cache_path=$NAVSIM_EXP_ROOT/all_info_training_cache \
    agent.config.model_version=3 \
    agent.config.num_mode=3 \
    agent.config.use_cmd_embed=True \
    agent.config.traj_cmd_loss_weight=0 \
    trainer.params.num_nodes=$MLP_WORKER_NUM \
    trainer.params.devices=8
    # resume_ckpt_path="'/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/training_w4d_agent_3mode_cmdpred/2025.10.07.19.47.18/lightning_logs/version_0/checkpoints/epoch=49-step=16650.ckpt'"
