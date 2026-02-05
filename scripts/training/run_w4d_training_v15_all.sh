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
experiment_name=training_w4d_agent_4mode_navtrain_all_dino_lora/256_50_epoch_lr_5e-4

torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --nproc_per_node=8 \
    --node_rank=$MLP_ROLE_INDEX \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    --config-name ${config} \
    agent=transfuser_agent \
    dataloader.params.batch_size=8 \
    dataloader.params.num_workers=12 \
    experiment_name=$experiment_name \
    train_test_split=$TRAIN_TEST_SPLIT \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    cache_path=$NAVSIM_EXP_ROOT/feature_cache_navtrain \
    agent.config.model_version='dino_lora' \
    agent.config.num_mode=4 \
    agent.config.traj_cmd_loss_weight=0 \
    agent.config.use_cmd_embed=False \
    agent.config.use_wm=True \
    agent.config.num_frames=4 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.wm_loss_weight=0.5 \
    agent.lr=5e-4 \
    trainer.params.num_nodes=$MLP_WORKER_NUM \
    trainer.params.devices=8 \
    worker.threads_per_node=14 \
    trainer.params.precision='bf16-mixed' \
    trainer.params.accumulate_grad_batches=1 \
    trainer.params.max_epochs=30 \
    # resume_ckpt_path="'/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/training_w4d_agent_3mode_all_navtrain/2025.10.16.08.35.16/lightning_logs/version_0/checkpoints/epoch=29-step=12120.ckpt'"
