export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OPENBLAS_CORETYPE=Haswell
export HYDRA_FULL_ERROR=1

export NAVSIM_DEVKIT_ROOT="$HOME/Driving/navsim_workspace/world4drive"
export NAVSIM_EXP_ROOT="$HOME/Driving/navsim_workspace/exp"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/Driving/navsim_workspace/dataset/maps"
export OPENSCENE_DATA_ROOT="$HOME/Driving/navsim_workspace/dataset"


TRAIN_TEST_SPLIT=navtrain
experiment_name=training_w4d_agent_rl_lr1e-5_2adv_high_ep_5bc_2entropy_2diversity_bs32_ep89


python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    --config-name default_training_rl \
    agent=transfuser_agent_rl \
    dataloader.params.batch_size=32 \
    experiment_name=$experiment_name \
    train_test_split=$TRAIN_TEST_SPLIT \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    cache_path=$NAVSIM_EXP_ROOT/cache_for_training \
    agent.config.training_mode=ft \
    agent.checkpoint_path=/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/training_w4d_agent_diff_cmd/2025.10.02.23.00.37/lightning_logs/version_0/checkpoints/epoch89.ckpt \


