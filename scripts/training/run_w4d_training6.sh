export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OPENBLAS_CORETYPE=Haswell
export HYDRA_FULL_ERROR=1

export NAVSIM_DEVKIT_ROOT="$HOME/Driving/navsim_workspace/world4drive"
export NAVSIM_EXP_ROOT="$HOME/Driving/navsim_workspace/exp"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/Driving/navsim_workspace/dataset/maps"
export OPENSCENE_DATA_ROOT="$HOME/Driving/navsim_workspace/dataset"

config="all_navtrain_training" # this config uses the entire navtrain dataset for training
TRAIN_TEST_SPLIT=navtrain
experiment_name=training_w4d_agent_4mode_all_navtrain_ar_4chunk_wm_temp_ensemble


python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    --config-name ${config} \
    agent=transfuser_agent \
    dataloader.params.batch_size=32 \
    experiment_name=$experiment_name \
    train_test_split=$TRAIN_TEST_SPLIT \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    cache_path=$NAVSIM_EXP_ROOT/all_info_training_cache \
    agent.config.model_version=12 \
    agent.config.use_ar=True \
    agent.config.use_ar_wm=True \
    agent.config.num_mode=4 \
    # resume_ckpt_path="'/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/training_w4d_agent_4mode_all_navtrain_ar_4chunk_wm/2025.10.31.22.47.36/lightning_logs/version_0/checkpoints/epoch79.ckpt'"
