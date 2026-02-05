export HYDRA_FULL_ERROR=1
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/datasets/maps"
export NAVSIM_EXP_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp"
export NAVSIM_DEVKIT_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive"
export OPENSCENE_DATA_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/datasets"
TRAIN_TEST_SPLIT=navtest

export PYTHONPATH=/vepfs-mlp2/c20250502/haoce/wlb/world4drive/worldmirror:$PYTHONPATH
torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --nproc_per_node=8 \
    --node_rank=$MLP_ROLE_INDEX \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_multi_node.py \
    agent=transfuser_agent \
    agent.config.model_version='refine_temporal_short_step' \
    agent.config.num_mode=4 \
    agent.config.use_cmd_embed=False \
    agent.config.cache_mode=True \
    experiment_name=transfuser_cache_for_training_debug \
    cache_path=$NAVSIM_EXP_ROOT/camera_path_feature_cache_navtest \
    train_test_split=$TRAIN_TEST_SPLIT \
    # worker=sequential \