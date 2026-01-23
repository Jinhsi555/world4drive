export HYDRA_FULL_ERROR=1
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/datasets/maps"
export NAVSIM_EXP_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp"
export NAVSIM_DEVKIT_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive"
export OPENSCENE_DATA_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/datasets"
TRAIN_TEST_SPLIT=navtrain

export PYTHONPATH=/vepfs-mlp2/c20250502/haoce/wlb/world4drive/worldmirror:$PYTHONPATH

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    agent=transfuser_agent \
    agent.config.model_version=15 \
    agent.config.num_mode=4 \
    agent.config.use_cmd_embed=False \
    agent.config.cache_mode=True \
    experiment_name=transfuser_cache_for_training \
    cache_path=$NAVSIM_EXP_ROOT/dino_geometry_cache_trainval \
    train_test_split=$TRAIN_TEST_SPLIT \
    worker.threads_per_node=112
    # worker=sequential \