export HYDRA_FULL_ERROR=1

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/datasets/maps"
export NAVSIM_EXP_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp"
export NAVSIM_DEVKIT_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive"
export OPENSCENE_DATA_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/datasets"
export SUBSCORE_PATH="/notexist"

TRAIN_TEST_SPLIT=navtest
CHECKPOINT="'/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp/training_w4d_agent_3mode_navtrain_v15_ypx/2026.01.17.06.20.10/lightning_logs/version_0/checkpoints/epoch=94-step=31635.ckpt'"
CACHE_PATH=/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp/metric_cache

export CUDA_VISIBLE_DEVICES=0

python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage_gpu.py \
    agent=transfuser_agent \
    agent.checkpoint_path=$CHECKPOINT \
    trainer.params.precision=32 \
    trainer.params.accelerator=gpu \
    +trainer.params.devices=8 \
    trainer.params.num_nodes=1 \
    experiment_name=pdms_w4d_32gpu_4bs \
    metric_cache_path=$CACHE_PATH \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent.config.model_version=15 \
    agent.config.num_mode=4 \
    agent.config.use_wm=False \
    agent.config.use_cmd_embed=False \
    traffic_agents=non_reactive
