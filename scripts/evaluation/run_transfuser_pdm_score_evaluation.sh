export HYDRA_FULL_ERROR=1

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/datasets/maps"
export NAVSIM_EXP_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp"
export NAVSIM_DEVKIT_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive"
export OPENSCENE_DATA_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/world4drive/datasets"

TRAIN_TEST_SPLIT=navtest
CHECKPOINT="'/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp/training_w4d_agent_3mode_with_cmd_embed/2026.01.13.17.42.33/lightning_logs/version_0/checkpoints/epoch=99-step=33300.ckpt'"
CACHE_PATH=/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
agent.checkpoint_path=$CHECKPOINT \
agent.config.model_version=3 \
agent.config.num_mode=3 \
agent.config.traj_cmd_loss_weight=0 \
experiment_name=pdms_w4d_32gpu_4bs \
metric_cache_path=$CACHE_PATH \
