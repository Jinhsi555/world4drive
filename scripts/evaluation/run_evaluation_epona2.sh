export SUBSCORE_PATH="/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/Epona/exp/test_traj_results/360000iter_only_train_traj_10sample.pkl"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OPENBLAS_CORETYPE=Haswell
export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT="/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive"
split=navtest
metric_cache_path='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/metric_cache'
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles
ckpt_path='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/training_w4d_agent_4mode_all_navtrain_ar_4chunk_wm/2025.10.31.22.47.36/lightning_logs/version_0/checkpoints/epoch79.ckpt'
experiment_name='eval/test-w4d-navtest_debug'


python /mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/navsim/planning/script/run_pdm_score_one_stage_gpu_epona.py \
agent=transfuser_agent \
agent.checkpoint_path=${ckpt_path} \
trainer.params.precision=32 \
experiment_name=${experiment_name} \
+cache_path=null \
metric_cache_path=${metric_cache_path} \
train_test_split=${split} \
agent.config.use_wm=False \
agent.config.model_version=12 \
traffic_agents=non_reactive \
agent.config.num_mode=4 \
agent.config.use_ar=True \
agent.config.use_ar_wm=True \
+closed_traj_evaluation=True
