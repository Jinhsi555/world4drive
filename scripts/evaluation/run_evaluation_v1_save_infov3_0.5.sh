export SUBSCORE_PATH="/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/Epona/exp/test_traj_results/460000iter_traj.pkl"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OPENBLAS_CORETYPE=Haswell
export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT="/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive"

metric_cache_path='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/metric_cache_training'
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles
ckpt_path='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/training_w4d_agent_4mode_all_navtrain_ar_4chunk_wm/2025.10.31.22.47.36/lightning_logs/version_0/checkpoints/epoch79.ckpt'
experiment_name='eval/test-w4d-navtrain_save_info'

traj_vocab_path="/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/Epona/exp/traj_vocab_8192_8_3.npy"
navtrain_score_path="/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/Epona/exp/exp"
# 运行Python脚本
python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_v1_voc_save_infov5.py \
    experiment_name=${experiment_name} \
    +cache_path=null \
    metric_cache_path=${metric_cache_path} \
    train_test_split=navtrain \
    +traj_vocab_path=${traj_vocab_path} \
    +navtrain_score_path=${navtrain_score_path} \
    +time_horizon=0.5