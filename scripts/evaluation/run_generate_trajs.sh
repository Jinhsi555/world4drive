export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=9
export NAVSIM_DEVKIT_ROOT='/data/hdd01/xingzb/workspace/navsim-main'
TRAIN_TEST_SPLIT=navtest

# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_trajs.py \
# agent=transfuser_agent \
# agent.config.use_depth=True \
# dataloader.params.batch_size=64 \
# experiment_name=generate_trajs_depthembed \
# train_test_split=$TRAIN_TEST_SPLIT \
# use_cache_without_dataset=True \
# force_cache_computation=False \
# cache_path=/data/hdd01/xingzb/navsim_exp/testing_cache \
# agent.checkpoint_path='/data/hdd01/xingzb/navsim_exp/training_transfuser_agent_depthembed/2025.03.03.08.43.13/lightning_logs/version_0/checkpoints/epoch\=44-step\=5985.ckpt'

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_trajs.py \
agent=transfuser_agent \
agent.config.use_depth=True \
dataloader.params.batch_size=64 \
experiment_name=generate_trajs_depthembed \
train_test_split=$TRAIN_TEST_SPLIT \
use_cache_without_dataset=True \
force_cache_computation=False \
cache_path=/data/hdd01/xingzb/navsim_exp/testing_cache \
agent.checkpoint_path='/data/hdd01/xingzb/navsim_exp/training_transfuser_agent_depthembed/2025.03.03.08.43.13/lightning_logs/version_0/checkpoints/epoch\=49-step\=6650.ckpt'

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_trajs.py \
agent=transfuser_agent \
agent.config.use_depth=True \
dataloader.params.batch_size=64 \
experiment_name=generate_trajs_depthembed \
train_test_split=$TRAIN_TEST_SPLIT \
use_cache_without_dataset=True \
force_cache_computation=False \
cache_path=/data/hdd01/xingzb/navsim_exp/testing_cache \
agent.checkpoint_path='/data/hdd01/xingzb/navsim_exp/training_transfuser_agent_depthembed/2025.03.03.08.43.13/lightning_logs/version_0/checkpoints/epoch\=54-step\=7315.ckpt'

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_trajs.py \
agent=transfuser_agent \
agent.config.use_depth=True \
dataloader.params.batch_size=64 \
experiment_name=generate_trajs_depthembed \
train_test_split=$TRAIN_TEST_SPLIT \
use_cache_without_dataset=True \
force_cache_computation=False \
cache_path=/data/hdd01/xingzb/navsim_exp/testing_cache \
agent.checkpoint_path='/data/hdd01/xingzb/navsim_exp/training_transfuser_agent_depthembed/2025.03.03.08.43.13/lightning_logs/version_0/checkpoints/epoch\=64-step\=8645.ckpt'

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_trajs.py \
agent=transfuser_agent \
agent.config.use_depth=True \
dataloader.params.batch_size=64 \
experiment_name=generate_trajs_depthembed \
train_test_split=$TRAIN_TEST_SPLIT \
use_cache_without_dataset=True \
force_cache_computation=False \
cache_path=/data/hdd01/xingzb/navsim_exp/testing_cache \
agent.checkpoint_path='/data/hdd01/xingzb/navsim_exp/training_transfuser_agent_depthembed/2025.03.03.08.43.13/lightning_logs/version_0/checkpoints/epoch\=69-step\=9310.ckpt'

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_trajs.py \
agent=transfuser_agent \
agent.config.use_depth=True \
dataloader.params.batch_size=64 \
experiment_name=generate_trajs_depthembed \
train_test_split=$TRAIN_TEST_SPLIT \
use_cache_without_dataset=True \
force_cache_computation=False \
cache_path=/data/hdd01/xingzb/navsim_exp/testing_cache \
agent.checkpoint_path='/data/hdd01/xingzb/navsim_exp/training_transfuser_agent_depthembed/2025.03.03.08.43.13/lightning_logs/version_0/checkpoints/epoch\=74-step\=9975.ckpt'

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_trajs.py \
agent=transfuser_agent \
agent.config.use_depth=True \
dataloader.params.batch_size=64 \
experiment_name=generate_trajs_depthembed \
train_test_split=$TRAIN_TEST_SPLIT \
use_cache_without_dataset=True \
force_cache_computation=False \
cache_path=/data/hdd01/xingzb/navsim_exp/testing_cache \
agent.checkpoint_path='/data/hdd01/xingzb/navsim_exp/training_transfuser_agent_depthembed/2025.03.03.08.43.13/lightning_logs/version_0/checkpoints/epoch\=84-step\=11305.ckpt'

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_trajs.py \
agent=transfuser_agent \
agent.config.use_depth=True \
dataloader.params.batch_size=64 \
experiment_name=generate_trajs_depthembed \
train_test_split=$TRAIN_TEST_SPLIT \
use_cache_without_dataset=True \
force_cache_computation=False \
cache_path=/data/hdd01/xingzb/navsim_exp/testing_cache \
agent.checkpoint_path='/data/hdd01/xingzb/navsim_exp/training_transfuser_agent_depthembed/2025.03.03.08.43.13/lightning_logs/version_0/checkpoints/epoch\=89-step\=11970.ckpt'

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_trajs.py \
agent=transfuser_agent \
agent.config.use_depth=True \
dataloader.params.batch_size=64 \
experiment_name=generate_trajs_depthembed \
train_test_split=$TRAIN_TEST_SPLIT \
use_cache_without_dataset=True \
force_cache_computation=False \
cache_path=/data/hdd01/xingzb/navsim_exp/testing_cache \
agent.checkpoint_path='/data/hdd01/xingzb/navsim_exp/training_transfuser_agent_depthembed/2025.03.03.08.43.13/lightning_logs/version_0/checkpoints/epoch\=94-step\=12635.ckpt'