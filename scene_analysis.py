import os
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # 从 matplotlib.colors 导入 LogNorm
from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from pathlib import Path
from pyquaternion import Quaternion
import numpy.typing as npt
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from collections import defaultdict
from sklearn.cluster import KMeans
import gzip
import torch
from typing import Dict
def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict

def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict

SYNTHETIC_SENSOR_PATH='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/navhard_two_stage/sensor_blobs'
SYNTHETIC_SCENES_PATH='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/navhard_two_stage/synthetic_scene_pickles'
NAVSIM_LOG_PATH='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/meta_datas/trainval'
ORIGINAL_SENSOR_PATH='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/sensor_blobs'

NAVSIM_LOG_PATH_NAVHARD='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/meta_datas/test'

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

FILTER = "navhard_two_stage" # navtrain
# metric_cache_path = '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/metric_cache_training'
metric_cache_path = '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/metric_cache_navhard_two_stage'


hydra.initialize(config_path="./navsim/planning/script/config/common/train_test_split")
scene_filter_cfg = hydra.compose(config_name=FILTER).scene_filter
scene_filter: SceneFilter = instantiate(scene_filter_cfg)
# scene_filter_val: SceneFilter = instantiate(hydra.compose(config_name='navtest'))
# openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))


## scene_loader
# scene_loader = SceneLoader(
#         synthetic_sensor_path=Path(SYNTHETIC_SENSOR_PATH),
#         original_sensor_path=Path(ORIGINAL_SENSOR_PATH),
#         data_path=Path(NAVSIM_LOG_PATH),
#         synthetic_scenes_path=Path(SYNTHETIC_SCENES_PATH),
#         scene_filter=scene_filter,
#     )

scene_loader = SceneLoader(
        synthetic_sensor_path=Path(SYNTHETIC_SENSOR_PATH),
        original_sensor_path=Path(ORIGINAL_SENSOR_PATH),
        data_path=Path(NAVSIM_LOG_PATH_NAVHARD),
        synthetic_scenes_path=Path(SYNTHETIC_SCENES_PATH),
        scene_filter=scene_filter,
    )

scene_loader_tokens = scene_loader.tokens    # list

## metric_cache_loader当中没有控制命令
metric_cache_loader = MetricCacheLoader(Path(metric_cache_path))

metric_cache_loader_tokens = metric_cache_loader.tokens # list

tokens = list(set(scene_loader_tokens) & set(metric_cache_loader_tokens))
scene_frames_dicts = scene_loader.scene_frames_dicts

print(f'tokens_to_use: {len(tokens)}')
n_clusters = 6
human_trajs_by_cmd = defaultdict(list)
cur_frame_id = scene_filter.num_history_frames - 1  # 3
training_cache_path = Path('/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/cache_for_training')
navhard_pkl_path = Path('/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/dataset/navhard_two_stage/openscene_meta_datas')
for token in tqdm(tokens):
    # cur_frame_info = scene_frames_dicts[token][cur_frame_id]
    
    ### navhard
    navhard_pkl_files = sorted(navhard_pkl_path.glob("*.pkl"))
    for info_path in tqdm(navhard_pkl_files):
        scene_info = pickle.load(open(info_path, 'rb'))
        cur_frame_info = scene_info[3]
        driving_command = cur_frame_info['driving_command'] # np(4,) int
        cur_human_traj = np.array(1)
        cmd_id = int(np.argmax(driving_command))
        human_trajs_by_cmd[cmd_id].append(cur_human_traj)


    
    info_path = navhard_pkl_path / f'{token}.pkl'
    navhard_info = pickle.load(open(info_path, 'rb'))
    cur_frame_info = navhard_info[3]['driving_command']
    


    driving_command = cur_frame_info['driving_command'] # np(4,) int
    log_name = cur_frame_info['log_name']
    # feature_path = training_cache_path / log_name / token / 'transfuser_feature.gz'
    target_path = training_cache_path / log_name / token / 'transfuser_target.gz'
    # feature_dict = load_feature_target_from_pickle(feature_path)
    target_data_dict = load_feature_target_from_pickle(target_path)
    cur_human_traj = target_data_dict['trajectory']  # torch(8,3)
    cmd_id = int(np.argmax(driving_command))
    human_trajs_by_cmd[cmd_id].append(cur_human_traj.numpy())

    # cur_metric_cache = metric_cache_loader.get_from_token(token)
    # driving_command = cur_frame_info.ego_status.driving_command # np(4,) int
    # cur_human_traj = cur_metric_cache.human_trajectory.poses # np(8,3)
    # cmd_id = int(np.argmax(driving_command))
    # human_trajs_by_cmd[cmd_id].append(cur_human_traj)

cluster_centers = []
for cmd_id in sorted(human_trajs_by_cmd.keys()):
    trajs = np.stack(human_trajs_by_cmd[cmd_id], axis=0)
    num_trajs = trajs.shape[0]
    print(f'Command {cmd_id}: collected {num_trajs} trajectories.')
    if num_trajs < n_clusters:
        raise ValueError(f'Not enough trajectories for command {cmd_id} to form {n_clusters} clusters.')
    features = trajs.reshape(num_trajs, -1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(features)
    centers = kmeans.cluster_centers_.reshape(n_clusters, trajs.shape[1], trajs.shape[2])
    cluster_centers.append(centers)

output_path = Path('/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/kmeans_navsim_four_cmd_traj_6.npy')
output_path.parent.mkdir(parents=True, exist_ok=True)
np.save(output_path, np.stack(cluster_centers, axis=0))
print(f'Saved cluster centers to {output_path}')


## 可视化分析
traj_np_path = '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/kmeans_navsim_four_cmd_traj_6.npy'
visualize_traj(traj_np_path)
