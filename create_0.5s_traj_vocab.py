import os
import pickle
from tqdm import tqdm
import math
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
def compute_relative_traj(abs_traj):
    """
    针对 heading 已归一化到[-π, π]且从0起始的优化版相对轨迹计算
    核心：x/y 直接差分，heading 计算最小角度差（确保 dheading ∈ [-π, π]）
    """
    N = abs_traj.shape[0]
    
    # 1. x, y 相对变化（线性量，直接差分）
    dx = torch.diff(abs_traj[:, 0], dim=0)  # (N-1,)：x方向相对上一步变化
    dy = torch.diff(abs_traj[:, 1], dim=0)  # (N-1,)：y方向相对上一步变化
    
    # 2. heading 相对变化（处理[-π, π]周期性，计算最短路径转角）
    heading_prev = abs_traj[:-1, 2]  # 前一步航向角（N-1,）
    heading_curr = abs_traj[1:, 2]   # 当前步航向角（N-1,）
    dheading_raw = heading_curr - heading_prev  # 原始差分
    
    # 确保 dheading ∈ [-π, π]（处理跨 -π/π 边界的情况，如 π → -π 实际转角为 0）
    dheading = ((dheading_raw + math.pi) % (2 * math.pi)) - math.pi
    
    # 3. 拼接相对轨迹（第一个动作相对于初始原点(0,0,0)，符合heading从0起始）
    rel_traj_first = abs_traj[0:1, :]  # 第一个动作：(x0, y0, heading0)（heading0=0）
    rel_traj_rest = torch.stack([dx, dy, dheading], dim=1)  # 后续动作：(dx, dy, dheading)
    rel_traj = torch.cat([rel_traj_first, rel_traj_rest], dim=0)  # (8,3)：8个相对动作
    
    return rel_traj

SYNTHETIC_SENSOR_PATH='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/navhard_two_stage/sensor_blobs'
SYNTHETIC_SCENES_PATH='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/navhard_two_stage/synthetic_scene_pickles'
NAVSIM_LOG_PATH='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/meta_datas/trainval'
ORIGINAL_SENSOR_PATH='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/sensor_blobs'

NAVSIM_LOG_PATH_NAVHARD='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/meta_datas/test'

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

FILTER = "navtrain" # navtrain
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
        # synthetic_sensor_path=Path(SYNTHETIC_SENSOR_PATH),
        original_sensor_path=Path(ORIGINAL_SENSOR_PATH),
        data_path=Path(NAVSIM_LOG_PATH),
        # synthetic_scenes_path=Path(SYNTHETIC_SCENES_PATH),
        scene_filter=scene_filter,
    )

scene_loader_tokens = scene_loader.tokens    # list

## metric_cache_loader当中没有控制命令
# metric_cache_loader = MetricCacheLoader(Path(metric_cache_path))

# metric_cache_loader_tokens = metric_cache_loader.tokens # list

# tokens = list(set(scene_loader_tokens) & set(metric_cache_loader_tokens))
tokens = list(set(scene_loader_tokens))
scene_frames_dicts = scene_loader.scene_frames_dicts

print(f'tokens_to_use: {len(tokens)}')
n_clusters = 2048
all_step_trajs = []
cur_frame_id = scene_filter.num_history_frames - 1  # 3
training_cache_path = Path('/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/cache_for_training')
navhard_pkl_path = Path('/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/dataset/navhard_two_stage/openscene_meta_datas')
for token in tqdm(tokens):
    # info_path = navhard_pkl_path / f'{token}.pkl'
    # if not info_path.exists():
        # continue
    info_dict = scene_frames_dicts[token]
    # with open(info_path, 'rb') as f:
        # navhard_info = pickle.load(f)
    cur_frame_info = info_dict[cur_frame_id]
    log_name = cur_frame_info['log_name']
    target_path = training_cache_path / log_name / token / 'transfuser_target.gz'
    if not target_path.exists():
        continue
    target_data_dict = load_feature_target_from_pickle(target_path)
    full_8_step_traj = target_data_dict['trajectory']
    diff_8_step_traj = compute_relative_traj(full_8_step_traj)
    # 将8个步的相对轨迹拆分成单步轨迹
    # for step_idx in range(diff_8_step_traj.shape[0]):
    #     step_traj = diff_8_step_traj[step_idx:step_idx+1, :].numpy()  # (1, 3)
    #     all_step_trajs.append(step_traj)
    # step_traj = target_data_dict['trajectory']
    # if isinstance(step_traj, torch.Tensor):
    #     step_traj = step_traj.detach().cpu().numpy()
    # else:
    #     step_traj = np.asarray(step_traj)
    # all_step_trajs.extend(step_traj)
    all_step_trajs.append(diff_8_step_traj.numpy())
if not all_step_trajs:
    raise ValueError("未收集到任何单步轨迹用于聚类。")
step_trajs = np.concatenate(all_step_trajs, axis=0)
# 保存当前的单步轨迹聚类结果，以便后续加载
np.save('/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/navsim_rel_single_step_traj_2048.npy', step_trajs)
print(f'Collected {step_trajs.shape[0]} single-step trajectories for clustering.')
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(step_trajs)
cluster_centers = kmeans.cluster_centers_
output_path = Path('/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/kmeans_navsim_rel_single_step_traj_2048.npy')
output_path.parent.mkdir(parents=True, exist_ok=True)
np.save(output_path, cluster_centers)
print(f'Saved cluster centers to {output_path}')

## 可视化分析
cluster_fig_path = output_path.with_suffix(".png")
cluster_fig_path.parent.mkdir(parents=True, exist_ok=True)
centers = cluster_centers

fig, axes = plt.subplots(1, 2, figsize=(15, 4))
axes[0].quiver(
    np.zeros_like(centers[:, 0]),
    np.zeros_like(centers[:, 1]),
    centers[:, 0],
    centers[:, 1],
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.002,
    alpha=0.6,
)
axes[0].set_xlim(centers[:, 0].min() * 1.1, centers[:, 0].max() * 1.1)
axes[0].set_ylim(centers[:, 1].min() * 1.1, centers[:, 1].max() * 1.1)
axes[0].set_xlabel("dx")
axes[0].set_ylabel("dy")
axes[0].set_title("dx vs dy")

axes[1].hist(centers[:, 2], bins=80)
axes[1].set_xlabel("dheading")
axes[1].set_ylabel("count")
axes[1].set_title("dheading 分布")

# axes[2].scatter(centers[:, 0], centers[:, 2], s=5, alpha=0.6)
# axes[2].set_xlabel("dx")
# axes[2].set_ylabel("dheading")
# axes[2].set_title("dx vs dheading")

plt.tight_layout()
plt.savefig(cluster_fig_path, dpi=200)
plt.close(fig)
print(f'Saved cluster visualization to {cluster_fig_path}')