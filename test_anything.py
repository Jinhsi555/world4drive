# 查看npy文件
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
# nf = '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/kmeans_navsim_traj_20.npy'
# data = np.load(nf)  # 假设npy文件中存储的是一个数组
# print(data.shape)  # 打印数组的形状

# 读取一个jpg文件，查看其尺寸
from PIL import Image
img_path = '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/dataset/sensor_blobs/trainval/2021.05.12.19.36.12_veh-35_00005_00204/CAM_B0/0a9839a7b1425aad.jpg'
img = Image.open(img_path)
print(img.size)  # 输出图像的尺寸 (宽, 高)
with open('/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/training_w4d_agent_4mode_all_navtrain_ar_4chunk_wm_mlp_ensemble/navtest_eval_one_stage_gt_future/epoch4_navtest.pkl', 'rb') as f:
    pkl_data = pickle.load(f)
print(len(pkl_data))

# with open('/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/dataset/navhard_two_stage/synthetic_scene_pickles/00a0f25bb297f4bb2.pkl', 'rb') as f:
#     pkl_data2 = pickle.load(f)
# print(len(pkl_data2))

# with open('/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/traj_pdm_v2/ori/navtrain_16384.pkl', 'rb') as f:
#     pkl_data2 = pickle.load(f)
# print(len(pkl_data2))

# with open('/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/GTRS/traj_final/16384.npy', 'rb') as f:
#     np_data = np.load(f)
# print(np_data.shape)

# ## 加载npz文件,读取内容
# traj_npz_path = '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/openscenes_train_traj_coords.npz'
# traj_npz_data = np.load(traj_npz_path)
# print(traj_npz_data.files)  # 输出所有的数组名称
# # 输出所有数组的内容
# for array_name in traj_npz_data.files:


traj_np_path = '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/kmeans_navsim_traj_20.npy'
def visualize_traj(np_path: str, traj_idx: int | None = None) -> None:
    data = np.load(np_path)
    print(f'Loaded: {np_path}, shape: {data.shape}')
    if data.ndim < 3 or data.shape[-1] < 2:
        raise ValueError('期望数据形状至少为 (num_traj, num_points, 2)。')

    output_dir = '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/vis/traj_anchor_vis'
    os.makedirs(output_dir, exist_ok=True)

    if traj_idx is not None:
        if not (0 <= traj_idx < data.shape[0]):
            raise IndexError(f'轨迹索引超出范围 0 ~ {data.shape[0] - 1}')
        traj = data[traj_idx]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(traj[..., 0], traj[..., 1], marker='o')
        ax.set_title(f'Trajectory {traj_idx} (XY)')
        output_path = os.path.join(output_dir, f'trajectory_{traj_idx}.png')
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        for idx, traj in enumerate(data):
            ax.plot(traj[..., 0], traj[..., 1], marker='o', markersize=2, linewidth=1, alpha=0.7)
        ax.set_title(f'All {data.shape[0]} Trajectories (XY)')
        output_path = os.path.join(output_dir, 'trajectories_all.png')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'Saved figure to {output_path}')


visualize_traj(traj_np_path)
print("Done")


