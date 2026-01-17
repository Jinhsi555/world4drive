import os
from pathlib import Path
import argparse
import multiprocessing as mp
import math

# 设置无交互后端以便多进程绘图
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hydra
from hydra.utils import instantiate
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from navsim.common.dataclasses import Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.visualization.plots import (
    configure_bev_ax,
    configure_ax,
)
from navsim.visualization.bev import add_configured_bev_on_ax
from navsim.visualization.camera import add_camera_ax

# 使用系统自带 DejaVu Sans，避免缺失字体告警
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# 全局变量 (在 fork 模式下子进程可直接复用，Linux 默认 fork)
all_traj = None
scene_loader = None
save_dir: Path | None = None
BEST_COLOR = '#e15759'
HUMAN_COLOR = '#1f77b4'
OTHERS_COLOR = BEST_COLOR

## 目前仅能绘制非合成场景，合成场景下无人类轨迹
def _process_one(task):
    """单个样本的绘图处理函数。task: (idx, sample_token)"""
    idx, sample_token = task
    try:
        global all_traj, scene_loader, save_dir
        data = all_traj[sample_token]
        scene = scene_loader.get_scene_from_token(sample_token)

        # 兼容 all_trajectories / all_trajectory 两种名称
        if 'all_trajectories' in data:
            trajs = data['all_trajectories']
        elif 'all_trajectory' in data:
            trajs = data['all_trajectory']
        else:
            raise KeyError("缺少轨迹字段: all_trajectories 或 all_trajectory")
        trajs_cls = data['cls_logits']                            # (num_traj,)
        trajs_prob = F.softmax(torch.as_tensor(trajs_cls, dtype=torch.float32), dim=-1)
        best_traj_idx = int(torch.argmax(trajs_prob).cpu().numpy())

        human_traj_dataclass: Trajectory = scene.get_future_trajectory()
        human_xy = np.array([[s[0], s[1]] for s in human_traj_dataclass.poses])

        frame_idx = scene.scene_metadata.num_history_frames - 1
        frame = scene.frames[frame_idx]

        # 6 个单独 BEV 子图
        NUM_GRID = 6
        # 自动计算 rows×cols ≥ NUM_GRID
        cols = math.ceil(math.sqrt(NUM_GRID))
        rows = math.ceil(NUM_GRID / cols)
        # 确保网格足够
        assert rows * cols >= NUM_GRID, f"子图网格太小: {rows}×{cols} < {NUM_GRID}"
        ############################ 6 个子图 ############################

        ############################ 20 个子图 ############################
        # trajs: np.ndarray = data['all_trajectory']              # (num_traj, T, 3)
        # trajs_cls = data['cls_logits']                            # (num_traj,)
        # trajs_prob = F.softmax(torch.as_tensor(trajs_cls, dtype=torch.float32), dim=-1)
        # best_traj_idx = int(torch.argmax(trajs_prob).cpu().numpy())

        # human_traj_dataclass: Trajectory = scene.get_future_trajectory()
        # human_xy = np.array([[s[0], s[1]] for s in human_traj_dataclass.poses])

        # frame_idx = scene.scene_metadata.num_history_frames - 1
        # frame = scene.frames[frame_idx]

        # # 20 个单独 BEV 子图
        # NUM_GRID = 20
        # rows, cols = 4, 5
        ############################ 20 个子图 ############################
        num_trajs_to_plot = min(NUM_GRID, trajs.shape[0])
        fig_grid, axes_grid = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
        axes_flat = axes_grid.flatten()

        for i in range(NUM_GRID):
            ax_bev_i = axes_flat[i]
            if i < num_trajs_to_plot:
                add_configured_bev_on_ax(ax_bev_i, scene.map_api, frame)
                configure_bev_ax(ax_bev_i)
                configure_ax(ax_bev_i)
                this_traj = trajs[i]
                lw = 2.4
                alpha = 1.0
                z = 5
                ax_bev_i.plot(this_traj[:, 1], this_traj[:, 0], color=BEST_COLOR, linewidth=lw, alpha=alpha, zorder=z)
                ax_bev_i.scatter(
                    this_traj[:, 1], this_traj[:, 0],
                    s=14 if i == best_traj_idx else 10,
                    c=BEST_COLOR,
                    edgecolors='white',
                    linewidths=0.4,
                    alpha=1.0,
                    zorder=z + 0.5,
                )
                if human_xy.shape[0] > 0:
                    ax_bev_i.plot(human_xy[:, 1], human_xy[:, 0], color=HUMAN_COLOR, linewidth=1.2, alpha=1.0, zorder=4)
                prob = float(trajs_prob[i].cpu().numpy()) if i < trajs_prob.shape[0] else 0.0
                title_suffix = ' (best)' if i == best_traj_idx else ''
                ax_bev_i.set_title(
                    f"traj {i}{title_suffix}\nprob={prob:.2f}",
                    fontsize=8,
                    color='#d62728' if i == best_traj_idx else '#222222'
                )
            else:
                ax_bev_i.axis('off')

        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], color=BEST_COLOR, lw=2.4, label='best trajectory'),
            Line2D([0], [0], marker='o', color=BEST_COLOR, lw=0, markersize=5, label='trajectory points'),
            Line2D([0], [0], color=HUMAN_COLOR, lw=1.4, label='human'),
        ]
        fig_grid.legend(handles=legend_elems, loc='upper center', ncol=3, fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.995))
        fig_grid.suptitle(f"Sample {sample_token} - 前 {num_trajs_to_plot} 条候选轨迹 (best={best_traj_idx})", fontsize=12)
        fig_grid.tight_layout(rect=[0, 0, 1, 0.94])

        out_path_grid = save_dir / f"{idx:04d}_{sample_token}_grid{NUM_GRID}.png"
        fig_grid.savefig(out_path_grid, dpi=160)
        plt.close(fig_grid)
        return True, f"[Saved Grid{NUM_GRID}] {out_path_grid}"
    except Exception as e:
        return False, f"[Error {sample_token}] {e}"


def main():
    parser = argparse.ArgumentParser(description='并行多轨迹可视化')
    parser.add_argument('--traj-path', type=str, default='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/training_w4d_agent_rl_lr1e-5_2adv_5decaybc_0.1humanlike_0.5entropy_1diversity_bs32_ep59/epoch0_navhard.pkl')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--filter', type=str, default='navhard_two_stage')
    parser.add_argument('--save-dir', type=str, default='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/vis/training_w4d_agent_rl_lr1e-5_2adv_5decaybc_0.1humanlike_0.5entropy_1diversity_bs32_ep59')
    parser.add_argument('--num-workers', type=int, default=8, help='进程数 (1 表示串行)')
    parser.add_argument('--chunk-size', type=int, default=4, help='imap_unordered chunk size')
    parser.add_argument('--limit', type=int, default=-1, help='仅处理前 N 个样本；-1 表示全部')
    args = parser.parse_args()

    global all_traj, scene_loader, save_dir

    print(f"[Load] trajectory pickle: {args.traj_path}")
    all_traj = pickle.load(open(args.traj_path, 'rb'))

    SPLIT = args.split
    FILTER = args.filter

    hydra.initialize(config_path='../../navsim/planning/script/config/common/train_test_split/scene_filter')
    cfg = hydra.compose(config_name=FILTER)
    scene_filter: SceneFilter = instantiate(cfg)
    openscene_data_root = Path(os.getenv('OPENSCENE_DATA_ROOT'))

    scene_loader = SceneLoader(
        openscene_data_root / f"meta_datas/{SPLIT}",
        openscene_data_root / f"sensor_blobs/{SPLIT}",
        scene_filter,
        openscene_data_root / 'navhard_two_stage/sensor_blobs',
        openscene_data_root / 'navhard_two_stage/synthetic_scene_pickles',
        sensor_config=SensorConfig.build_all_sensors(),
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_tokens = list(all_traj.keys())
    if args.limit > 0:
        sample_tokens = sample_tokens[:args.limit]
    tasks = [(i, tok) for i, tok in enumerate(sample_tokens)]

    print(f"总样本数: {len(tasks)} | workers={args.num_workers} chunk={args.chunk_size}")
    if len(tasks) == 0:
        return

    if args.num_workers <= 1:
        # 串行回退
        for t in tasks:
            ok, msg = _process_one(t)
            print(msg)
    else:
        # 进程池并行
        with mp.Pool(processes=args.num_workers) as pool:
            for ok, msg in pool.imap_unordered(_process_one, tasks, chunksize=args.chunk_size):
                print(msg)

    print('完成: 全部场景多轨迹可视化已保存。')


if __name__ == '__main__':
    main()
