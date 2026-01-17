import io
from typing import Any, Callable, List, Tuple

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Scene
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax, add_configured_bev_on_ax_perturbed, add_trajectory_to_bev_ax_perturbed
from navsim.visualization.camera import add_annotations_to_camera_ax, add_camera_ax, add_lidar_to_camera_ax
from navsim.visualization.config import BEV_PLOT_CONFIG, CAMERAS_PLOT_CONFIG, TRAJECTORY_CONFIG
from nuplan.common.actor_state.ego_state import EgoState
from navsim.common.dataclasses import Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import absolute_to_relative_poses

def configure_bev_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the plt ax object for birds-eye-view plots
    :param ax: matplotlib ax object
    :return: configured ax object
    """

    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_aspect("equal")

    # NOTE: x forward, y sideways
    ax.set_xlim(-margin_y / 2, margin_y / 2)
    ax.set_ylim(-margin_x / 2, margin_x / 2)

    # NOTE: left is y positive, right is y negative
    ax.invert_xaxis()

    return ax


def configure_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the ax object for general plotting
    :param ax: matplotlib ax object
    :return: ax object without a,y ticks
    """
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def configure_all_ax(ax: List[List[plt.Axes]]) -> List[List[plt.Axes]]:
    """
    Iterates through 2D ax list/array to apply configurations
    :param ax: 2D list/array of matplotlib ax object
    :return: configure axes
    """
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            configure_ax(ax[i][j])

    return ax


def plot_bev_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, plt.Axes]:
    """
    General plot for birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def plot_bev_with_agent(scene: Scene, agent: AbstractAgent) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory(scene.get_agent_input())

    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax

def plot_bev_with_agent_ourtraj(scene: Scene, traj: np.ndarray, pdms) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """
    from navsim.common.dataclasses import Trajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    # human_trajectory = scene.get_future_trajectory()
    agent_trajectory = traj[:,:, :3] # np.ndarray (8193, 41,3) [x, y, heading] 实际只取前3维
    
    trajectory_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    # add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    for i in range(agent_trajectory.shape[0]):
        traj_i = agent_trajectory[i]
        traj_i = Trajectory(traj_i, trajectory_sampling)
        if i == 0:
            add_trajectory_to_bev_ax(ax, traj_i, TRAJECTORY_CONFIG["pdm_closed"])
            
    add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def plot_bev_with_perturbed_trajectories(
    scene: Scene, 
    pdm_trajectory: np.ndarray, 
    vocab_trajectories: np.ndarray,
    pdm_scores: np.ndarray,
    perturbation_info: dict = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    可视化扰动后的轨迹，包括PDM轨迹和排名前几的词表轨迹
    :param scene: navsim scene dataclass
    :param pdm_trajectory: PDM基准轨迹 (T, 3)
    :param vocab_trajectories: 词表轨迹 (N, T, 3)
    :param pdm_scores: 词表轨迹的PDM分数 (N,)
    :param perturbation_info: 扰动信息字典，包含ego_offset等
    :return: figure and ax object of matplotlib
    """
    from navsim.common.dataclasses import Trajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    
    trajectory_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
    frame_idx = scene.scene_metadata.num_history_frames - 1
    
    # 创建更大的图以容纳图例和分数信息
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    
    # 1. 绘制PDM基准轨迹
    pdm_traj_obj = Trajectory(pdm_trajectory, trajectory_sampling)
    add_trajectory_to_bev_ax(ax, pdm_traj_obj, TRAJECTORY_CONFIG["pdm_closed"])
    
    # 2. 对PDM分数排序，找到前几名
    score_indices = np.argsort(pdm_scores)[::-1]  # 降序排列
    top_scores = pdm_scores[score_indices]
    
    # 3. 绘制最佳轨迹
    if len(score_indices) > 0:
        best_idx = score_indices[0]
        best_traj = Trajectory(vocab_trajectories[best_idx], trajectory_sampling)
        add_trajectory_to_bev_ax(ax, best_traj, TRAJECTORY_CONFIG["best_vocab"])
    
    # 4. 绘制第2-4名轨迹
    for rank in range(1, min(4, len(score_indices))):
        traj_idx = score_indices[rank]
        traj_obj = Trajectory(vocab_trajectories[traj_idx], trajectory_sampling)
        add_trajectory_to_bev_ax(ax, traj_obj, TRAJECTORY_CONFIG["top_vocab"])
    
    configure_bev_ax(ax)
    configure_ax(ax)
    
    # 5. 添加图例和分数信息
    legend_elements = [
        plt.Line2D([0], [0], color=TRAJECTORY_CONFIG["pdm_closed"]["line_color"], 
                  linewidth=3, label='PDM Closed'),
        plt.Line2D([0], [0], color=TRAJECTORY_CONFIG["best_vocab"]["line_color"], 
                  linewidth=2.5, label=f'Best Vocab (Score: {top_scores[0]:.3f})'),
        plt.Line2D([0], [0], color=TRAJECTORY_CONFIG["top_vocab"]["line_color"], 
                  linewidth=1.5, linestyle='--', label='Top 2-4 Vocab')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 6. 添加分数和扰动信息文本
    score_text = f"Top 5 PDM Scores:\n"
    for i in range(min(5, len(top_scores))):
        score_text += f"{i+1}. {top_scores[i]:.4f}\n"
    
    if perturbation_info and 'ego_offset' in perturbation_info:
        offset = perturbation_info['ego_offset']
        score_text += f"\nEgo Offset:\n"
        score_text += f"dx: {offset[0]:.3f}m\n"
        score_text += f"dy: {offset[1]:.3f}m\n"
        score_text += f"dθ: {offset[2]:.3f}rad"
    
    ax.text(0.02, 0.98, score_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig, ax

    


def plot_cameras_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_camera_ax(ax[0, 0], frame.cameras.cam_l0)
    add_camera_ax(ax[0, 1], frame.cameras.cam_f0)
    add_camera_ax(ax[0, 2], frame.cameras.cam_r0)

    add_camera_ax(ax[1, 0], frame.cameras.cam_l1)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_camera_ax(ax[1, 2], frame.cameras.cam_r1)

    add_camera_ax(ax[2, 0], frame.cameras.cam_l2)
    add_camera_ax(ax[2, 1], frame.cameras.cam_b0)
    add_camera_ax(ax[2, 2], frame.cameras.cam_r2)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_lidar(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the lidar pc) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_lidar_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.lidar)

    add_lidar_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.lidar)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_lidar_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.lidar)

    add_lidar_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.lidar)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_annotations(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the bounding boxes) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_annotations_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.annotations)

    add_annotations_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.annotations)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_annotations_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.annotations)

    add_annotations_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.annotations)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def frame_plot_to_pil(
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
) -> List[Image.Image]:
    """
    Plots a frame according to plotting function and return a list of PIL images
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices to save
    :return: list of PIL images
    """

    images: List[Image.Image] = []

    for frame_idx in tqdm(frame_indices, desc="Rendering frames"):
        fig, ax = callable_frame_plot(scene, frame_idx)

        # Creating PIL image from fig
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        images.append(Image.open(buf).copy())

        # close buffer and figure
        buf.close()
        plt.close(fig)

    return images


def frame_plot_to_gif(
    file_name: str,
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
    duration: float = 500,
) -> None:
    """
    Saves a frame-wise plotting function as GIF (hard G)
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices
    :param file_name: file path for saving to save
    :param duration: frame interval in ms, defaults to 500
    """
    images = frame_plot_to_pil(callable_frame_plot, scene, frame_indices)
    images[0].save(file_name, save_all=True, append_images=images[1:], duration=duration, loop=0)


def concat_scenes_to_gif_with_labels(
    file_name: str,
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scenes: List[Scene],
    frame_indices_list: List[List[int]],
    scene_labels: List[str],
    duration: float = 500,
):
    images: List[Image.Image] = []

    for scene, frame_indices, label in zip(scenes, frame_indices_list, scene_labels):
        for frame_idx in tqdm(frame_indices, desc=f"Rendering {label}"):
            fig, ax = callable_frame_plot(scene, frame_idx)

            # 🔵 Add label to the figure
            fig.text(
                0.1,
                0.95,
                label,
                fontsize=12,
                color="black",
                weight="bold",
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.6),
            )

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            images.append(Image.open(buf).copy())

            buf.close()
            plt.close(fig)

    images[0].save(file_name, save_all=True, append_images=images[1:], duration=duration, loop=0)

def plot_perturbation_validation(
    scene: Scene,
    original_pdm_trajectory: np.ndarray,
    perturbed_pdm_trajectory: np.ndarray,
    vocab_trajectories: np.ndarray,
    pdm_scores: np.ndarray,
    ego_offset: np.ndarray,
    save_path: str = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    验证扰动后轨迹的可视化函数，帮助判断扰动是否正确
    :param scene: navsim scene dataclass
    :param original_pdm_trajectory: 原始PDM轨迹 (T, 3)
    :param perturbed_pdm_trajectory: 扰动后PDM轨迹 (T, 3)
    :param vocab_trajectories: 扰动后的词表轨迹 (N, T, 3)
    :param pdm_scores: 词表轨迹的PDM分数 (N,)
    :param ego_offset: ego扰动量 [dx, dy, dtheta]
    :param save_path: 保存路径（可选）
    :return: figure and ax object of matplotlib
    """
    from navsim.common.dataclasses import Trajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    
    trajectory_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
    frame_idx = scene.scene_metadata.num_history_frames - 1
    
    # 创建更大的图以容纳更多信息
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    
    # 1. 绘制原始PDM轨迹（半透明）
    original_traj_obj = Trajectory(original_pdm_trajectory, trajectory_sampling)
    original_config = TRAJECTORY_CONFIG["pdm_closed"].copy()
    original_config["line_color_alpha"] = 0.3
    original_config["fill_color_alpha"] = 0.3
    original_config["line_style"] = ":"
    add_trajectory_to_bev_ax(ax, original_traj_obj, original_config)
    
    # 2. 绘制扰动后的PDM轨迹
    perturbed_traj_obj = Trajectory(perturbed_pdm_trajectory, trajectory_sampling)
    add_trajectory_to_bev_ax(ax, perturbed_traj_obj, TRAJECTORY_CONFIG["pdm_closed"])
    
    # 3. 对PDM分数排序
    score_indices = np.argsort(pdm_scores)[::-1]
    top_scores = pdm_scores[score_indices]
    
    # 4. 绘制最佳词表轨迹
    if len(score_indices) > 0 and top_scores[0] > 0:
        best_idx = score_indices[0]
        best_traj = Trajectory(vocab_trajectories[best_idx], trajectory_sampling)
        add_trajectory_to_bev_ax(ax, best_traj, TRAJECTORY_CONFIG["best_vocab"])
    
    # 5. 绘制第2-4名轨迹
    valid_count = 0
    for rank in range(1, min(4, len(score_indices))):
        traj_idx = score_indices[rank]
        if top_scores[rank] > 0:  # 只绘制有效分数的轨迹
            traj_obj = Trajectory(vocab_trajectories[traj_idx], trajectory_sampling)
            add_trajectory_to_bev_ax(ax, traj_obj, TRAJECTORY_CONFIG["top_vocab"])
            valid_count += 1
    
    configure_bev_ax(ax)
    configure_ax(ax)
    
    # 6. 创建图例
    legend_elements = [
        plt.Line2D([0], [0], color=TRAJECTORY_CONFIG["pdm_closed"]["line_color"], 
                  linewidth=2, linestyle=':', alpha=0.5, label='Original PDM'),
        plt.Line2D([0], [0], color=TRAJECTORY_CONFIG["pdm_closed"]["line_color"], 
                  linewidth=3, label='Perturbed PDM'),
    ]
    
    if len(score_indices) > 0 and top_scores[0] > 0:
        legend_elements.append(
            plt.Line2D([0], [0], color=TRAJECTORY_CONFIG["best_vocab"]["line_color"], 
                      linewidth=2.5, label=f'Best Vocab (Score: {top_scores[0]:.3f})')
        )
    
    if valid_count > 0:
        legend_elements.append(
            plt.Line2D([0], [0], color=TRAJECTORY_CONFIG["top_vocab"]["line_color"], 
                      linewidth=1.5, linestyle='--', label=f'Top 2-4 Vocab ({valid_count} valid)')
        )
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 7. 添加详细信息文本
    info_text = f"Perturbation Validation\n"
    info_text += f"Ego Offset: dx={ego_offset[0]:.3f}m, dy={ego_offset[1]:.3f}m, dθ={ego_offset[2]:.3f}rad\n\n"
    info_text += f"PDM Scores Distribution:\n"
    
    # 统计不同分数区间的轨迹数量
    positive_scores = pdm_scores[pdm_scores > 0]
    zero_scores = pdm_scores[pdm_scores == 0]
    negative_scores = pdm_scores[pdm_scores < 0]
    
    info_text += f"Positive scores: {len(positive_scores)}/{len(pdm_scores)}\n"
    info_text += f"Zero scores: {len(zero_scores)}/{len(pdm_scores)}\n"
    info_text += f"Negative scores: {len(negative_scores)}/{len(pdm_scores)}\n\n"
    
    if len(positive_scores) > 0:
        info_text += f"Best score: {np.max(positive_scores):.4f}\n"
        info_text += f"Mean positive: {np.mean(positive_scores):.4f}\n"
        info_text += f"Std positive: {np.std(positive_scores):.4f}\n"
    
    # 判断扰动是否合理
    info_text += f"\nValidation Status:\n"
    if len(positive_scores) > len(pdm_scores) * 0.1:  # 至少10%的轨迹有正分数
        info_text += "✓ Perturbation seems VALID\n"
        info_text += f"  ({len(positive_scores)}/{len(pdm_scores)} trajectories viable)\n"
    else:
        info_text += "✗ Perturbation may be INVALID\n"
        info_text += f"  (Only {len(positive_scores)}/{len(pdm_scores)} trajectories viable)\n"
    
    # 检查轨迹偏移是否合理
    start_offset = np.linalg.norm(perturbed_pdm_trajectory[0, :2] - original_pdm_trajectory[0, :2])
    if abs(start_offset - np.linalg.norm(ego_offset[:2])) < 0.1:
        info_text += "✓ Spatial offset matches expectation\n"
    else:
        info_text += "✗ Spatial offset inconsistent\n"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
            facecolor="white", alpha=0.9))
    
    plt.title(f"Perturbation Validation: dx={ego_offset[0]:.2f}, dy={ego_offset[1]:.2f}, dθ={ego_offset[2]:.2f}", 
              fontsize=12, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    return fig, ax


def validate_trajectory_perturbation(
    original_pdm_trajectory: np.ndarray,
    perturbed_pdm_trajectory: np.ndarray,
    vocab_trajectories: np.ndarray,
    pdm_scores: np.ndarray,
    ego_offset: np.ndarray,
    verbose: bool = True
) -> dict:
    """
    数值验证轨迹扰动的正确性
    :param original_pdm_trajectory: 原始PDM轨迹
    :param perturbed_pdm_trajectory: 扰动后PDM轨迹
    :param vocab_trajectories: 词表轨迹
    :param pdm_scores: PDM分数
    :param ego_offset: ego扰动量
    :param verbose: 是否打印详细信息
    :return: 验证结果字典
    """
    
    results = {}
    
    # 1. 检查起始点偏移
    start_offset_actual = perturbed_pdm_trajectory[0, :2] - original_pdm_trajectory[0, :2]
    start_offset_expected = ego_offset[:2]
    spatial_error = np.linalg.norm(start_offset_actual - start_offset_expected)
    
    results['spatial_offset_error'] = spatial_error
    results['spatial_offset_valid'] = spatial_error < 0.2  # 20cm误差容忍
    
    # 2. 检查航向角偏移
    heading_offset_actual = perturbed_pdm_trajectory[0, 2] - original_pdm_trajectory[0, 2]
    heading_offset_expected = ego_offset[2]
    heading_error = abs(heading_offset_actual - heading_offset_expected)
    
    results['heading_offset_error'] = heading_error
    results['heading_offset_valid'] = heading_error < 0.1  # 0.1 rad误差容忍
    
    # 3. 分析PDM分数分布
    positive_scores = pdm_scores[pdm_scores > 0]
    zero_scores = pdm_scores[pdm_scores == 0]
    negative_scores = pdm_scores[pdm_scores < 0]
    
    results['num_positive_scores'] = len(positive_scores)
    results['num_zero_scores'] = len(zero_scores)
    results['num_negative_scores'] = len(negative_scores)
    results['total_trajectories'] = len(pdm_scores)
    results['positive_ratio'] = len(positive_scores) / len(pdm_scores)
    
    # 4. 评估扰动合理性
    results['perturbation_valid'] = (
        results['spatial_offset_valid'] and 
        results['heading_offset_valid'] and 
        results['positive_ratio'] > 0.05  # 至少5%的轨迹可行
    )
    
    # 5. 计算分数统计
    if len(positive_scores) > 0:
        results['best_score'] = float(np.max(positive_scores))
        results['mean_positive_score'] = float(np.mean(positive_scores))
        results['std_positive_score'] = float(np.std(positive_scores))
    else:
        results['best_score'] = 0.0
        results['mean_positive_score'] = 0.0
        results['std_positive_score'] = 0.0
    
    if verbose:
        print("=== Trajectory Perturbation Validation ===")
        print(f"Ego offset: dx={ego_offset[0]:.3f}, dy={ego_offset[1]:.3f}, dθ={ego_offset[2]:.3f}")
        print(f"Spatial offset error: {spatial_error:.3f}m ({'✓' if results['spatial_offset_valid'] else '✗'})")
        print(f"Heading offset error: {heading_error:.3f}rad ({'✓' if results['heading_offset_valid'] else '✗'})")
        print(f"Score distribution: {len(positive_scores)}/{len(pdm_scores)} positive ({results['positive_ratio']:.1%})")
        if len(positive_scores) > 0:
            print(f"Best score: {results['best_score']:.4f}")
            print(f"Mean positive score: {results['mean_positive_score']:.4f} ± {results['std_positive_score']:.4f}")
        print(f"Overall validation: {'✓ VALID' if results['perturbation_valid'] else '✗ INVALID'}")
        print("=" * 45)
    
    return results

def convert_absolute_to_relative_trajectory(abs_trajectory: np.ndarray, ego_state: EgoState) -> np.ndarray:
    """
    将绝对坐标轨迹转换为相对坐标轨迹，用于可视化
    :param abs_trajectory: 绝对坐标轨迹 (T, 3)
    :param ego_state: ego状态
    :return: 相对坐标轨迹 (T-1, 3)
    """
    # 跳过第一个点（当前位置），取后续40个点
    future_abs_trajectory = abs_trajectory  # (41, 3)
    
    # 转换为StateSE2对象
    absolute_states = [StateSE2(x, y, heading) for x, y, heading in future_abs_trajectory]
    
    # 转换为相对坐标
    relative_states = absolute_to_relative_poses(absolute_states)
    
    # 转换回numpy数组
    relative_trajectory = np.array([[state.x, state.y, state.heading] for state in relative_states])
    
    return relative_trajectory

def plot_perturbation_comparison(
    scene: Scene,
    original_ego_state: EgoState,
    original_pdm_trajectory: np.ndarray,
    original_vocab_trajectories: np.ndarray,
    original_pdm_scores: np.ndarray,
    perturbed_ego_state: EgoState,
    perturbed_pdm_trajectory: np.ndarray,
    perturbed_vocab_trajectories: np.ndarray,
    perturbed_pdm_scores: np.ndarray,
    ego_offset: np.ndarray,
    save_path: str = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    对比可视化扰动前后的轨迹，并排显示两个场景
    :param scene: navsim scene dataclass
    :param original_ego_state: 原始ego状态
    :param original_pdm_trajectory: 原始PDM轨迹 (41, 3) 绝对坐标
    :param original_vocab_trajectories: 原始词表轨迹 (N, 40, 3) 相对坐标
    :param original_pdm_scores: 原始PDM分数 (N,)
    :param perturbed_ego_state: 扰动后ego状态
    :param perturbed_pdm_trajectory: 扰动后PDM轨迹 (41, 3) 绝对坐标
    :param perturbed_vocab_trajectories: 扰动后词表轨迹 (N, 40, 3) 相对坐标
    :param perturbed_pdm_scores: 扰动后PDM分数 (N,)
    :param ego_offset: ego扰动量 [dx, dy, dtheta]
    :param save_path: 保存路径（可选）
    :return: figure and axes objects
    """
    
    
    trajectory_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
    frame_idx = scene.scene_metadata.num_history_frames - 1
    
    # 将绝对坐标轨迹转换为相对坐标轨迹用于可视化
    original_pdm_relative = convert_absolute_to_relative_trajectory(original_pdm_trajectory, original_ego_state)[1:]
    perturbed_pdm_relative = convert_absolute_to_relative_trajectory(perturbed_pdm_trajectory, perturbed_ego_state)[1:]
    # perturbed_vocab_trajectories = convert_absolute_to_relative_trajectory(perturbed_vocab_trajectories, perturbed_ego_state)[1:]
    perturbed_vocab_trajectories_relative = np.zeros((perturbed_vocab_trajectories.shape[0], perturbed_vocab_trajectories.shape[1] - 1, 3))
    for i in range(perturbed_vocab_trajectories.shape[0]):
        perturbed_vocab_trajectories_relative[i] = convert_absolute_to_relative_trajectory(perturbed_vocab_trajectories[i], perturbed_ego_state)[1:]

    # 创建1x2的子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # === 左图：原始场景 ===
    add_configured_bev_on_ax(ax1, scene.map_api, scene.frames[frame_idx])
    
    # 绘制原始ego位置
    ego_x, ego_y = original_ego_state.rear_axle.x, original_ego_state.rear_axle.y
    ego_heading = original_ego_state.rear_axle.heading
    
    # 绘制原始PDM轨迹（使用相对坐标）
    original_pdm_traj_obj = Trajectory(original_pdm_relative, trajectory_sampling)
    add_trajectory_to_bev_ax(ax1, original_pdm_traj_obj, TRAJECTORY_CONFIG["pdm_closed"])
    
    # 绘制原始最佳词表轨迹
    original_score_indices = np.argsort(original_pdm_scores)[::-1]
    if len(original_score_indices) > 0 and original_pdm_scores[original_score_indices[0]] > 0:
        best_idx = original_score_indices[0]
        best_traj = Trajectory(original_vocab_trajectories[best_idx], trajectory_sampling)
        add_trajectory_to_bev_ax(ax1, best_traj, TRAJECTORY_CONFIG["best_vocab"])
    
    # 绘制原始前几名轨迹
    # for rank in range(1, min(4, len(original_score_indices))):
    #     traj_idx = original_score_indices[rank]
    #     if original_pdm_scores[traj_idx] > 0:
    #         traj_obj = Trajectory(original_vocab_trajectories[traj_idx], trajectory_sampling)
    #         add_trajectory_to_bev_ax(ax1, traj_obj, TRAJECTORY_CONFIG["top_vocab"])

    # 绘制轨迹词表的前两个轨迹
    for traj_idx in range(1):
        traj_obj = Trajectory(original_vocab_trajectories[10], trajectory_sampling)
        add_trajectory_to_bev_ax(ax1, traj_obj, TRAJECTORY_CONFIG["top_vocab"])
    
    configure_bev_ax(ax1)
    configure_ax(ax1)
    ax1.set_title("Original Scene", fontsize=14, fontweight='bold')
    
    # === 右图：扰动后场景 ===
    add_configured_bev_on_ax_perturbed(ax2, scene.map_api, scene.frames[frame_idx], ego_offset=ego_offset)
    
    # 绘制扰动后ego位置
    perturbed_ego_x = perturbed_ego_state.rear_axle.x
    perturbed_ego_y = perturbed_ego_state.rear_axle.y
    perturbed_ego_heading = perturbed_ego_state.rear_axle.heading

    
    # 绘制扰动后PDM轨迹（使用相对坐标）
    perturbed_pdm_traj_obj = Trajectory(perturbed_pdm_relative, trajectory_sampling)
    add_trajectory_to_bev_ax_perturbed(ax2, perturbed_pdm_traj_obj, TRAJECTORY_CONFIG["pdm_closed"], ego_offset)
    
    # 绘制扰动后最佳词表轨迹
    perturbed_score_indices = np.argsort(perturbed_pdm_scores)[::-1]
    if len(perturbed_score_indices) > 0 and perturbed_pdm_scores[perturbed_score_indices[0]] > 0:
        best_idx = perturbed_score_indices[0]
        best_traj = Trajectory(perturbed_vocab_trajectories_relative[best_idx], trajectory_sampling)
        add_trajectory_to_bev_ax_perturbed(ax2, best_traj, TRAJECTORY_CONFIG["best_vocab"], ego_offset)

    # 绘制扰动后前几名轨迹
    # for rank in range(1, min(4, len(perturbed_score_indices))):
    #     traj_idx = perturbed_score_indices[rank]
    #     if perturbed_pdm_scores[traj_idx] > 0:
    #         traj_obj = Trajectory(perturbed_vocab_trajectories_relative[traj_idx], trajectory_sampling)
    #         add_trajectory_to_bev_ax_perturbed(ax2, traj_obj, TRAJECTORY_CONFIG["top_vocab"], ego_offset)

    # 绘制轨迹词表的前两个轨迹
    for traj_idx in range(1):
        traj_obj = Trajectory(perturbed_vocab_trajectories_relative[10], trajectory_sampling)
        add_trajectory_to_bev_ax_perturbed(ax2, traj_obj, TRAJECTORY_CONFIG["top_vocab"], ego_offset)

    configure_bev_ax(ax2)
    configure_ax(ax2)
    ax2.set_title("Perturbed Scene", fontsize=14, fontweight='bold')
    
    # === 添加图例和信息 ===
    # 左图图例
    legend1_elements = [
        plt.Line2D([0], [0], color=TRAJECTORY_CONFIG["pdm_closed"]["line_color"], 
                  linewidth=3, label='PDM Closed'),
        plt.Line2D([0], [0], color=TRAJECTORY_CONFIG["best_vocab"]["line_color"], 
                  linewidth=2.5, label=f'Best (Score: {original_pdm_scores[original_score_indices[0]]:.3f})'),
        plt.Line2D([0], [0], color=TRAJECTORY_CONFIG["top_vocab"]["line_color"], 
                  linewidth=1.5, linestyle='--', label='Top 2-4'),
        plt.scatter([], [], c='red', s=50, marker='o', label='Ego Position')
    ]
    ax1.legend(handles=legend1_elements, loc='upper right', fontsize=9)
    
    # 右图图例
    legend2_elements = [
        plt.Line2D([0], [0], color=TRAJECTORY_CONFIG["pdm_closed"]["line_color"], 
                  linewidth=3, label='PDM Closed'),
        plt.Line2D([0], [0], color=TRAJECTORY_CONFIG["best_vocab"]["line_color"], 
                  linewidth=2.5, label=f'Best (Score: {perturbed_pdm_scores[perturbed_score_indices[0]]:.3f})'),
        plt.Line2D([0], [0], color=TRAJECTORY_CONFIG["top_vocab"]["line_color"], 
                  linewidth=1.5, linestyle='--', label='Top 2-4'),
        plt.scatter([], [], c='orange', s=50, marker='s', label='Perturbed Ego'),
        plt.scatter([], [], c='red', s=50, marker='o', alpha=0.5, label='Original Ego'),
        plt.Line2D([0], [0], color='black', linewidth=1, linestyle='--', alpha=0.6, label='Offset')
    ]
    ax2.legend(handles=legend2_elements, loc='upper right', fontsize=9)
    
    # === 添加统计信息 ===
    # 统计原始场景
    orig_positive = np.sum(original_pdm_scores > 0)
    orig_total = len(original_pdm_scores)
    orig_best_score = np.max(original_pdm_scores) if orig_total > 0 else 0
    
    # 统计扰动场景
    pert_positive = np.sum(perturbed_pdm_scores > 0)
    pert_total = len(perturbed_pdm_scores)
    pert_best_score = np.max(perturbed_pdm_scores) if pert_total > 0 else 0
    
    # 添加总体标题和信息
    fig.suptitle(f'Perturbation Comparison: dx={ego_offset[0]:.2f}m, dy={ego_offset[1]:.2f}m, dθ={ego_offset[2]:.2f}rad', 
                 fontsize=16, fontweight='bold')
    
    # 在底部添加统计对比
    stats_text = f"""
    Original Scene: {orig_positive}/{orig_total} viable trajectories, Best Score: {orig_best_score:.4f}
    Perturbed Scene: {pert_positive}/{pert_total} viable trajectories, Best Score: {pert_best_score:.4f}
    Score Change: {pert_best_score - orig_best_score:+.4f}, Viable Change: {pert_positive - orig_positive:+d}
    """
    
    fig.text(0.5, 0.02, stats_text.strip(), ha='center', va='bottom', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison visualization saved to: {save_path}")
    
    return fig, (ax1, ax2)


def plot_ego_state_comparison(
    scene: Scene,
    original_ego_state: EgoState,
    perturbed_ego_state: EgoState,
    ego_offset: np.ndarray,
    save_path: str = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    专门对比ego状态变化的可视化函数
    :param scene: navsim scene dataclass
    :param original_ego_state: 原始ego状态
    :param perturbed_ego_state: 扰动后ego状态
    :param ego_offset: ego扰动量 [dx, dy, dtheta]
    :param save_path: 保存路径（可选）
    :return: figure and ax objects
    """
    frame_idx = scene.scene_metadata.num_history_frames - 1
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    
    # 原始ego位置和朝向
    orig_x, orig_y = original_ego_state.rear_axle.x, original_ego_state.rear_axle.y
    orig_heading = original_ego_state.rear_axle.heading
    
    # 扰动后ego位置和朝向
    pert_x, pert_y = perturbed_ego_state.rear_axle.x, perturbed_ego_state.rear_axle.y
    pert_heading = perturbed_ego_state.rear_axle.heading
    
    # 绘制ego位置
    ax.scatter([orig_x], [orig_y], c='blue', s=200, marker='o', 
               label='Original Ego', zorder=6, edgecolors='black', linewidth=2)
    ax.scatter([pert_x], [pert_y], c='red', s=200, marker='s', 
               label='Perturbed Ego', zorder=6, edgecolors='black', linewidth=2)
    
    # 绘制位置偏移箭头
    ax.annotate('', xy=(pert_x, pert_y), xytext=(orig_x, orig_y),
                arrowprops=dict(arrowstyle='->', lw=3, color='green', alpha=0.8))
    
    # 绘制ego朝向箭头
    arrow_length = 3.0  # 箭头长度
    
    # 原始朝向
    orig_end_x = orig_x + arrow_length * np.cos(orig_heading)
    orig_end_y = orig_y + arrow_length * np.sin(orig_heading)
    ax.annotate('', xy=(orig_end_x, orig_end_y), xytext=(orig_x, orig_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.7))
    
    # 扰动后朝向
    pert_end_x = pert_x + arrow_length * np.cos(pert_heading)
    pert_end_y = pert_y + arrow_length * np.sin(pert_heading)
    ax.annotate('', xy=(pert_end_x, pert_end_y), xytext=(pert_x, pert_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.7))
    
    configure_bev_ax(ax)
    configure_ax(ax)
    
    # 添加图例
    legend_elements = [
        plt.scatter([], [], c='blue', s=100, marker='o', label='Original Ego', edgecolors='black'),
        plt.scatter([], [], c='red', s=100, marker='s', label='Perturbed Ego', edgecolors='black'),
        plt.Line2D([0], [0], color='green', linewidth=3, alpha=0.8, label='Position Offset'),
        plt.Line2D([0], [0], color='blue', linewidth=2, alpha=0.7, label='Original Heading'),
        plt.Line2D([0], [0], color='red', linewidth=2, alpha=0.7, label='Perturbed Heading')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # 添加详细信息
    distance_offset = np.linalg.norm([pert_x - orig_x, pert_y - orig_y])
    heading_diff = pert_heading - orig_heading
    
    info_text = f"""Ego State Perturbation Details:
    
Position Offset:
  • dx: {ego_offset[0]:.3f}m (actual: {pert_x - orig_x:.3f}m)
  • dy: {ego_offset[1]:.3f}m (actual: {pert_y - orig_y:.3f}m)
  • Distance: {distance_offset:.3f}m

Heading Offset:
  • dθ: {ego_offset[2]:.3f}rad (actual: {heading_diff:.3f}rad)
  • dθ: {np.degrees(ego_offset[2]):.1f}° (actual: {np.degrees(heading_diff):.1f}°)

Original Ego: ({orig_x:.2f}, {orig_y:.2f}, {np.degrees(orig_heading):.1f}°)
Perturbed Ego: ({pert_x:.2f}, {pert_y:.2f}, {np.degrees(pert_heading):.1f}°)"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
            facecolor="white", alpha=0.9))
    
    plt.title(f'Ego State Perturbation: dx={ego_offset[0]:.2f}, dy={ego_offset[1]:.2f}, dθ={ego_offset[2]:.2f}rad', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Ego state comparison saved to: {save_path}")
    
    return fig, ax

def debug_ego_state_perturbation(
    original_ego_state: EgoState,
    perturbed_ego_state: EgoState,
    expected_offset: np.ndarray
) -> dict:
    """
    调试函数：验证ego状态扰动是否正确应用
    :param original_ego_state: 原始ego状态
    :param perturbed_ego_state: 扰动后ego状态  
    :param expected_offset: 期望的扰动量 [dx, dy, dtheta]
    :return: 调试信息字典
    """
    
    # 计算实际偏移
    actual_dx = perturbed_ego_state.rear_axle.x - original_ego_state.rear_axle.x
    actual_dy = perturbed_ego_state.rear_axle.y - original_ego_state.rear_axle.y
    actual_dtheta = perturbed_ego_state.rear_axle.heading - original_ego_state.rear_axle.heading
    
    actual_offset = np.array([actual_dx, actual_dy, actual_dtheta])
    
    # 计算误差
    offset_error = np.abs(actual_offset - expected_offset)
    
    debug_info = {
        'original_position': [original_ego_state.rear_axle.x, original_ego_state.rear_axle.y, original_ego_state.rear_axle.heading],
        'perturbed_position': [perturbed_ego_state.rear_axle.x, perturbed_ego_state.rear_axle.y, perturbed_ego_state.rear_axle.heading],
        'expected_offset': expected_offset.tolist(),
        'actual_offset': actual_offset.tolist(),
        'offset_error': offset_error.tolist(),
        'max_error': np.max(offset_error),
        'perturbation_applied': np.max(offset_error) < 1e-6,  # 如果误差小于1e-6，认为扰动成功应用
        'states_identical': (
            original_ego_state.rear_axle.x == perturbed_ego_state.rear_axle.x and
            original_ego_state.rear_axle.y == perturbed_ego_state.rear_axle.y and
            original_ego_state.rear_axle.heading == perturbed_ego_state.rear_axle.heading
        )
    }
    
    print("=== Ego State Perturbation Debug ===")
    print(f"Original: x={debug_info['original_position'][0]:.6f}, y={debug_info['original_position'][1]:.6f}, θ={debug_info['original_position'][2]:.6f}")
    print(f"Perturbed: x={debug_info['perturbed_position'][0]:.6f}, y={debug_info['perturbed_position'][1]:.6f}, θ={debug_info['perturbed_position'][2]:.6f}")
    print(f"Expected offset: dx={expected_offset[0]:.6f}, dy={expected_offset[1]:.6f}, dθ={expected_offset[2]:.6f}")
    print(f"Actual offset: dx={actual_offset[0]:.6f}, dy={actual_offset[1]:.6f}, dθ={actual_offset[2]:.6f}")
    print(f"Offset error: dx_err={offset_error[0]:.6f}, dy_err={offset_error[1]:.6f}, dθ_err={offset_error[2]:.6f}")
    print(f"Max error: {debug_info['max_error']:.6f}")
    print(f"Perturbation applied: {'✓' if debug_info['perturbation_applied'] else '✗'}")
    print(f"States identical: {'✗ (PROBLEM!)' if debug_info['states_identical'] else '✓'}")
    print("=" * 40)
    
    return debug_info

