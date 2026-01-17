import logging
import lzma
import os
import pickle
import traceback
import uuid
import json
import gc  # [新增] 引入垃圾回收模块
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import pdm_score, pdm_score_batch_trajectories
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.script.run_pdm_score import compute_final_scores, calculate_individual_mapping_scores, \
    create_scene_aggregators
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory

from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def run_pdm_score(
    args: List[Dict[str, Union[List[str], DictConfig]]]
) -> List[Dict[str, Any]]: 
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    base_vocab_trajectory = args[0]['model_trajectory']

    # 只取在gt轨迹相差一定角度范围内的轨迹去做评测，所以需要预先筛选轨迹
    angle_threshold = cfg.get("angle_threshold", 20)  # 角度
    lat_dist_threshold = cfg.get("lat_dist_threshold", 5.0)  # 横向距离
    lon_dist_threshold = cfg.get("lon_dist_threshold", 10.0)  # 纵向距离
    # dist_threshold = cfg.get("dist_threshold", 10)  # 终点距离
    # 综合考虑横向和纵向距离和角度的权重
    w_lat = cfg.get("w_lat", 10)
    w_lon = cfg.get("w_lon", 1)
    w_angle = cfg.get("w_angle", 5)
    # Top-K 策略
    top_k = cfg.get("top_k", 256)

    # 选择需要的time_horizon
    time_horizon = cfg.get("time_horizon", 4.0)
    # 确保time_horizon是float类型
    time_horizon = float(time_horizon)
    print(f"time_horizon: {time_horizon}")

    # 选择time_horizon
    # time_horizon = cfg.get("time_horizon", 2.0)
    sampling_num_poses = int(time_horizon / 0.1)
    # # proposal_sampling = TrajectorySampling(time_horizon=time_horizon, interval_length=0.1)
    cfg.simulator.proposal_sampling.num_poses = sampling_num_poses
    cfg.scorer.proposal_sampling.num_poses = sampling_num_poses

    

    # 只实例化必要的组件
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
    )

    results: List[Dict[str, Any]] = []
    results_trajectory_scores = {}

    # 预定义目标键值以避免重复创建
    target_keys = [
        'no_at_fault_collisions', 'drivable_area_compliance', 'driving_direction_compliance',
        'traffic_light_compliance', 'ego_progress', 'time_to_collision_within_bound',
        'lane_keeping', 'history_comfort', 'pdm_score'
    ]

    if cfg.traffic_agents == "non_reactive":
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
        )
    elif cfg.traffic_agents == "reactive":
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
        )

    tokens_to_evaluate = list(
        set(scene_loader.tokens) & set(metric_cache_loader.tokens)
    )
    # 只取10个tokens做测试
    # tokens_to_evaluate = tokens_to_evaluate[:2]

    # 预先确定interval_length
    if base_vocab_trajectory.shape[1] == 40:
        interval_length = 0.1
    elif base_vocab_trajectory.shape[1] == 8:
        interval_length = 0.5
    else:
        interval_length = 0.1  # 默认值

    # 计算proposal_sampling和需要的估计步数
    proposal_sampling = TrajectorySampling(time_horizon=time_horizon, interval_length=0.1)
    num_poses = int(time_horizon / interval_length)
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            # scene_dict = scene_loader.scene_frames_dicts[token]
            # 得到gt轨迹，作为参考
            gt_traj = metric_cache.human_trajectory     # [8,3]
            # 把gt轨迹编进model_trajectory中，保留gt的分数
            current_model_trajectory = np.concatenate(
                (gt_traj.poses[np.newaxis, :, :], base_vocab_trajectory), axis=0
            )  # [N+1,8,3]

            # 只取gt轨迹周围一定角度范围内的轨迹去做评测
            model_trajectory_sub = current_model_trajectory[:, :num_poses, :]  # 截取对应时间范围内的轨迹

            # 开始筛选,使用8s轨迹的终点信息去筛选
            target_gt_traj = current_model_trajectory[0]  # gt轨迹(8,3)
            # ## A 计算gt终点与各轨迹终点的距离，使用直线距离去选择
            # distances = np.linalg.norm(
            #     model_trajectory_sub[:, -1, :2] - target_gt_traj[-1, :2], axis=1
            # )  # [N+1,]
            # mask_dist = distances <= dist_threshold
            # A 计算gt终点与各轨迹终点的横向和纵向距离
            lat_dists = np.abs(current_model_trajectory[:, -1, 1] - target_gt_traj[-1, 1])  # 横向距离
            lon_dists = np.abs(current_model_trajectory[:, -1, 0] - target_gt_traj[-1, 0])  # 纵向距离
            mask_lat = lat_dists <= lat_dist_threshold
            mask_lon = lon_dists <= lon_dist_threshold

            # B 计算角度偏差，选取终点方向与gt轨迹终点方向偏差在一定范围内的轨迹
            target_yaw = current_model_trajectory[0, -1, 2]
            ## 计算角度差后把弧度制转成角度制
            yaw_diffs = np.abs(current_model_trajectory[:, -1, 2] - target_yaw)
            yaw_diffs = np.minimum(yaw_diffs, 2 * np.pi - yaw_diffs)  # 转换为最小角度差
            yaw_diffs_deg = np.degrees(yaw_diffs) # 转成角度制
            mask_angle = yaw_diffs_deg <= angle_threshold

            # C 综合mask
            final_mask = mask_lat & mask_lon & mask_angle 
            

            # D 获取筛选后的索引，方便索引到原始vocab
            valid_indices = np.where(final_mask)[0]
            ## 如果筛选结果为空，为了防止报错，可以选择距离最近的那一个
            if len(valid_indices) == 0:
                logger.warning(f"No trajectories found within threshold for token {token}. Using closest one.")
                # valid_indices = np.array([np.argmin(distances)])
            elif len(valid_indices) < top_k:
                logger.warning(f"Only {len(valid_indices)} trajectories found within threshold for token {token}. Using all of them.")

            # 根据横向偏移的均匀采样
            valid_lat_dists = lat_dists[valid_indices]
            sorted_arg_indices = np.argsort(valid_lat_dists)
            ## 生成均匀采样的索引
            linspace_indices = np.linspace(0, len(valid_indices) - 1, min(top_k, len(valid_indices)), dtype=int)
            # 选择最终的索引
            ## 先取排序后的索引，再映射回原始索引
            selected_indices = valid_indices[sorted_arg_indices[linspace_indices]]

            # E 提取候选轨迹
            selected_vocab_trajectories = model_trajectory_sub[selected_indices]  # [M, num_poses, 3]
            




            pdm_results, pdm_scores = pdm_score_batch_trajectories(
                metric_cache=metric_cache,
                vocab_trajectories=selected_vocab_trajectories,
                # future_sampling=TrajectorySampling(time_horizon=time_horizon, interval_length=0.1), # simulator的sampling需要是0.1
                future_sampling=proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
                time_horizon=time_horizon,
                interval_length=interval_length,
            )

            # 直接使用批量处理的结果,pdm_results是列表，每个元素是一个轨迹的评测结果是dataframe结构,最终希望得到的
            # for key in target_keys:
            #     for result in pdm_results:
            #         results_trajectory_scores.setdefault(token, {}).setdefault(key, {}).setdefault(f'{time_horizon}', []).append(result[key].values[0])

            # **新增：存储索引**
            # 7. 保存结果
            # 注意：pdm_results 的第 0 个是 GT，第 1 到 M 个是 candidates，候选集当中也是这样，所以词表中的索引需要-1
            # 我们保存的是 candidate 在原始词表中的索引

            
            # 这里的 selected_indices 对应的是 pdm_results[0:] 的元素
            # 我们为 GT 分配一个特殊的索引 (例如 -1)
            full_indices = selected_indices - 1  # 将索引调整为对应候选轨迹的索引，GT轨迹索引为-1

            for i, result in enumerate(pdm_results):
                # 获取该轨迹对应的原始索引
                origin_idx = int(full_indices[i])
                
                # 存储各个指标
                for key in target_keys:
                    val = result[key].values[0]
                    results_trajectory_scores.setdefault(token, {}).setdefault(key, {}).setdefault(f'{time_horizon}', []).append(val)
                
                results_trajectory_scores.setdefault(token, {}).setdefault('selected_indices', []).append(origin_idx)

            current_token_data = results_trajectory_scores[token]
    
            # 1. 批量转换分数为 float16 (压缩效果：64bit -> 16bit, 省75%)
            for key in target_keys:
                if key in current_token_data and f'{time_horizon}' in current_token_data[key]:
                    raw_list = current_token_data[key][f'{time_horizon}']
                    # 【关键修正】在这里统一转换
                    current_token_data[key][f'{time_horizon}'] = np.array(raw_list, dtype=np.float16)

            # 2. 批量转换索引为 int16 (压缩效果：64bit -> 16bit, 省75%)
            # int16 范围是 -32768 到 32767，足够覆盖你的 8192 个词表 ID
            if 'selected_indices' in current_token_data:
                indices_list = current_token_data['selected_indices']
                # 【关键修正】在这里统一转换
                current_token_data['selected_indices'] = np.array(indices_list, dtype=np.int16)

            # 将转换后的数据保存
            # results_trajectory_scores[token] = current_token_data

            
            # 找到最佳轨迹
            # pdm_scores = results_trajectory_scores[token]['pdm_score']
            # if len(pdm_scores) > 0:
            #     best_idx = np.argmax(pdm_scores)
            #     results_trajectory_scores[token]['best_voc_traj'] = model_trajectory[best_idx]
        except Exception as e:
            logger.warning(f"Agent failed for token {token}: {str(e)}")
            continue

        results.append(results_trajectory_scores)
        # if idx == 1:
        #     break
    
        
    return results

# [新增] 辅助函数：将数据分批
def chunk_list(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# [新增] 辅助函数：追加写入 Pickle
def append_to_pickle(file_path: Path, data: Dict):
    """
    以追加模式写入 pickle 文件。
    注意：这样写入的文件包含多个 pickle 对象，读取时需要循环读取。
    """
    with open(file_path, 'ab') as f:
        pickle.dump(data, f)

# [新增] 辅助函数：如何读取这种追加的 Pickle 文件 (供你参考使用)
def load_appended_pickle(file_path: Path):
    data = {}
    with open(file_path, 'rb') as f:
        while True:
            try:
                batch_data = pickle.load(f)
                data.update(batch_data)
            except EOFError:
                break
    return data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    
    """

    build_logger(cfg)

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    
    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    
    # 如果提供了JSON文件，则根据JSON文件筛选tokens
    # json_scene_file = cfg.get("json_scene_file", None)
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    logger.info(f"使用所有tokens进行评测: {len(tokens_to_evaluate)}")
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")

    # 加载轨迹词典
    traj_vocab_path = cfg.get("traj_vocab_path", None)
    trajs_vocab = np.load(traj_vocab_path, allow_pickle=True)   # (8192,40,3)
    
    # 构建数据点用于并行处理    
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "model_trajectory": trajs_vocab
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    worker = build_worker(cfg)
    # worker_results = worker_map(worker, run_pdm_score, data_points)
    
    # 创建保存目录
    save_dir = Path(cfg.navtrain_score_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存轨迹分数
    # 获取time_horizon信息
    time_horizon = cfg.get("time_horizon", 2.0)
    time_horizon = float(time_horizon)
    navtrain_score_save_path = save_dir / f"navtrain_score_timehorizon_{time_horizon}.pkl"
    # navtrain_score_save_path = save_dir / "navtrain_score.pkl"

    # # 如果文件已存在，先删除，避免追加到旧文件
    # if navtrain_score_save_path.exists():
    #     logger.info(f"Existing file found at {navtrain_score_save_path}, deleting to start fresh.")
    #     os.remove(navtrain_score_save_path)

    # --- [核心修改] 分批处理循环 ---

    # 设置 Batch Size (根据你的内存大小调整，比如 10, 20, 50)
    # 含义：每次处理多少个 Log 文件
    BATCH_SIZE = cfg.get("processing_batch_size", 96) 
    
    total_processed = 0
    logger.info(f"Starting batch processing. Total logs: {len(data_points)}, Batch size: {BATCH_SIZE}")

    for batch_idx, batch_data in enumerate(chunk_list(data_points, BATCH_SIZE)):
        logger.info(f"Processing Batch {batch_idx + 1}...")
        
        # A. 运行 Worker Map (仅针对当前 Batch)
        # 这样 worker_results 只会包含这几个 log 的结果，不会撑爆内存
        worker_results = worker_map(worker, run_pdm_score, batch_data)
        
        # B. 聚合当前 Batch 的结果
        batch_results_dict = {}
        for result in worker_results:
            if isinstance(result, dict):
                # 假设 result 是 {token: {scores...}} 结构
                # 如果 run_pdm_score 返回的是 list of dicts，需要调整这里的逻辑
                if isinstance(result, list): # 你的 run_pdm_score 返回的是 List[Dict]
                     for item in result:
                         batch_results_dict.update(item)
                else:
                    batch_results_dict.update(result)
            else:
                logger.warning(f"Unexpected result type in batch: {type(result)}")

        # C. 追加写入磁盘
        if batch_results_dict:
            append_to_pickle(navtrain_score_save_path, batch_results_dict)
            total_processed += len(batch_results_dict)
            logger.info(f"Batch {batch_idx + 1} saved. Total processed tokens: {total_processed}")
        
        # D. [关键] 清理内存
        del worker_results
        del batch_results_dict
        gc.collect() # 强制运行垃圾回收

    logger.info(f"All done. Results saved to: {navtrain_score_save_path}")
    logger.info("Use the provided `load_appended_pickle` function logic to read this file.")


if __name__ == "__main__":
    main()