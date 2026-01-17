import logging
import lzma
import os
import pickle
import traceback
import uuid
import json
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
    model_trajectory = args[0]['model_trajectory']

    # 只取在gt轨迹相差一定角度范围内的轨迹去做评测，所以需要预先筛选轨迹
    angle_threshold = cfg.get("angle_threshold", 20)  # 角度

    # 选择time_horizon
    # time_horizon = cfg.get("time_horizon", 2.0)
    # sampling_num_poses = int(time_horizon / 0.1)
    # # proposal_sampling = TrajectorySampling(time_horizon=time_horizon, interval_length=0.1)
    # cfg.simulator.proposal_sampling.num_poses = sampling_num_poses
    # cfg.scorer.proposal_sampling.num_poses = sampling_num_poses

    

    # 只实例化必要的组件
    # simulator: PDMSimulator = instantiate(cfg.simulator)
    # scorer: PDMScorer = instantiate(cfg.scorer)
    
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

    # if cfg.traffic_agents == "non_reactive":
    #     traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
    #         cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
    #     )
    # elif cfg.traffic_agents == "reactive":
    #     traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
    #         cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
    #     )

    tokens_to_evaluate = list(
        set(scene_loader.tokens) & set(metric_cache_loader.tokens)
    )

    # 预先确定interval_length
    if model_trajectory.shape[1] == 40:
        interval_length = 0.1
    elif model_trajectory.shape[1] == 8:
        interval_length = 0.5
    else:
        interval_length = 0.1  # 默认值
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            scene_dict = scene_loader.scene_frames_dicts[token]
            # 得到gt轨迹，作为参考
            gt_traj = metric_cache.human_trajectory     # [8,3]
            # 把gt轨迹编进model_trajectory中，保留gt的分数
            model_trajectory = np.concatenate(
                (gt_traj.poses[np.newaxis, :, :], model_trajectory), axis=0
            )  # [N+1,8,3]

            # 只取gt轨迹周围一定角度范围内的轨迹去做评测
            # 暂时只取一部分traj来计算pdms
            time_horizon_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            # time_horizon_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            for time_horizon in time_horizon_list:
                # 重新实例化
                simulator: PDMSimulator = instantiate(cfg.simulator)
                scorer: PDMScorer = instantiate(cfg.scorer)
                metric_cache = metric_cache_loader.get_from_token(token)

                proposal_sampling = TrajectorySampling(time_horizon=time_horizon, interval_length=0.1)
                simulator.proposal_sampling = proposal_sampling
                scorer.proposal_sampling = proposal_sampling
                num_poses = int(time_horizon / interval_length)
                model_trajectory_sub = model_trajectory[:, :num_poses, :]  # 截取对应时间范围内的轨迹
                # 加载正确的traffic_agents_policy
                if cfg.traffic_agents == "non_reactive":
                    traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
                        cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
                    )
                elif cfg.traffic_agents == "reactive":
                    traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
                        cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
                    )


                pdm_results, pdm_scores = pdm_score_batch_trajectories(
                    metric_cache=metric_cache,
                    vocab_trajectories=model_trajectory_sub,
                    # future_sampling=TrajectorySampling(time_horizon=time_horizon, interval_length=0.1), # simulator的sampling需要是0.1
                    future_sampling=proposal_sampling,
                    simulator=simulator,
                    scorer=scorer,
                    traffic_agents_policy=traffic_agents_policy,
                    time_horizon=time_horizon,
                    interval_length=interval_length,
                )

                # 直接使用批量处理的结果,pdm_results是列表，每个元素是一个轨迹的评测结果是dataframe结构,最终希望得到的
                for key in target_keys:
                    for result in pdm_results:
                        results_trajectory_scores.setdefault(token, {}).setdefault(key, {}).setdefault(f'{time_horizon}', []).append(result[key].values[0])
        
                    
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
    worker_results = worker_map(worker, run_pdm_score, data_points)

    all_results = {}
    for result in worker_results:
        if isinstance(result, dict):
            all_results.update(result)
        else:
            logger.warning(f"Unexpected result type: {type(result)}, skipping.")
    
    # 创建保存目录
    save_dir = Path(cfg.navtrain_score_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存轨迹分数
    navtrain_score_save_path = save_dir / "navtrain_score.pkl"
    with open(navtrain_score_save_path, 'wb') as f:
        pickle.dump(all_results, f)
    logger.info(f"Navtrain scores saved to: {navtrain_score_save_path}")

    logger.info(f"Processed {len(all_results)} tokens, Results saved to: {save_dir}")




if __name__ == "__main__":
    main()