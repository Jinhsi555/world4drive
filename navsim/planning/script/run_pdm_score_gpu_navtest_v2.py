# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pickle
import traceback
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

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
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.script.run_pdm_score import compute_final_scores, calculate_individual_mapping_scores, \
    create_scene_aggregators
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"


def run_pdm_score_wo_inference(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
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

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
            simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

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

    pdm_results: List[pd.DataFrame] = []

    # navtest

    traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
    )

    scene_loader_tokens = scene_loader.tokens

    tokens_to_evaluate = list(set(scene_loader_tokens) & set(metric_cache_loader.tokens))

    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing navtest scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = model_trajectory[token]['trajectory']

            score_row, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
            score_row["valid"] = True
            # score_row["log_name"] = metric_cache.log_name
            # score_row["frame_type"] = metric_cache.scene_type
            # score_row["start_time"] = metric_cache.timepoint.time_s
            # end_pose = StateSE2(
            #     x=trajectory.poses[-1, 0],
            #     y=trajectory.poses[-1, 1],
            #     heading=trajectory.poses[-1, 2],
            # )
            # absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            # score_row["endpoint_x"] = absolute_endpoint.x
            # score_row["endpoint_y"] = absolute_endpoint.y
            # score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
            # score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
            # score_row["ego_simulated_states"] = [ego_simulated_states]  # used for two-frames extended comfort

        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row = pd.DataFrame([PDMResults.get_empty_results()])
            score_row["valid"] = False
        score_row["token"] = token

        pdm_results.append(score_row)

    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader_inference = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    dataset = Dataset(
        scene_loader=scene_loader_inference,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        is_training=False,
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    # 如果之前走过前向过程，则直接读取pickle文件
    if not os.path.exists(os.getenv('SUBSCORE_PATH')):

        metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

        tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
        num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
        num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
        if num_missing_metric_cache_tokens > 0:
            logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
        if num_unused_metric_cache_tokens > 0:
            logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
        logger.info(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")

        assert len(dataset) == len(tokens_to_evaluate), f'dataloader: {len(dataset)}, tokens: {len(tokens_to_evaluate)}'

        trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
        predictions = trainer.predict(
            AgentLightningModule(
                cfg=cfg.agent.config,
                agent=agent,
            ),
            dataloader,
            return_predictions=True
        )

        dist.barrier()
        all_predictions = [None for _ in range(dist.get_world_size())]

        if dist.is_initialized():
            dist.all_gather_object(all_predictions, predictions)
        else:
            all_predictions.append(predictions)

        if dist.get_rank() != 0:
            return None

        merged_predictions = {}
        for proc_prediction in all_predictions:
            for d in proc_prediction:
                merged_predictions.update(d)

        
        pkl_path = os.getenv('SUBSCORE_PATH')
        pickle.dump(merged_predictions, open(pkl_path, 'wb'))
    else:
        pkl_path = os.getenv('SUBSCORE_PATH')
        merged_predictions = pickle.load(open(pkl_path, 'rb'))

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "model_trajectory": merged_predictions
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    worker = build_worker(cfg)
    score_rows: List[pd.DataFrame] = worker_map(worker, run_pdm_score_wo_inference, data_points)

    pdm_score_df = pd.concat(score_rows)

    # 去掉多余的列
    pdm_score_df = pdm_score_df.drop(columns=['multiplicative_metrics_prod', 'weighted_metrics', 'weighted_metrics_array'])

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios

    # numeric_cols = pdm_score_df.select_dtypes(include=["number", "boolean"]).columns.drop(
    #     labels=["valid"], errors="ignore"
    # )
    # average_values = pdm_score_df[numeric_cols].mean(skipna=True)
    # average_row = pd.Series({col: None for col in pdm_score_df.columns}, dtype=object)
    # average_row.loc[numeric_cols] = average_values

    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)

    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row


    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['pdm_score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )


if __name__ == "__main__":
    main()
