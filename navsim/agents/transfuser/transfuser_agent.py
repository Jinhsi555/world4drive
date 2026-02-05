from typing import Any, List, Dict, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR, OneCycleLR, LinearLR, SequentialLR
import pytorch_lightning as pl
import copy
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
# from navsim.agents.transfuser.transfuser_model import TransfuserModel
# from navsim.agents.transfuser.law_model_release import LawModel
from navsim.agents.transfuser.law_model import LAWModel
# from navsim.agents.transfuser.w4d_model import W4DModel
# from navsim.agents.transfuser.w4d_model_v2 import W4DModel
# from navsim.agents.transfuser.w4d_model_v3 import W4DModel
# from navsim.agents.transfuser.w4d_model_v4 import W4DModel
# from navsim.agents.transfuser.w4d_model_v5 import W4DModel

from navsim.agents.transfuser.transfuser_callback import TransfuserCallback
from navsim.agents.transfuser.transfuser_loss import transfuser_loss
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.agents.transfuser.worldadapter_features import WorldAdapterFeatureBuilder, WorldAdapterTargetBuilder
from navsim.common.dataclasses import SensorConfig,AgentInput, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

from hydra.core.global_hydra import GlobalHydra
import hydra
from hydra.utils import instantiate
from navsim.evaluate.pdm_score import pdm_score, pdm_score_batch_trajectories
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.distributions import MultivariateNormal
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from pathlib import Path
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy
import lzma
import pickle
import warnings
# 该正则匹配以“No device id is提供via”开头后面任意内容的UserWarning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"No device id is provided via.*"
)

class TransfuserAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        if self._config.model_version == 4:
            from navsim.agents.transfuser.w4d_model_v4 import W4DModel
        elif self._config.model_version == 1:
            from navsim.agents.transfuser.w4d_model import W4DModel
        elif self._config.model_version == 2:
            from navsim.agents.transfuser.w4d_model_v2 import W4DModel
        elif self._config.model_version == 3:
            from navsim.agents.transfuser.w4d_model_v3 import W4DModel
        elif self._config.model_version == 5:
            from navsim.agents.transfuser.w4d_model_v5 import W4DModel
        elif self._config.model_version == 6:
            from navsim.agents.transfuser.w4d_model_v6 import W4DModel
        elif self._config.model_version == 7:
            from navsim.agents.transfuser.w4d_model_v7 import W4DModel
        elif self._config.model_version == 8:
            from navsim.agents.transfuser.w4d_model_v8 import W4DModel
        elif self._config.model_version == 9:
            from navsim.agents.transfuser.w4d_model_v9 import W4DModel
        elif self._config.model_version == 10:
            from navsim.agents.transfuser.w4d_model_v10 import W4DModel
        elif self._config.model_version == 11:
            from navsim.agents.transfuser.w4d_model_v11 import W4DModel
        elif self._config.model_version == 12:
            from navsim.agents.transfuser.w4d_model_v12 import W4DModel
        elif self._config.model_version == 13:
            from navsim.agents.transfuser.w4d_model_v13 import W4DModel
        elif self._config.model_version == 14:
            from navsim.agents.transfuser.w4d_model_v14 import W4DModel
        elif self._config.model_version == 15:
            from navsim.agents.transfuser.w4d_model_v15 import W4DModel
        elif self._config.model_version == 'single_view':
            from navsim.agents.transfuser.w4d_model_v15_single_view import W4DModel
        elif self._config.model_version == 'wm':
            from navsim.agents.transfuser.w4d_model_v15_wm import W4DModel
        elif self._config.model_version == 'refine':
            from navsim.agents.transfuser.w4d_model_v15_refine import W4DModel
        elif self._config.model_version == 'refine_temporal':
            from navsim.agents.transfuser.w4d_model_v15_refine_temporal_wm import W4DModel
        elif self._config.model_version == 'refine_temporal_short_step':
            from navsim.agents.transfuser.w4d_model_v15_refine_temporal_wm_short_step import W4DModel
        elif self._config.model_version == 'dino_lora':
            from navsim.agents.transfuser.w4d_model_dino_lora import W4DModel

        if self._config.model_name == "W4D":
            self._transfuser_model = W4DModel(config)
        else:
            self._transfuser_model = LAWModel(config)

        if self._config.training_mode == 'ft':
            # 记录pdms=0的场景数
            self.num_pdms_zero_frames = 0
            self.initialize()
            ## 如果是ft模式，需要把感知backbone设置为不更新参数
            # self._transfuser_model.image_encoder.requires_grad_(False)
            if self._config.only_cls_head:

                self._transfuser_model.requires_grad_(False)

                self._transfuser_model._trajectory_head._cls_mlp.requires_grad_(True)

            self._ref_model = copy.deepcopy(self._transfuser_model)
            self._ref_model.requires_grad_(False)

            ## 初始化pdm scorer
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
                hydra.initialize(config_path="../../planning/script/config/pdm_scoring")
                FILTER = "default_run_pdm_score_gpu"
                metric_cache_path = "/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/metric_cache_training"
                overrides = [
                    f"metric_cache_path={metric_cache_path}",
                ]
                pdm_score_cfg = hydra.compose(config_name=FILTER, overrides=overrides)
                self.metric_cache_loader = MetricCacheLoader(Path(pdm_score_cfg.metric_cache_path))
                self.simulator: PDMSimulator = instantiate(pdm_score_cfg.simulator)
                self.scorer: PDMScorer = instantiate(pdm_score_cfg.scorer)
                if pdm_score_cfg.traffic_agents == "non_reactive":
                    self.traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
                        pdm_score_cfg.traffic_agents_policy.non_reactive, self.simulator.proposal_sampling
                    )
                elif pdm_score_cfg.traffic_agents == "reactive":
                    self.traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
                        pdm_score_cfg.traffic_agents_policy.reactive, self.simulator.proposal_sampling
                    )


    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        # if torch.cuda.is_available():
        #     state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        # else:
            # state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
            #     "state_dict"
            # ]
        # state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
        #     "state_dict"
        # ]
        # self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]
        ## 过滤掉不匹配的键，比如ref系列的键
        self.load_state_dict(
            {k.replace("agent.", ""): v for k, v in state_dict.items()},
            strict=False
        )

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            cam_l0=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            cam_l1=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            cam_l2=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            cam_r0=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            cam_r1=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            cam_r2=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            cam_b0=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            lidar_pc=[],
        )
        # return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [WorldAdapterTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [WorldAdapterFeatureBuilder(config=self._config)]

    # def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     """Inherited, see superclass."""
    #     return self._transfuser_model(features)
    def forward_train(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._transfuser_model.forward_train(features)
    
    def forward_train_curriculum(self, features: Dict[str, torch.Tensor], ar_ratio: float = 0.0, global_step: int = 0) -> Dict[str, torch.Tensor]:
        """
        课程学习训练方法，代理调用 model 的 forward_train_curriculum
        
        Args:
            features: 输入特征
            ar_ratio: 自回归比例 (0.0 = 纯 teacher forcing, 1.0 = 纯自回归)
            global_step: 全局训练步数，用于 DDP 多卡同步
        """
        if hasattr(self._transfuser_model, 'forward_train_curriculum'):
            return self._transfuser_model.forward_train_curriculum(features, ar_ratio, global_step)
        else:
            # 如果 model 没有实现 curriculum 方法，回退到普通训练
            return self._transfuser_model.forward_train(features)
    
    def forward_test(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._transfuser_model.forward_test(features)
    

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        logging_prefix: str = None,
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        # return transfuser_loss(targets, predictions, self._config)
        if self._config.training_mode == 'sl':
            
            return self._transfuser_model.compute_loss(features,targets,
                                                    predictions,
                                                    logging_prefix=logging_prefix)
        elif self._config.training_mode == 'ft':

            ## 计算参考模型输出
            with torch.no_grad():
                ref_predictions = self._ref_model.forward_train(features)
                ref_trajectory = ref_predictions['trajectory']
                ref_cls_logits = ref_predictions['cls_logits']
                ref_all_trajectories = ref_predictions['all_trajectories']

            actor_trajectory = predictions['trajectory']    # [B, T, 3]
            actor_trajectory_cls = predictions['cls_logits']    # logits
            actor_all_trajectories = predictions['all_trajectories']    # [B, K, T, 3]
            B = actor_all_trajectories.shape[0]
            K = actor_all_trajectories.shape[1]

            ## 计算trajctory的格式
            if actor_trajectory.shape[1] == 40:
                interval_length = 0.1
            else:
                interval_length = 0.5
            trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=interval_length)
            
            ## 计算每条轨迹的pdm scores
            tokens = features['token']

            all_no_at_fault_collisions = torch.zeros(B, K + 2, device=actor_trajectory.device)  # [B, K+2]
            all_drivable_area_compliance = torch.zeros(B, K + 2, device=actor_trajectory.device)  # [B, K+2]
            all_driving_direction_compliance = torch.zeros(B, K + 2, device=actor_trajectory.device)  # [B, K+2]
            all_traffic_light_compliance = torch.zeros(B, K + 2, device=actor_trajectory.device)  # [B, K+2]
            all_ego_progress = torch.zeros(B, K + 2, device=actor_trajectory.device)  # [B, K+2]
            all_time_to_collision_within_bound = torch.zeros(B, K + 2, device=actor_trajectory.device)  # [B, K+2]
            all_lane_keeping = torch.zeros(B, K + 2, device=actor_trajectory.device)  # [B, K+2]
            all_history_comfort = torch.zeros(B, K + 2, device=actor_trajectory.device)  # [B, K+2]

            all_pdm_scores = torch.zeros(B, K + 2, device=actor_trajectory.device) # [B, K+2]
    
            ### 合并所有的轨迹用于计算pdms
            all_trajs_for_pdm = torch.cat(
                [actor_trajectory.unsqueeze(1), 
                 ref_trajectory.unsqueeze(1), 
                 actor_all_trajectories], dim=1)  # [B, K+2, T, 3]
            all_trajs_for_pdm_np = all_trajs_for_pdm.detach().cpu().numpy()
            for batch_idx, token in enumerate(tokens):

                metric_cache: MetricCache = self.metric_cache_loader.get_from_token(token)
                cur_trajs = all_trajs_for_pdm_np[batch_idx]  # [K+2, T, 3]

                batch_pdm_results, batch_pdm_scores = pdm_score_batch_trajectories(
                    metric_cache=metric_cache,
                    vocab_trajectories=cur_trajs,
                    future_sampling=self.simulator.proposal_sampling,
                    simulator=self.simulator,
                    scorer=self.scorer,
                    traffic_agents_policy=self.traffic_agents_policy,
                )
                all_no_at_fault_collisions[batch_idx] = torch.tensor([batch_pdm_result['no_at_fault_collisions'][0] for batch_pdm_result in batch_pdm_results]).to(all_no_at_fault_collisions.device)
                all_drivable_area_compliance[batch_idx] = torch.tensor([batch_pdm_result['drivable_area_compliance'][0] for batch_pdm_result in batch_pdm_results]).to(all_drivable_area_compliance.device)
                all_driving_direction_compliance[batch_idx] = torch.tensor([batch_pdm_result['driving_direction_compliance'][0] for batch_pdm_result in batch_pdm_results]).to(all_driving_direction_compliance.device)
                all_traffic_light_compliance[batch_idx] = torch.tensor([batch_pdm_result['traffic_light_compliance'][0] for batch_pdm_result in batch_pdm_results]).to(all_traffic_light_compliance.device)
                all_ego_progress[batch_idx] = torch.tensor([batch_pdm_result['ego_progress'][0] for batch_pdm_result in batch_pdm_results]).to(all_ego_progress.device)
                all_time_to_collision_within_bound[batch_idx] = torch.tensor([batch_pdm_result['time_to_collision_within_bound'][0] for batch_pdm_result in batch_pdm_results]).to(all_time_to_collision_within_bound.device)
                all_lane_keeping[batch_idx] = torch.tensor([batch_pdm_result['lane_keeping'][0] for batch_pdm_result in batch_pdm_results]).to(all_lane_keeping.device)
                all_history_comfort[batch_idx] = torch.tensor([batch_pdm_result['history_comfort'][0] for batch_pdm_result in batch_pdm_results]).to(all_history_comfort.device)
                all_pdm_scores[batch_idx] = torch.tensor(batch_pdm_scores).to(all_pdm_scores.device)

            ## 提取actor和ref的各项指标
            # actor指标
            actor_no_at_fault_collisions = all_no_at_fault_collisions[:, 0]  #
            actor_drivable_area_compliance = all_drivable_area_compliance[:, 0]  #
            actor_driving_direction_compliance = all_driving_direction_compliance[:, 0]  #
            actor_traffic_light_compliance = all_traffic_light_compliance[:, 0]  #
            actor_ego_progress = all_ego_progress[:, 0]  #
            actor_time_to_collision_within_bound = all_time_to_collision_within_bound[:, 0]  #
            actor_lane_keeping = all_lane_keeping[:, 0]  #
            actor_history_comfort = all_history_comfort[:, 0]  #
            actor_pdm_score = all_pdm_scores[:, 0]  # [B,]

            # ref指标
            ref_no_at_fault_collisions = all_no_at_fault_collisions[:, 1]  # [B,]
            ref_drivable_area_compliance = all_drivable_area_compliance[:, 1]  # [B,]
            ref_driving_direction_compliance = all_driving_direction_compliance[:, 1]  # [B,]
            ref_traffic_light_compliance = all_traffic_light_compliance[:, 1]  # [B,]
            ref_ego_progress = all_ego_progress[:, 1]  # [B,]
            ref_time_to_collision_within_bound = all_time_to_collision_within_bound[:, 1]  # [B,]
            ref_lane_keeping = all_lane_keeping[:, 1]  # [B,]
            ref_history_comfort = all_history_comfort[:, 1]  # [B,]
            ref_pdm_score = all_pdm_scores[:, 1]  # [B,]

            # 多模态轨迹各指标
            no_at_fault_collisions = all_no_at_fault_collisions[:, 2:]  # [B, K]
            drivable_area_compliance = all_drivable_area_compliance[:, 2:]  # [B, K]
            driving_direction_compliance = all_driving_direction_compliance[:, 2:]  # [B, K]
            traffic_light_compliance = all_traffic_light_compliance[:, 2:]  # [B, K]
            ego_progress = all_ego_progress[:, 2:]  # [B, K]
            time_to_collision_within_bound = all_time_to_collision_within_bound[:, 2:]  # [B, K]
            lane_keeping = all_lane_keeping[:, 2:]  # [B, K]
            history_comfort = all_history_comfort[:, 2:]  # [B, K]
            pdm_scores = all_pdm_scores[:, 2:]  # [B, K]

            ## debug
            ### 如果有一个batch当中存在某个pdms=0的情况，则打印出来
            if (pdm_scores == 0).any():
                # print("存在pdm_score=0的情况，请检查！")
                # print(f'pdm_scores: {pdm_scores}')
                ### 计算当前batc当中，出现pdm_scores=0的场景数
                # zero_frames = (pdm_scores == 0).any(dim=1)  # [B,]
                self.num_pdms_zero_frames += 1
                # print(f"num_pdms_zero_frames: {self.num_pdms_zero_frames}")


            ## 计算reward
            # actor_reward = actor_pdm_score  # [B,]
            # ref_reward = ref_pdm_score    # [B,]
            # rewards = pdm_scores  # [B, K]

            ### 考虑使用激进策略，高ep的风格 #####################
            rewards = 0.9 * pdm_scores + 0.1 * ego_progress
            actor_reward = 0.9 * actor_pdm_score + 0.1 * actor_ego_progress  # [B,]
            ref_reward = 0.9 * ref_pdm_score + 0.1 * ref_ego_progress  # [B,]

            ### 考虑使用激进策略，高ep的风格 #####################

            ### 考虑添加人类相似度的奖励 #####################
            human_traj = targets['trajectory']  # [B, T, 3]
            actor_traj_idx = torch.argmax(actor_trajectory_cls, dim=1)  # [B,]
            ref_traj_idx = torch.argmax(ref_cls_logits, dim=1)  # [B,]
            #### 计算终点距离
            all_final_dist = - torch.norm(actor_all_trajectories[:, :, -1, :] 
                                          - human_traj[:, -1, :].unsqueeze(1), dim=-1)  # [B, K]
            # 使用exp映射来获得0-1的相似性分数
            all_final_end_sim = torch.exp(all_final_dist / 10)  # [B, K]
            actor_final_end_sim = all_final_end_sim[torch.arange(B), actor_traj_idx]  # [B,]

            # 修改下面这一行，确保维度对齐
            ref_all_final_dist = - torch.norm(ref_all_trajectories[:, :, -1, :] 
                                              - human_traj[:, -1, :].unsqueeze(1), dim=-1)  # [B, K]
            ref_all_final_end_sim = torch.exp(ref_all_final_dist / 10)  # [B, K]
            ref_final_end_sim = ref_all_final_end_sim[torch.arange(B), ref_traj_idx]  # [B,]
            # #### 计算速度和加速度的相似性
            # dt=0.5
            # vel_pred = (actor_all_trajectories[:, :, 1:, :] - actor_all_trajectories[:, :, :-1, :]) / dt
            # vel_human = (human_traj[:, 1:, :] - human_traj[:, :-1, :]) / dt
            # acc_pred = (vel_pred[:, :, 1:, :] - vel_pred[:, :, :-1, :]) / dt
            # acc_human = (vel_human[:, 1:, :] - vel_human[:, :-1, :]) / dt
            # # 速度角度相似性（cosine）
            # cos_sim = torch.nn.functional.cosine_similarity(
            #     vel_pred.mean(dim=2), vel_human.mean(dim=1, keepdim=True), dim=-1
            # )

            # # 速度和加速度的分布差异
            # vel_diff = torch.norm(vel_pred.mean(dim=2) - vel_human.mean(dim=1, keepdim=True), dim=-1)
            # acc_diff = torch.norm(acc_pred.mean(dim=2) - acc_human.mean(dim=1, keepdim=True), dim=-1)

            # # 归一化各种奖励
            # cos_sim = (1 + cos_sim) / 2  # 将cos_sim从[-1, 1]映射到[0, 1]
            # vel_sim = torch.exp(-vel_diff / 2)  # 使用指数映射将差异转换为相似性分数
            # acc_sim = torch.exp(-acc_diff / 2)  # 使用指数映射将差异转换为相似性分数

            # reward_kinematic = 0.3 * cos_sim + 0.2 * vel_sim + 0.2 * acc_sim + 0.3 * all_final_end_sim

            # actor_reward = 0.9 * actor_pdm_score + 0.1 * actor_final_end_sim  # [B,]
            # ref_reward = 0.9 * ref_pdm_score + 0.1 * ref_final_end_sim  # [B,]
            # rewards = 0.9 * pdm_scores + 0.1 * all_final_end_sim  # [B, K]
            ### 考虑添加人类相似度的奖励 #####################

            ## 计算advantage
            adv = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)  # [B, K]
            #### 使用温度系数来调整advantage的分布
            adv_temp = getattr(self._config, "adv_temp", 1.0)
            adv = adv / adv_temp

            ## 使用绝对baseline并标准化advantage (zero-mean, unit-variance)
            # raw_adv = rewards - ref_reward.unsqueeze(1)  # [B, K]
            # adv = (raw_adv - raw_adv.mean(dim=1, keepdim=True)) / (raw_adv.std(dim=1, keepdim=True) + 1e-8)  # [B, K]

            ## 计算logprob,这里特意使用ratio=1，表示不使用old policy
            cur_log_probs = torch.log_softmax(actor_trajectory_cls, dim=-1)  # (B, K)
            old_log_probs = cur_log_probs.clone().detach()  # (B, K)

            ratio = torch.exp(cur_log_probs - old_log_probs)  # (B, K)

            ## 计算policy loss
            actor_loss1 = -adv * ratio  # (B, K)
            actor_loss2 = -adv * torch.clamp(ratio, 0.8, 1.2)  # (B, K)
            actor_loss = torch.mean(torch.max(actor_loss1, actor_loss2))

            ## 1. 计算KL loss, 避免策略更新过大
            kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
            actor_log_probs = torch.nn.functional.log_softmax(actor_trajectory_cls, dim=-1)
            ref_probs = torch.nn.functional.softmax(ref_cls_logits, dim=-1).detach()
            kl_loss = kl_loss_fn(actor_log_probs, ref_probs) * self._config.kl_loss_weight

            # ## 2. 计算cross entropy loss, 优化cls_logits使其与奖励对齐
            # best_idx = torch.argmax(rewards, dim=1)  # [B,]
            # ce_loss = torch.nn.CrossEntropyLoss()
            # score_align_loss = ce_loss(actor_trajectory_cls, best_idx) * self._config.ce_loss_weight

            ## 2. 计算对比损失，使得奖励高的模态得分更高
            best_idx = torch.argmax(rewards, dim=1)  # [B,]
            temperature = getattr(self._config, "contrastive_temperature", 1.0)
            # 缩放 logits
            logits_scaled = actor_trajectory_cls / temperature  # (B, K)
            # 取正样本 logits
            pos_logits = logits_scaled[torch.arange(B), best_idx]  # (B,)
            # 计算所有负样本的 log-sum-exp
            neg_logits_sum = torch.logsumexp(logits_scaled, dim=1)  # (B,)
            # InfoNCE loss = -pos + logsumexp(all)
            contrastive_loss = (neg_logits_sum - pos_logits).mean() * self._config.ce_loss_weight

            score_align_loss = contrastive_loss

            ## 3. 计算BC loss, 模仿人类轨迹,使用概率采样,引入线性衰减因子
            # decay_factor = 1 - (self.trainer.global_step / self.trainer.estimated_stepping_batches)
            # best_traj = actor_all_trajectories[torch.arange(B), best_idx]  # [B, T, 3]
            reward_probs = torch.softmax(rewards, dim=1)  # [B, K]
            sampled_idx = torch.multinomial(reward_probs, num_samples=1).squeeze(1)  # [B,]，按概率采样1个模态
            human_traj = targets['trajectory']  # [B, T, 3]
            sampled_traj = actor_all_trajectories[torch.arange(B), sampled_idx]  # [B, T, 3]
            ade_bc_loss = torch.mean(torch.norm(sampled_traj - human_traj, p=1,dim=-1)) * self._config.ade_best_traj_loss_weight
            fde_bc_loss = torch.mean(torch.norm(sampled_traj[:, -1] - human_traj[:, -1], p=1,dim=-1)) * self._config.fde_best_traj_loss_weight
            #### 衰减因子
            # ade_bc_loss = ade_bc_loss * decay_factor
            # fde_bc_loss = fde_bc_loss * decay_factor
            #### 计算多模态轨迹间的终点的L2 loss来评价轨迹多样性
            traj_endpoints = actor_all_trajectories[:, :, -1, :]  # [B, K, 3]
            pairwise_dist = torch.norm(
                traj_endpoints.unsqueeze(2) - traj_endpoints.unsqueeze(1), dim=-1
            )  # [B, K, K]
            mask = ~torch.eye(K, dtype=torch.bool, device=pairwise_dist.device)
            # 对每个样本，求所有不同轨迹终点距离的均值
            # diversity = pairwise_dist.masked_select(mask).view(B, -1).mean(dim=1)  # [B,]
            ##### 计算最近的距离，来防止轨迹聚集
            pairwise_dist_for_min = pairwise_dist.clone()
            pairwise_dist_for_min.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
            min_dist, _ = pairwise_dist_for_min.view(B, -1).min(dim=1) # [B,]
            diversity_weight = getattr(self._config, "diversity_end_loss_weight", 1.0)
            # diversity_loss = -diversity.mean() * diversity_weight
            diversity_loss = -min_dist.mean() * diversity_weight
            

            
            
            ## 4. 计算熵损失，鼓励策略多样性
            actor_probs = torch.nn.functional.softmax(actor_trajectory_cls, dim=-1)
            actor_entropy = -torch.sum(actor_probs * torch.log(actor_probs + 1e-8), dim=-1).mean()
            entropy_loss = - actor_entropy * self._config.entropy_loss_weight
            ## 计算总loss
            loss_dict = {}

            ## 记录一下diff信息，后续需要删除，只记录，不计算loss

            actor_reward_diff = (actor_reward - ref_reward).mean()
            actor_pdms_diff = (actor_pdm_score - ref_pdm_score).mean()
            actor_nc_diff = (actor_no_at_fault_collisions - ref_no_at_fault_collisions).mean()
            actor_dac_diff = (actor_drivable_area_compliance - ref_drivable_area_compliance).mean()
            actor_ddc_diff = (actor_driving_direction_compliance - ref_driving_direction_compliance).mean()
            actor_tlc_diff = (actor_traffic_light_compliance - ref_traffic_light_compliance).mean()
            actor_ep_diff = (actor_ego_progress - ref_ego_progress).mean()
            actor_ttc_diff = (actor_time_to_collision_within_bound - ref_time_to_collision_within_bound).mean()
            actor_lk_diff = (actor_lane_keeping - ref_lane_keeping).mean()
            actor_hc_diff = (actor_history_comfort - ref_history_comfort).mean()
            loss_dict['actor_reward_diff'] = actor_reward_diff
            loss_dict['actor_pdms_diff'] = actor_pdms_diff
            loss_dict['actor_nc_diff'] = actor_nc_diff
            loss_dict['actor_dac_diff'] = actor_dac_diff
            loss_dict['actor_ddc_diff'] = actor_ddc_diff
            loss_dict['actor_tlc_diff'] = actor_tlc_diff
            loss_dict['actor_ep_diff'] = actor_ep_diff
            loss_dict['actor_ttc_diff'] = actor_ttc_diff
            loss_dict['actor_lk_diff'] = actor_lk_diff
            loss_dict['actor_hc_diff'] = actor_hc_diff

            loss_dict['actor_nc_score'] = actor_no_at_fault_collisions.mean()
            loss_dict['actor_dac_score'] = actor_drivable_area_compliance.mean()
            loss_dict['actor_ddc_score'] = actor_driving_direction_compliance.mean()
            loss_dict['actor_tlc_score'] = actor_traffic_light_compliance.mean()
            loss_dict['actor_ep_score'] = actor_ego_progress.mean()
            loss_dict['actor_ttc_score'] = actor_time_to_collision_within_bound.mean()
            loss_dict['actor_lk_score'] = actor_lane_keeping.mean()
            loss_dict['actor_hc_score'] = actor_history_comfort.mean()
            loss_dict['actor_pdm_score'] = actor_pdm_score.mean()

            #### 记录存在0pdm的batch数
            loss_dict['num_pdms_zero_frames'] = self.num_pdms_zero_frames
            #### 记录当前轨迹类人相似度
            loss_dict['actor_final_end_sim'] = actor_final_end_sim.mean()  


            loss_dict['actor_loss'] = actor_loss
            loss_dict['score_align_loss'] = score_align_loss
            loss_dict['kl_loss'] = kl_loss
            loss_dict['ade_bc_loss'] = ade_bc_loss
            loss_dict['fde_bc_loss'] = fde_bc_loss
            loss_dict['entropy_loss'] = entropy_loss
            loss_dict['diversity_loss'] = diversity_loss

            
            return loss_dict

    def update_target_encoder(self, momentum: float):
        if hasattr(self._transfuser_model, 'update_target_encoder'):
            self._transfuser_model.update_target_encoder(momentum)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        # return torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr)
        if self._config.training_mode == 'sl':
            # return torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr)
            
            # 使用线性预热 + 余弦退火学习率调度
            optimizer = torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr)
            
            # 读取最大训练步数
            max_step = self.trainer.estimated_stepping_batches
            
            # 预热步数：前10%的训练步数用于预热
            warmup_steps = int(0.1 * max_step)
            cosine_steps = max_step - warmup_steps
            
            # 线性预热调度器：从0线性增加到初始学习率
            warmup_scheduler = LinearLR(
                optimizer=optimizer,
                start_factor=0.01,  # 从初始学习率的1%开始
                end_factor=1.0,     # 预热结束时达到完整学习率
                total_iters=warmup_steps
            )
            
            # 余弦退火学习率调度器
            cosine_scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=cosine_steps,
                eta_min=1e-6,  # 最小学习率
            )
            
            # 组合预热和余弦退火
            scheduler = SequentialLR(
                optimizer=optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]  # 在warmup_steps步之后切换到余弦退火
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # 每步更新学习率
                },
            }
        elif self._config.training_mode == 'ft':
            ### 使用定制学习率策略
            param_groups: List[Dict[str, Any]] = []
            lr_main = self._lr
            lr_backbone = lr_main * 0.1
            backbone_params, other_params = [], []
            for name, param in self._transfuser_model.named_parameters():
                if not param.requires_grad:
                    continue
                if "image_encoder" in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
            if backbone_params:
                param_groups.append({"params": backbone_params, "lr": lr_backbone})
            if other_params:
                param_groups.append({"params": other_params, "lr": lr_main})
            weight_decay = getattr(self._config, "weight_decay", 0.01)

            optimizer = torch.optim.AdamW(param_groups, lr=lr_main, weight_decay=weight_decay)

            # optimizer = torch.optim.AdamW(self._transfuser_model.parameters(), lr=self._lr, weight_decay=weight_decay)

             # 读取最大iter数
            max_step = self.trainer.estimated_stepping_batches
            
            ## one cycle lr
            final_div_factor = self._lr / 1e-7
            scheduler = OneCycleLR(
                optimizer=optimizer,
                max_lr=self._lr,
                pct_start=0.25,
                total_steps=max_step,
                final_div_factor=final_div_factor,
            )

            ## 余弦退火学习率
            # scheduler = CosineAnnealingLR(
            #     optimizer=optimizer,
            #     T_max=max_step,
            #     eta_min=1e-3 * lr_main,
            # )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""

        if self._config.training_mode == 'ft':
            return [
                TransfuserCallback(self._config),
                pl.callbacks.ModelCheckpoint(every_n_epochs=1, save_top_k=-1),
                pl.callbacks.LearningRateMonitor(logging_interval='step')  # 监控学习率，每步记录
            ]
        else:
            return [
                TransfuserCallback(self._config),
                pl.callbacks.ModelCheckpoint(every_n_epochs=5, save_top_k=-1),
                pl.callbacks.LearningRateMonitor(logging_interval='step')  # 监控学习率，每步记录
            ]
    
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        # forward pass
        with torch.no_grad():
            predictions = self.forward_test(features)
            poses = predictions["trajectory"].squeeze(0).numpy()


        # extract trajectory
        return Trajectory(poses)
