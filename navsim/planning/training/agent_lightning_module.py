import os
from typing import Dict, Tuple
import numpy as np
import time

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.utils.util import CosineScheduler



def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__qualname__} took {end - start:.4f} seconds to execute\n")
        return result
    return wrapper

class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, 
                 cfg: TransfuserConfig,
                 agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self._cfg = cfg
        self.agent = agent

    @timed
    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        if logging_prefix=='train' and not self._cfg.use_wm:
            prediction = self.agent.forward_train(features)
            loss = self.agent.compute_loss(features, targets, prediction, logging_prefix=logging_prefix)
        elif logging_prefix=='train' and self._cfg.use_wm:
            # current t
            prediction = self.agent.forward_train(features)
            
            # # prev t-1
            # prev_features = {}
            # prev_features['camera_feature'] = features['camera_feature_prev_3']
            # prev_features['status_feature'] = features['status_feature_prev_3']

            # prev_targets = {}
            # prev_targets['trajectory'] = targets['trajectory_prev_3']

            # # self.agent._transfuser_model.prev_keyval_feat = None
            # prev_prediction = self.agent.forward_train(prev_features)
            
            # prev_prediction['next_latent'] = prediction['cur_latent']
            
            # next t+8
            next_features = {}
            next_features['camera_feature'] = features['camera_feature_next_8']
            next_features['status_feature'] = features['status_feature_next_8']
            
            next_prediction = self.agent.forward_train(next_features)

            prediction['next_latent'] = next_prediction['cur_latent']

            # compute loss
            # prev_loss = self.agent.compute_loss(prev_features, prev_targets, prev_prediction, logging_prefix=logging_prefix)          
            cur_loss = self.agent.compute_loss(features, targets, prediction, logging_prefix=logging_prefix)
            
            # # 为prev_loss中的损失项添加前缀prev_
            # prev_loss = {f'prev_{k}': v for k, v in prev_loss.items()}

            # # 合并两个损失字典
            # loss = {**prev_loss, **cur_loss}
            loss = {**cur_loss}


        elif logging_prefix=='val' and not self._cfg.use_wm:
            prediction = self.agent.forward_test(features)
            loss = self.agent.compute_loss(features, targets, prediction, logging_prefix=logging_prefix)

        elif logging_prefix=='val' and self._cfg.use_wm:
            # # 间隔2 frame
            # prev_features = {}
            # prev_features['camera_feature'] = features['camera_feature_prev_3']
            # prev_features['status_feature'] = features['status_feature_prev_3']

            # prev_prediction = self.agent.forward_test(prev_features)

            prediction = self.agent.forward_test(features)

            # next t+8
            next_features = {}
            next_features['camera_feature'] = features['camera_feature_next_8']
            next_features['status_feature'] = features['status_feature_next_8']
            
            next_prediction = self.agent.forward_train(next_features)

            prediction['next_latent'] = next_prediction['cur_latent']

            # prediction = self.agent.forward_test(features)
            loss = self.agent.compute_loss(features, targets, prediction, logging_prefix=logging_prefix)
        # 分别记录每个损失项
        for loss_name, loss_value in loss.items():
            self.log(
                f"{logging_prefix}/{loss_name}",  # 日志名称，例如 "train/traj_loss"
                loss_value,  # 损失值
                on_step=True,  # 记录每一步的损失
                on_epoch=True,  # 记录每个 epoch 的损失
                prog_bar=True,  # 在进度条中显示
                sync_dist=True  # 分布式训练时同步
            )

        if logging_prefix == 'train' and self._cfg.training_mode == 'ft':
            ## 从字典中移除不需要的损失项
            for key in ['actor_reward_diff', 'actor_pdms_diff', 'actor_nc_diff', 'actor_dac_diff',
                        'actor_ddc_diff', 'actor_tlc_diff', 'actor_ep_diff', 'actor_ttc_diff',
                        'actor_lk_diff', 'actor_hc_diff',
                        'actor_nc_score', 'actor_dac_score', 'actor_ddc_score', 'actor_tlc_score',
                        'actor_ep_score', 'actor_ttc_score', 'actor_lk_score', 'actor_hc_score',
                        'actor_pdm_score', 'num_pdms_zero_frames', 'actor_final_end_sim'
                        ]:
                if key in loss:
                    loss.pop(key)

        total_loss = sum(loss.values())

        if logging_prefix == 'train':
            grad_params = []
            if hasattr(self.agent, "parameters"):
                grad_params = [p for p in self.agent.parameters() if p.requires_grad]
            if grad_params:
                device = grad_params[0].device

                def _grad_norm(loss_tensor: Tensor) -> Tensor:
                    grads = torch.autograd.grad(
                        outputs=loss_tensor,
                        inputs=grad_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    norm_sq = torch.zeros(1, device=device)
                    for grad in grads:
                        if grad is not None:
                            norm_sq = norm_sq + grad.pow(2).sum()
                    return torch.sqrt(norm_sq + 1e-12)

                for loss_name, loss_value in loss.items():
                    if isinstance(loss_value, Tensor) and loss_value.requires_grad:
                        target = loss_value if loss_value.ndim == 0 else loss_value.sum()
                        grad_norm = _grad_norm(target)
                    else:
                        grad_norm = torch.zeros(1, device=device)
                    self.log(
                        f"{logging_prefix}_grad/{loss_name}",
                        grad_norm,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )

                if isinstance(total_loss, Tensor) and total_loss.requires_grad:
                    total_target = total_loss if total_loss.ndim == 0 else total_loss.sum()
                    total_grad_norm = _grad_norm(total_target)
                    self.log(
                        f"{logging_prefix}_grad/total",
                        total_grad_norm,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                    )
        return total_loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")
    
    def test_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        features, targets = batch
        prediction = self.agent.forward_test(features)
        # =================== save trajectory cond, trajecotry nononai, trajectory and navi =======================
        log_dir = self.logger.log_dir if self.logger else "./default_log_dir"
        trajs_dir_path = os.path.join(log_dir, "trajs")
        os.makedirs(trajs_dir_path, exist_ok=True)
        
        trajs_num = prediction['trajectory'].shape[0]
        for i in range(trajs_num):
            token = features['token'][i]
            traj = prediction['trajectory'][i].squeeze(0).cpu().numpy()
            np.save(f'{trajs_dir_path}/{token}.npy', traj)
        return prediction

    def predict_step(self, 
                     batch,
                     batch_idx: int):
        self.agent.eval()
        features, _ = batch
        with torch.no_grad():
            if self._cfg.model_name == "W4D":
                if self._cfg.use_wm:
                    prev_features = {}
                    prev_features['camera_feature'] = features['camera_feature_prev_3']
                    prev_features['status_feature'] = features['status_feature_prev_3']

                    self.agent._transfuser_model.prev_keyval_feat = None
                    prev_prediction = self.agent.forward_test(prev_features)

                    prediction = self.agent.forward_test(features)
                    
                else:
                    prediction = self.agent.forward_test(features)

                poses = prediction['trajectory'].cpu().numpy()  # (B, T, 3)
                cls_logits = prediction['cls_logits'].cpu().numpy()
                all_trajectories = prediction['all_trajectories'].cpu().numpy() # (B, K, T, 3)
                tokens = features['token']

                if poses.shape[1] == 40:
                    interval_length = 0.1
                else:
                    interval_length = 0.5
                
                result = {}
                if not self._cfg.use_wm:
                    for (pose,
                        cls_logit,
                        all_trajectory,
                        token) in zip(poses,
                                    cls_logits,
                                    all_trajectories,
                                    tokens):
                        result[token] = {
                            'trajectory': Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
                            'cls_logits': cls_logit,
                            'all_trajectory': all_trajectory
                        }
                else:
                    for (pose,
                        cls_logit,
                        all_trajectory,
                        token) in zip(poses,
                                    cls_logits,
                                    all_trajectories,
                                    tokens):
                        result[token] = {
                            'trajectory': Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
                            'cls_logits': cls_logit,
                            'all_trajectory': all_trajectory
                        }
            elif self._cfg.model_name == "LAW":
                prediction = self.agent.forward_test(features)

                poses = prediction['trajectory'].cpu().numpy()  # (B, T, 3)
                tokens = features['token']
                
                if poses.shape[1] == 40:
                    interval_length = 0.1
                else:
                    interval_length = 0.5
                
                result = {}
                for (pose,
                    token) in zip(poses,
                                tokens):
                    result[token] = {
                        'trajectory': Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
                    }
        return result

            

    def configure_optimizers(self):
        """Inherited, see superclass."""
        self.agent.trainer = self.trainer
        return self.agent.get_optimizers()
