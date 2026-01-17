from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import timm
import time
import math  # 添加math以支持gen_sineembed_for_position中使用

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from navsim.common.enums import StateSE2Index

import torchvision.models as models
import torch.nn.functional as F
import torchvision.models as models
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
import timm
from torch.linalg import inv


class ResNet34Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet34Backbone, self).__init__()
        # Load a pre-trained ResNet-34 model
        resnet = models.resnet34(pretrained=pretrained)
        # resnet=timm.create_model('resnet34', pretrained=False)
        # resnet=TransfuserBackbone(config).resnet34(pretrained=pretrained)
        # Remove the fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        # Extract features from the last convolutional layer
        res = self.backbone(x)
        return res

class W4DModel(nn.Module):
    def __init__(self, 
            config: TransfuserConfig,
            ):
        super().__init__()

        self._config = config
        #pretrained here
        self.image_encoder = timm.create_model(
                    config.image_architecture, pretrained=False, features_only=True
                )
        self.image_encoder.load_state_dict(torch.load(
            '/lpai/ARDrive/ckpts/resnet34-333f7ec4.pth', 
            map_location='cpu', 
            weights_only=False
            ),strict=False)


        self.image_fc = nn.Linear(512, 256)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._num_poses = config.trajectory_sampling.num_poses
        

        #TODO: petr position_embedding
        # num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 20*12 + 1
        num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 256+1
        # num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 256+1
        # num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 120+1

        ############################################多模态相关初始化########################################
        # 增加多模态轨迹嵌入
        self._num_mode = config.num_mode 
        if self._num_mode == 20:
            # 使用diffusiondrive的20模态初始化
            multi_mode_ref = np.load('/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/kmeans_navsim_traj_20.npy')
            self.waypoint_mode_ref = nn.Parameter(
                torch.tensor(multi_mode_ref, dtype=torch.float32), # [20, 8, 2]
                requires_grad=False,
            )
        elif self._num_mode == 18:
            # 使用navtrain聚类的的18模态初始化
            multi_mode_ref = np.load('/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/kmeans_navsim_four_cmd_traj_6.npy')
            self.waypoint_mode_ref = nn.Parameter(
                torch.tensor(multi_mode_ref, dtype=torch.float32), # [3, 6, 8, 3]
                requires_grad=False,
            )
        elif self._num_mode == 4:
            if self._config.use_ar:
                for i in range(self._num_poses):
                    setattr(self, f'waypoint_mode_ref_ar_{i + 1}', nn.Embedding(self._num_mode * (i + 1), config.tf_d_model))  # [num_poses, d_model]
            else:
                # 使用左中右三种命令来指引模态
                self.waypoint_mode_ref = nn.Embedding(self._num_mode * self._num_poses, config.tf_d_model)  # [num_mode * num_poses, d_model]
        # else:
        #     self._mode_embedding = nn.Embedding(self._num_mode * self._num_poses, config.tf_d_model)   # [num_mode * num_poses, d_model]


        # 添加AR的相关编码初始化
        if self._config.use_ar:
            for i in range(self._num_poses):
                setattr(self, f'waypoint_query_ar_{i + 1}', nn.Embedding(self._num_mode * (i + 1), config.tf_d_model))  # [num_poses, d_model]

            if self._num_mode == 4 and self._config.action_chunk_size > 1:
                self.waypoint_query_ar_chunk = nn.Embedding(self._num_mode * self._config.action_chunk_size, config.tf_d_model)  # [num_mode * chunk_size, d_model]
                self.waypoint_mode_ref_ar_chunk = nn.Embedding(self._num_mode * self._config.action_chunk_size, config.tf_d_model)  # [num_mode * chunk_size, d_model]

                self._traj_encoder_ar_chunk = nn.Sequential(
                    nn.Linear(3, config.tf_d_model),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_model, config.tf_d_ffn),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_ffn, config.tf_d_model),
                )
                ar_chunk_wm_decoder_layer = nn.TransformerDecoderLayer(
                    d_model=config.tf_d_model,
                    nhead=config.tf_num_head,
                    dim_feedforward=config.tf_d_ffn,
                    dropout=config.tf_dropout,
                    batch_first=True,
                )
                self._tf_decoder_ar_chunk = nn.TransformerDecoder(ar_chunk_wm_decoder_layer, config.tf_num_layers)

                ar_chunk_wm_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.tf_d_model,
                nhead=config.tf_num_head,
                dim_feedforward=config.tf_d_ffn,
                dropout=config.tf_dropout,
                batch_first=True,
                )
                self.ar_chunk_wm_decoder = nn.TransformerDecoder(ar_chunk_wm_decoder_layer, config.tf_num_layers) # input: Bz, num_token, d_model


        
        # 编码多模态引导信息为嵌入
        self._mode_embedding = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )


        self._keyval_embedding = nn.Embedding(
            num_keyval, config.tf_d_model
        )  # 8x8 feature grid + trajectory

        # 添加历史帧的embedding，暂时只添加一个embedding
        self._history_frame_embedding = nn.Embedding(num_keyval, config.tf_d_model)

        if config.num_mode:
            self._query_embedding = nn.Embedding(self._num_mode * self._num_poses, config.tf_d_model)  # [num_mode * num_poses, d_model]
        else:
            self._query_embedding = nn.Embedding(self._num_poses, config.tf_d_model)
        
        # 时序特征融合
        self.prev_keyval_feat = None
        temp_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        ) 
        
        self._temp_decoder = nn.TransformerDecoder(temp_decoder_layer, config.tf_num_layers)
        
        # 轨迹特征提取
        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)

        # AR的逐步残差预测解码器
        if self._config.use_ar and self._num_mode == 4:
            for i in range(self._num_poses):
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=config.tf_d_model,
                    nhead=config.tf_num_head,
                    dim_feedforward=config.tf_d_ffn,
                    dropout=config.tf_dropout,
                    batch_first=True,
                )
                setattr(self, f'_tf_decoder_ar_{i + 1}', nn.TransformerDecoder(decoder_layer, 1))  # 每步一个decoder,每个decoder 1 layer
                setattr(self, f'traj_encoder_ar_{i + 1}', nn.Sequential(
                    nn.Linear((i + 1) * 3 * self._num_mode, config.tf_d_model),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_model, config.tf_d_ffn),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_ffn, config.tf_d_model),
                ))
            self._traj_encoder_ar = nn.Sequential(
                nn.Linear(3, config.tf_d_model),
                nn.ReLU(),
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )

        if config.num_mode and config.action_chunk_size <= 1:
            self._trajectory_head = TrajectoryHead(self._num_poses, config.tf_d_ffn, config.tf_d_model, config.num_mode)
        elif config.num_mode and config.action_chunk_size > 1:
            self._trajectory_head = TrajectoryHead(self._num_poses, config.tf_d_ffn, config.tf_d_model, config.num_mode, config.action_chunk_size)
        else:
            self._trajectory_head = TrajectoryHead(self._num_poses, config.tf_d_ffn, config.tf_d_model)


        # 多模态词表导入
        if self._config.use_vocab_trajs:
            vocab_trajs_dict_path = self._config.vocab_trajs_dict_path
            vocab_trajs_path = self._config.vocab_trajs_path
            vocab_trajs_dict = pickle.load(open(vocab_trajs_dict_path, 'rb'))
            vocab_trajs = np.load(vocab_trajs_path, allow_pickle=True).item()




        # loss weight
        self.wm_loss_weight= config.wm_loss_weight if hasattr(config, 'wm_loss_weight') else 0.2
        self.traj_loss_weight= config.traj_loss_weight if hasattr(config, 'traj_loss_weight') else 1.0
        self.traj_cls_loss_weight= config.traj_cls_loss_weight if hasattr(config, 'traj_cls_loss_weight') else 0.5

        # ---------------------------------------------------------------------------------------

            

        self.stride=32
        self.use_all_mb = config.use_all_mb if hasattr(config, 'use_all_mb') else False

        if self._config.use_ar_wm:
            num_wm_query = num_keyval   #256+1
            self._wm_query_embedding = nn.Embedding(num_wm_query, config.tf_d_model)
            wm_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.tf_d_model,
                nhead=config.tf_num_head,
                dim_feedforward=config.tf_d_ffn,
                dropout=config.tf_dropout,
                batch_first=True,
            )

            for i in range(self._num_poses):
                setattr(self, f'traj_encoder_ar_{i + 1}_wm', nn.Sequential(
                    nn.Linear((i + 1)*3, config.tf_d_model),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_model, config.tf_d_ffn),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_ffn, config.tf_d_model),
                ))
                setattr(self, f'_wm_decoder_ar_{i + 1}', nn.TransformerDecoder(wm_decoder_layer, 1)) # input: Bz, num_token, d_model
            


        if self._config.use_wm:
            num_wm_query = num_keyval   #256+1
            self._wm_query_embedding = nn.Embedding(num_wm_query, config.tf_d_model)
            self.traj_encoder = nn.Sequential(
                nn.Linear(self._num_poses * 3, config.tf_d_model),
                nn.ReLU(),
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )
            wm_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.tf_d_model,
                nhead=config.tf_num_head,
                dim_feedforward=config.tf_d_ffn,
                dropout=config.tf_dropout,
                batch_first=True,
            )
            self._wm_decoder = nn.TransformerDecoder(wm_decoder_layer, config.tf_num_layers) # input: Bz, num_token, d_model
        #wm
    def ar_wm_prediction(self, cur_img_feat, cur_traj, step):
        batch_size = cur_img_feat.shape[0]
        cur_traj = cur_traj.reshape(batch_size, 1, -1)  # [bs, 1, num_poses*step]
        traj_token_encoder = getattr(self, f'traj_encoder_ar_{step}_wm')
        traj_token = traj_token_encoder(cur_traj)  # [bs, 1, d_model]
        wm_next_latent = torch.zeros_like(cur_img_feat)  # [bs, num_token, d_model]
        input_tokens = torch.cat([cur_img_feat, traj_token], dim=1)  # [bs, num_token+1, d_model]
        wm_query = self._wm_query_embedding.weight[None, ...].repeat(batch_size, 1, 1)  # [bs, num_token, d_model]
        wm_decoder = getattr(self, f'_wm_decoder_ar_{step}')
        wm_next_latent = wm_decoder(wm_query, input_tokens)  # [bs, num_token, d_model]
        return wm_next_latent
    
    def ar_chunk_wm_prediction(self, cur_img_feat, cur_traj, step):
        batch_size = cur_img_feat.shape[0]
        cur_traj = cur_traj.reshape(batch_size, 1, -1)  # [bs, 1, num_poses]

        traj_token = self._traj_encoder_ar_chunk(cur_traj)  # [bs, 1, d_model]
        wm_next_latent = torch.zeros_like(cur_img_feat)  # [bs, num_token, d_model]
        input_tokens = torch.cat([cur_img_feat, traj_token], dim=1)  # [bs, num_token+1, d_model]
        wm_query = self._wm_query_embedding.weight[None, ...].repeat(batch_size, 1, 1)  # [bs, num_token, d_model]
        wm_next_latent = self.ar_chunk_wm_decoder(wm_query, input_tokens)  # [bs, num_token, d_model]
        return wm_next_latent

    def wm_prediction(self, cur_img_feat, cur_traj):
        batch_size = cur_img_feat.shape[0]
        cur_traj = cur_traj.reshape(batch_size, 1, -1)  # [bs, 1, num_poses*3]
        traj_token = self.traj_encoder(cur_traj)  # [bs, 1, d_model]
        wm_next_latent = torch.zeros_like(cur_img_feat)  # [bs, num_token, d_model]
        input_tokens = torch.cat([cur_img_feat, traj_token], dim=1)  # [bs, num_token+1, d_model]
        wm_query = self._wm_query_embedding.weight[None, ...].repeat(batch_size, 1, 1)  # [bs, num_token, d_model]
        wm_next_latent = self._wm_decoder(wm_query, input_tokens)  # [bs, num_token, d_model]
        return wm_next_latent
    ## TODO:需要考虑poses是delta形式
    def _create_temporal_ensemble(self):
        """创建时间集成配置"""
        return {
            'method': 'exponential',  # 可选: 'exponential', 'linear', 'equal'
            'decay_rate': 0.85,       # 指数衰减率
            'min_history': 1,         # 最少需要的历史帧数
            'max_history': self._config.action_chunk_size  # 最多使用的历史帧数
        }

    def _apply_temporal_ensemble(self, current_traj, history_trajs, current_step, 
                            batch_size, num_mode, chunk_size):
        """
        应用时间集成到当前轨迹预测
        
        Args:
            current_traj: 当前步预测的轨迹 [bs, num_mode*chunk_size, 8, 3]
            history_trajs: 历史轨迹列表
            current_step: 当前时间步
            batch_size: batch大小
            num_mode: 模态数量
            chunk_size: 块大小
        """
        ensemble_config = self._create_temporal_ensemble()
        
        # 收集相关历史轨迹
        related_trajs = self._collect_related_trajectories(
            history_trajs, current_step, ensemble_config['max_history']
        )
        
        if len(related_trajs) < ensemble_config['min_history']:
            return current_traj
        
        # 计算时间集成权重
        weights = self._compute_temporal_weights(
            len(related_trajs), ensemble_config
        )
        
        # 应用时间集成
        integrated_traj = self._integrate_trajectories(
            current_traj, related_trajs, weights, 
            batch_size, num_mode, chunk_size
        )
        
        return integrated_traj

    def _collect_related_trajectories(self, history_trajs, current_step, max_history):
        """收集相关历史轨迹"""
        related_trajs = []
        # 收集最多max_history个历史轨迹
        for past_idx in range(max_history):
            if current_step - past_idx - 2 >= 0: # 只要历史预测的轨迹 cur_step form 1-8
                related_trajs.append(history_trajs[current_step - past_idx - 2])
        
        # 反转顺序，使最近的轨迹在最后
        related_trajs.reverse()
        return related_trajs

    def _compute_temporal_weights(self, num_trajs, ensemble_config):
        """计算时间集成权重"""
        method = ensemble_config['method']
        decay_rate = ensemble_config['decay_rate']
        if num_trajs == 0:
            return None
        ## 存在历史轨迹后，需要把当前轨迹预测的权重也加上，认为当前预测的权重最大
        num_trajs += 1
        if method == 'exponential':
            # 指数衰减权重：越新的轨迹权重越大
            
            weights = [decay_rate ** i for i in range(num_trajs)]
            weights = weights[::-1]  # 反转，使最近的权重最大
        elif method == 'linear':
            # 线性衰减权重
            weights = np.linspace(1.0, 0.1, num_trajs)
        elif method == 'equal':
            # 等权重
            weights = np.ones(num_trajs)
        else:
            weights = np.ones(num_trajs)
        
        # 归一化
        weights = np.array(weights) / np.sum(weights)
        return torch.tensor(weights, dtype=torch.float32, device=self._keyval_embedding.weight.device)

    def _integrate_trajectories(self, current_traj, related_trajs, weights, 
                            batch_size, num_mode, chunk_size):
        """集成轨迹"""
        # 重塑当前轨迹: [bs, num_mode*chunk_size, 3] -> [bs, num_mode, chunk_size, 3]
        current_traj_reshaped = current_traj.reshape(
            batch_size, num_mode, chunk_size, 3
        )
        
        # 只对第一个waypoint进行时间集成（立即执行的动作）
        integrated_first_waypoint = torch.zeros_like(current_traj_reshaped[:, :, 0, :])

        # 添加当前预测（权重最大）
        integrated_first_waypoint += weights[0] * current_traj_reshaped[:, :, 0, :]

        # 集成历史轨迹
        num_his_traj = len(related_trajs)
        for i, hist_traj in enumerate(related_trajs):
            if i < len(weights) - 1:  # 权重[0]已经用于当前预测
                hist_traj_reshaped = hist_traj.reshape(
                    batch_size, num_mode, chunk_size, 3
                )
                # 使用历史轨迹中对应的waypoint
                integrated_first_waypoint += weights[i + 1] * hist_traj_reshaped[:, :, num_his_traj - i, :]
        
        # 用集成后的第一个waypoint替换原来的第一个waypoint
        current_traj_reshaped[:, :, 0, :] = integrated_first_waypoint

        # 重塑回原始形状
        integrated_traj = current_traj_reshaped.reshape(
            batch_size, num_mode * chunk_size, 3
        )
        
        return integrated_traj
    def gen_sineembed_for_position(self, pos_tensor, hidden_dim=256):
        """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
        """
        half_hidden_dim = hidden_dim // 2
        scale = 2 * math.pi
        dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
        x_embed = pos_tensor[..., 0] * scale
        y_embed = pos_tensor[..., 1] * scale
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1)
        return pos

    def _delta_to_absolute(self, delta_traj: torch.Tensor) -> torch.Tensor:
        # delta_traj: [bs, num_mode, num_poses, 3]，车体系增量
        dheading = delta_traj[..., 2]
        heading = torch.cumsum(dheading, dim=2)
        heading = (heading + math.pi) % (2 * math.pi) - math.pi

        heading_prev = torch.zeros_like(heading)
        heading_prev[..., 1:] = heading[..., :-1]

        dx_body = delta_traj[..., 0]
        dy_body = delta_traj[..., 1]
        cos_h = torch.cos(heading_prev)
        sin_h = torch.sin(heading_prev)
        dx_world = dx_body * cos_h - dy_body * sin_h
        dy_world = dx_body * sin_h + dy_body * cos_h

        x = torch.cumsum(dx_world, dim=2)
        y = torch.cumsum(dy_world, dim=2)

        return torch.stack((x, y, heading), dim=-1)

    def _forward_core(self, features, is_test=False):
        """
        将 forward_train/forward_test 中重复的处理提取到这里，
        返回 trajectory dict 和 keyval_final 供上层方法继续使用。
        """
        camera_feature = features['camera_feature']
        status_feature = features['status_feature']

        cmd = status_feature[..., :4]  # [bs, 4] one-hot
        cmd = torch.argmax(cmd, dim=-1)  # [bs] int

        batch_size = status_feature.shape[0]
        device = camera_feature.device

        # =============================== img backbone =======================================
        img_feat = self.image_encoder(camera_feature)[-1]   # [bs, 512, 8, 32]
        img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)    # [bs, 8*32, 512]
        img_feat = self.image_fc(img_feat.clone())  # -> [bs, 8*32, 256]

        # =============================== keyval trans =======================================
        status_encoding = self._status_encoding(status_feature)  # [bs, 256]
        keyval = torch.cat([img_feat, status_encoding[:, None]], dim=1)  # [bs, 256+1, 256]
        keyval = keyval.clone() + self._keyval_embedding.weight[None, ...]

        # 添加历史信息，进行时序特征融合
        ##############################################先使用间隔1帧进行测试########################################
        # camera_feature_prev = features['camera_feature_prev_2']
        # status_feature_prev = features['status_feature_prev_2']
        # cmd_prev = status_feature_prev[..., :4]  # [bs, 4] one-hot
        # cmd_prev = torch.argmax(cmd_prev, dim=-1)  # [bs] int

        # img_feat_prev = self.image_encoder(camera_feature_prev)[-1]
        # img_feat_prev = img_feat_prev.flatten(-2, -1).permute(0, 2, 1)
        # img_feat_prev = self.image_fc(img_feat_prev.clone()) # 512 -> 256 # [bs, 8*32, 256]
        # status_encoding_prev = self._status_encoding(status_feature_prev)
        # keyval_prev = torch.cat([img_feat_prev, status_encoding_prev[:, None]], dim=1)
        # keyval_prev = keyval_prev.clone() + self._keyval_embedding.weight[None, ...]  # [bs, 256+1, 256]

        # ##############################################先使用间隔1帧进行测试########################################

        # keyval_final = torch.zeros_like(keyval)

        # keyval_final = self._temp_decoder(keyval, keyval_prev)  # [bs, 256+1, 256]

        # keyval_final = keyval   # [bs, 64, 257 , 256]

        # =============================== query =============================================

        
        keyval_final = keyval   # [bs, 64, 257 , 256]


        # 使用action chunk的形式进行预测，每次预测action_chunk_size个waypoint，然后滑动窗口进行自回归的预测
        if self._config.use_ar and self._num_mode == 4 and self._config.action_chunk_size > 1:
            kv = keyval_final
            all_cmd_temp_traj = []
            all_temp_traj = []
            all_pred_next_latent = []
            
            all_next_latent = []
            all_step_ensembled_traj = []


            for i in range(1, self._num_poses + 1):
                q = self.waypoint_query_ar_chunk.weight.unsqueeze(0).repeat(batch_size, 1, 1)
                m = self.waypoint_mode_ref_ar_chunk.weight.unsqueeze(0).repeat(batch_size, 1, 1)
                m_feat = self._mode_embedding(m)
                ego_q = q + m_feat
                # Transformer 解码并生成所有模态轨迹

                out = self._tf_decoder_ar_chunk(ego_q, kv)
                last_trajectory = self._trajectory_head.forward_ar_chunk(out, cmd=cmd)
                all_traj = last_trajectory['all_trajectories']  # [bs, num_mode*chunk_size, 3]

                # 应用时间集成到当前预测 TODO:需要把poses改成delta形式，否则怎么对齐轨迹的位置？
                if i > 1 and len(all_cmd_temp_traj) > 0:
                    all_traj = self._apply_temporal_ensemble(
                        all_traj, all_cmd_temp_traj, i, 
                        batch_size, self._num_mode, self._config.action_chunk_size
                    )

                # 使用下一个chunk对应的图像特征进行轨迹预测，将本步的预测wm和真实的wm进行随机替换，提升鲁棒性
                ## 只在当前动作来预测下帧的wm，同时预测_num_mode个wm特征
                cur_action = all_traj.reshape(batch_size, self._num_mode, self._config.action_chunk_size, 3)  # [bs, num_mode, chunk_size, 3]
                cur_action_first = cur_action[:, :, 0, :].reshape(batch_size, self._num_mode * 1, 3)  # [bs, num_mode*1, 3]
                all_step_ensembled_traj.append(cur_action_first)

                all_cmd_temp_traj.append(all_traj)
                # all_temp_traj.append(last_trajectory['trajectory'])
                cur_pred_next_latent = []
                if self._config.use_ar_wm:
                    for wm_idx in range(self._num_mode):
                        # embed = self._traj_encoder_ar_chunk(cur_action_first[:, wm_idx, :]) # [bs, d_model]
                        # cur_kv = torch.cat([kv, embed.unsqueeze(1)], dim=1)  # [bs, num_token+1 +1, d_model]
                        cur_cmd_wm_next_latent = self.ar_chunk_wm_prediction(kv, cur_action_first[:, wm_idx, :], i)
                        cur_pred_next_latent.append(cur_cmd_wm_next_latent)
                    wm_next_latent = torch.stack(cur_pred_next_latent, dim=1)  # [bs, num_mode, num_token+1, d_model]
                    ### 选出对应cmd的wm_next_latent
                    true_cmd_wm_next_latent = wm_next_latent[torch.arange(batch_size), cmd, :, :]  # [bs, num_token + 1, d_model]
                    all_pred_next_latent.append(wm_next_latent)

                    # 训练时使用真实的wm_next_latent进行替换
                    if not is_test:
                        ## 计算真实的wm_next_latent供loss使用
                        with torch.no_grad():
                            camera_feature_next = features[f'camera_feature_next_{i}']
                            status_feature_next = features[f'status_feature_next_{i}']
                            img_feat_next = self.image_encoder(camera_feature_next)[-1]
                            img_feat_next = img_feat_next.flatten(-2, -1).permute(0, 2, 1)
                            img_feat_next = self.image_fc(img_feat_next.clone())
                            status_encoding_next = self._status_encoding(status_feature_next)
                            keyval_next = torch.cat([img_feat_next, status_encoding_next[:, None]], dim=1)
                            keyval_next = keyval_next.clone() + self._keyval_embedding.weight[None, ...]    # [bs, 256+1, 256]

                        all_next_latent.append(keyval_next)


                ## 存在ar_wm_replace_prob概率使得kv是预测的wm feature，而不是真实图像的feature，提升鲁棒性
                
                if torch.rand(1).item() < self._config.ar_wm_replace_prob and not is_test:
                    kv = true_cmd_wm_next_latent
                elif not is_test:
                    kv = keyval_next

                if is_test:
                    # 暂时先全部使用真实的wm feature
                    kv = true_cmd_wm_next_latent

                    # with torch.no_grad():
                    #     camera_feature_next = features[f'camera_feature_next_{i}']
                    #     status_feature_next = features[f'status_feature_next_{i}']
                    #     img_feat_next = self.image_encoder(camera_feature_next)[-1]
                    #     img_feat_next = img_feat_next.flatten(-2, -1).permute(0, 2, 1)
                    #     img_feat_next = self.image_fc(img_feat_next.clone())
                    #     status_encoding_next = self._status_encoding(status_feature_next)
                    #     keyval_next = torch.cat([img_feat_next, status_encoding_next[:, None]], dim=1)
                    #     keyval_next = keyval_next.clone() + self._keyval_embedding.weight[None, ...]    # [bs, 256+1, 256]

                    # all_next_latent.append(keyval_next)
                    # kv = keyval_next

            # 需要将chunk形式的轨迹转换为完整形式的轨迹输出
            trajectory = last_trajectory
            ensembled_traj = torch.stack(all_step_ensembled_traj, dim=2).reshape(batch_size, self._num_mode, self._num_poses, 3)  # [bs, num_mode , num_poses, 3]
            # 将当前的delta形式轨迹转换为绝对位置形式轨迹
            ensembled_traj = self._delta_to_absolute(ensembled_traj)
            ### TODO:使用一个mlp去把delta轨迹转换为绝对位置轨迹会不会更好一些？
            ## 选出当前cmd对应的轨迹输出
            trajectory['trajectory'] = ensembled_traj[torch.arange(batch_size), cmd, :, :]  # [bs, num_poses, 3]


            trajectory['all_cmd_step_trajectories'] = all_step_ensembled_traj  # list of [bs, num_mode, 3]
            trajectory['all_trajectories'] = ensembled_traj  # [bs, num_mode, num_poses, 3]
            if self._config.use_ar_wm:
                trajectory['all_pred_next_latent'] = all_pred_next_latent  # list of [bs, num_mode, num_token, d_model]
                trajectory['all_next_latent'] = all_next_latent  # list of [bs, num_token, d_model]
        else:
            ego_query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)  # [bs, num_mode * num_poses, 256] 或 [bs, num_poses, 256]
            # 使用左中右三种命令来指引模态
            mode_ref = self.waypoint_mode_ref.weight.reshape(self._num_mode, self._num_poses, self._config.tf_d_model)  # [num_mode, num_poses, 256]
            mode_ref = mode_ref.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, num_mode, num_poses, 256]
            mode_ref_feat = self._mode_embedding(mode_ref)  # [bs, num_mode, num_poses, 256]
            ego_query = ego_query.clone() + mode_ref_feat.reshape(batch_size, -1, mode_ref_feat.shape[-1]).clone()
            ego_query_out = self._tf_decoder(ego_query, keyval_final)
            trajectory = self._trajectory_head(ego_query_out, cmd=cmd)

        ##==============================轨迹特征提取========================================

        trajectory['cmd_gt'] = cmd
        
        return trajectory, keyval_final

    def forward_test(self, features) -> Dict[str, torch.Tensor]:
        # 直接调用公共核心逻辑，忽略 world model
        trajectory, _ = self._forward_core(features, is_test=True)
        return trajectory

    def forward_train(self, features) -> Dict[str, torch.Tensor]:
        # 先调用公共核心逻辑
        trajectory, keyval_final = self._forward_core(features)
        # 再添加 world model 预测（如果需要）
        if self._config.use_wm:
            wm_next_latent = self.wm_prediction(keyval_final, trajectory['trajectory'])
            trajectory['pred_next_latent'] = wm_next_latent
            trajectory['cur_latent'] = keyval_final
        return trajectory

    # the loss function for world model
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        logging_prefix: str = None,
    ) -> torch.Tensor:
        loss_dict = {}
        
        B = predictions['cmd_gt'].shape[0]


        # 轨迹损失
        if self._config.action_chunk_size > 1 and self._config.use_ar:
            trajectory_loss = torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])
            loss_dict["traj_loss"] = trajectory_loss * self.traj_loss_weight

        # 世界模型损失
        if self._config.use_ar_wm and 'all_pred_next_latent' in predictions:
            all_pred_next_latent = predictions.get('all_pred_next_latent', [])  # list of [bs, num_token, d_model]
            ## 选出对应cmd的next_latent
            cmd = predictions['cmd_gt']
            all_pred_next_latent = [pred[torch.arange(B), cmd, :, :] for pred in all_pred_next_latent]  # list of [bs, num_token, d_model]
            all_next_latent = predictions.get('all_next_latent', [])  # list of [bs, num_token, d_model]
            for i, pred_latent in enumerate(all_pred_next_latent):
                # 只计算前面7步
                # if i >= 7:
                #     break
                next_latent = all_next_latent[i].detach()
                wm_loss_ar = torch.nn.functional.mse_loss(pred_latent, next_latent)
                # 希望随着步数增加，这个loss的权重降低，以免过分影响前几步的预测，因为后续步数本身预测难度就大
                loss_dict[f'ar_wm_loss_step_{i+1}'] = wm_loss_ar * self.wm_loss_weight

            

        return loss_dict

class TrajectoryHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, num_mode: int = None, chunk_size: int = None):
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self._chunk_size = chunk_size

        # mode
        self._num_mode = num_mode
        # train head
        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, StateSE2Index.size()),
        )
        # cls head
        self._cls_mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, 1),
        )
        # AR head
        for i in range(self._num_poses):
            setattr(self, f'_mlp_ar_{i + 1}', nn.Sequential(
                nn.Linear(self._d_model, self._d_ffn),
                nn.ReLU(),
                nn.Linear(self._d_ffn, StateSE2Index.size()),
            ))
    def forward_ar(self, object_queries, cmd=None) -> Dict[str, torch.Tensor]:
        i = object_queries.shape[1] // self._num_mode  # 当前步数
        poses = getattr(self, f'_mlp_ar_{i}')(object_queries).reshape(-1, self._num_mode, i, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi

        cls_feature = object_queries.reshape(-1, self._num_mode, i, self._d_model).mean(2)  # [bs, num_mode, d_model]
        cls_score = self._cls_mlp(cls_feature).squeeze(-1)  # [bs, num_mode]
        if self._num_mode == 4:
            # cmd==3 表示未指定命令，
            # use_pred = (cmd == 3)
            # # 如果存在命令3，则打印警告
            # if use_pred.any():
            #     print("Warning: Using predicted command for some samples in TrajectoryHead.")
            batch_indices = torch.arange(poses.shape[0], device=poses.device)
            best_traj = poses[batch_indices, cmd]
            return {
                "trajectory": best_traj,
                "cls_logits": cls_score,
                "all_trajectories": poses,
            }
        
    def forward_ar_chunk(self, object_queries, cmd=None) -> Dict[str, torch.Tensor]:
        
        poses = self._mlp(object_queries).reshape(-1, self._num_mode, self._chunk_size, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        ## TODO:需要把poses改成delta形式

        cls_feature = object_queries.reshape(-1, self._num_mode, self._chunk_size, self._d_model).mean(2)  # [bs, num_mode, d_model]
        cls_score = self._cls_mlp(cls_feature).squeeze(-1)  # [bs, num_mode]
        if self._num_mode == 4:
            # cmd==3 表示未指定命令，
            # use_pred = (cmd == 3)
            # # 如果存在命令3，则打印警告
            # if use_pred.any():
            #     print("Warning: Using predicted command for some samples in TrajectoryHead.")
            batch_indices = torch.arange(poses.shape[0], device=poses.device)
            best_traj = poses[batch_indices, cmd]
            return {
                "trajectory": best_traj,
                "cls_logits": cls_score,
                "all_trajectories": poses,
            }

    def forward(self, object_queries, cmd=None) -> Dict[str, torch.Tensor]:
        if self._num_mode:
            poses = self._mlp(object_queries).reshape(-1, self._num_mode, self._num_poses, StateSE2Index.size())

            poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi

            cls_feature = object_queries.reshape(-1, self._num_mode, self._num_poses, self._d_model).mean(2)  # [bs, num_mode, d_model]
            cls_score = self._cls_mlp(cls_feature).squeeze(-1)  # [bs, num_mode]

            if self._num_mode == 20:
                best_traj_idx = torch.argmax(cls_score, dim=-1)
                batch_indices = torch.arange(poses.shape[0], device=poses.device)
                best_traj = poses[batch_indices, best_traj_idx]
                
                return {"trajectory": best_traj, 
                        "cls_logits": cls_score, 
                        "all_trajectories": poses,}
            elif self._num_mode == 18 and cmd is not None:
                # 根据命令选择对应的模式组
                # cmd: [bs], 0:left, 1:straight, 2:right
                mode_per_cmd = self._num_mode // 3  # 每个命令对应的模式数量
                start_idx = (cmd * mode_per_cmd).long()  # [bs]
                idx_offset = torch.arange(mode_per_cmd, device=poses.device)
                mode_indices = (start_idx[:, None] + idx_offset[None, :]).long()  # [bs, mode_per_cmd]
                expanded_idx = mode_indices[:, :, None, None].expand(-1, -1, self._num_poses, poses.size(-1))
                selected_poses = torch.gather(poses, 1, expanded_idx)  # [bs, mode_per_cmd, num_poses, 3]
                selected_cls_score = torch.gather(cls_score, 1, mode_indices)  # [bs, mode_per_cmd]
                best_traj_idx = torch.argmax(selected_cls_score, dim=-1)  # [bs]
                batch_indices = torch.arange(poses.shape[0], device=poses.device)
                best_traj = selected_poses[batch_indices, best_traj_idx]  # [bs, num_poses, 3]
        
                return {
                    "trajectory": best_traj,
                    "cls_logits": selected_cls_score,
                    "all_trajectories": selected_poses,
                    "all_cmd_trajectories": poses,  # 所有模式的轨迹
                }
            elif self._num_mode == 4 :
                # cmd==3 表示未指定命令，
                # use_pred = (cmd == 3)
                # # 如果存在命令3，则打印警告
                # if use_pred.any():
                #     print("Warning: Using predicted command for some samples in TrajectoryHead.")
                batch_indices = torch.arange(poses.shape[0], device=poses.device)
                best_traj = poses[batch_indices, cmd]
                return {
                    "trajectory": best_traj,
                    "cls_logits": cls_score,
                    "all_trajectories": poses,
                }

        else:
            poses = self._mlp(object_queries)  # [bs, num_poses, 3]
            poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
            return {"trajectory": poses}

