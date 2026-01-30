from tkinter import NO
from typing import Dict
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
from navsim.agents.transfuser.temporal_world_model import TemporalWorldModel

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
        # pretrained here
        # self.image_encoder = timm.create_model(
        #             config.image_architecture, pretrained=False, features_only=True
        #         )
        # self.image_encoder.load_state_dict(torch.load(
        #     '/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/resnet34-333f7ec4.pth', 
        #     map_location='cpu', 
        #     weights_only=False
        #     ),strict=False)
        
        # vision learnable query
        # 加入时序: [1, 3, 256, 1024] -> [1, 12, 3, 256, 1024]
        self.vision_query = nn.Parameter(
            torch.randn(1, config.num_frames, config.num_views, 256, config.tf_d_model),
            requires_grad=True,
        )  # [1, 3, 256, 1024] 表示对应 3 个视角的可学习 vision query
        
        # define transformer decoder for vision query
        self.vision_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        
        self.vision_decoder = nn.TransformerDecoder(self.vision_decoder_layer, 3)

        # self.image_fc = nn.Linear(512, 256)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._num_poses = config.trajectory_sampling.num_poses
        #TODO: petr position_embedding
        # num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 20*12 + 1
        num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 768+1
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
        elif self._num_mode == 3:
            if self._config.use_cmd_embed:
                # 使用左中右三种命令来指引模态 + 命令嵌入
                self.waypoint_mode_ref = nn.Embedding(self._num_mode, config.tf_d_model)  # 4 commands
            else:
                # 使用左中右三种命令来指引模态
                self.waypoint_mode_ref = nn.Embedding(self._num_mode * self._num_poses, config.tf_d_model)  # [num_mode * num_poses, d_model]
        elif self._num_mode == 4:
            if self._config.use_cmd_embed:
                # 使用左中右三种命令来指引模态 + 命令嵌入
                self.waypoint_mode_ref = nn.Embedding(self._num_mode, config.tf_d_model)  # 4 commands
            else:
                # 使用左中右三种命令来指引模态
                self.waypoint_mode_ref = nn.Embedding(self._num_mode * self._num_poses, config.tf_d_model)  # [num_mode * num_poses, d_model]
        # else:
        #     self._mode_embedding = nn.Embedding(self._num_mode * self._num_poses, config.tf_d_model)   # [num_mode * num_poses, d_model]

        # 编码多模态引导信息为嵌入
        self._mode_embedding = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )

        ############################################多模态相关初始化########################################

        # 控制命令预测头
        self._cmd_pred_query = nn.Embedding(1, config.tf_d_model)  # single query

        self._cmd_head_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._cmd_head_decoder = nn.TransformerDecoder(self._cmd_head_decoder_layer, 1)

        self._cmd_mlp = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, 4),  # 4 classes
        )


        self._keyval_embedding = nn.Embedding(
            num_keyval, config.tf_d_model
        )  # 8x8 feature grid + trajectory
        if config.num_mode:
            self._query_embedding = nn.Embedding(self._num_mode * self._num_poses, config.tf_d_model)  # [num_mode * num_poses, d_model]
        else:
            self._query_embedding = nn.Embedding(self._num_poses, config.tf_d_model)
        
        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)

        if config.num_mode:
            self._trajectory_head = TrajectoryHead(self._num_poses, config.tf_d_ffn, config.tf_d_model, config.num_mode)

            self.refine_block = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.tf_d_model,
                    nhead=config.tf_num_head,
                    dim_feedforward=config.tf_d_ffn,
                    dropout=config.tf_dropout,
                    batch_first=True,
                ), config.tf_num_layers
            )

            self.refine_traj_head = TrajectoryHead(self._num_poses, config.tf_d_ffn, config.tf_d_model, config.num_mode)
        else:
            self._trajectory_head = TrajectoryHead(self._num_poses, config.tf_d_ffn, config.tf_d_model)


        # 多模态词表导入
        if self._config.use_vocab_trajs:
            vocab_trajs_dict_path = self._config.vocab_trajs_dict_path
            vocab_trajs_path = self._config.vocab_trajs_path
            vocab_trajs_dict = pickle.load(open(vocab_trajs_dict_path, 'rb'))
            vocab_trajs = np.load(vocab_trajs_path, allow_pickle=True).item()




        # loss weight
        self.wm_loss_weight = 0.2
        self.traj_loss_weight= config.traj_loss_weight if hasattr(config, 'traj_loss_weight') else 1.0
        self.traj_cls_loss_weight= config.traj_cls_loss_weight if hasattr(config, 'traj_cls_loss_weight') else 0.5
        # 多样性损失权重与阈值（终点距离<margin即惩罚）
        self.diversity_loss_weight = getattr(config, 'diversity_loss_weight', 0.1)
        self.diversity_margin = getattr(config, 'diversity_margin', 1.0)  # 1m

        # ---------------- Soft responsibility (multi-modal supervision) settings ----------------
        self.soft_resp = getattr(config, 'soft_resp', False)  # 是否启用软责任
        self.soft_resp_tau = getattr(config, 'soft_resp_tau', 1.0)  # 温度
        self.soft_resp_topk = getattr(config, 'soft_resp_topk', 0)  # 0=全部, >0 只取距离最小的K个模式
        self.cls_entropy_weight = getattr(config, 'cls_entropy_weight', 0.0)  # 分类熵正则权重
        self.use_fde_for_weight = getattr(config, 'use_fde_for_weight', True)  # 用FDE(终点)还是ADE加权
        # ---------------------------------------------------------------------------------------

               
        # add world model here
        self.use_wm = self._config.use_wm
        self.stride=32
        self.use_all_mb = config.use_all_mb if hasattr(config, 'use_all_mb') else False

        if self.use_wm:
            num_wm_query = (self._config.num_frames-1) * (num_keyval - 1)
            self._wm_query_embedding = nn.Embedding(num_wm_query, config.tf_d_model)
            self.temporal_world_model = TemporalWorldModel(config, use_4d_rope=True)

    def forward_test(self, features) -> Dict[str, torch.Tensor]:
        camera_feature = features['camera_feature']
        num_frames = self._config.num_frames
        all_frames_camera_feature = torch.stack(
            [
                features['camera_feature_prev_3'],
                # features['camera_feature_prev_2'],
                # features['camera_feature_prev_1'],
                features['camera_feature'],
                # features['camera_feature_next_1'],
                # features['camera_feature_next_2'],
                # features['camera_feature_next_3'],
                features['camera_feature_next_4'],
                # features['camera_feature_next_5'],
                # features['camera_feature_next_6'],
                # features['camera_feature_next_7'],
                features['camera_feature_next_8'],
            ], dim=1
        )
        b, n, seq_len, dim = camera_feature.shape

        status_feature = features['status_feature']
        all_frames_status_feature = torch.stack(
            [
                features['status_feature_prev_3'],
                # features['status_feature_prev_2'],
                # features['status_feature_prev_1'],
                status_feature,
                # features['status_feature_next_1'],
                # features['status_feature_next_2'],
                # features['status_feature_next_3'],
                features['status_feature_next_4'],
                # features['status_feature_next_5'],
                # features['status_feature_next_6'],
                # features['status_feature_next_7'],
                features['status_feature_next_8'],
            ], dim=1
        )

        cmd = all_frames_status_feature[..., :2]  # [bs, num_frames, 4] one-hot
        cmd = torch.argmax(cmd, dim=-1)  # [bs, num_frames] int


        batch_size = status_feature.shape[0]
        device= camera_feature.device

        # img_feat = self.image_encoder(camera_feature)[-1]
        # img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)
        # img_feat = self.image_fc(img_feat.clone()) # 512 -> 256 # [bs, 8*32, 256]
        
        # # v1: concat 3 views in sequence length dimension  [bs, 3, 256, 1024] -> [bs, 768, 1024]
        # img_feat = camera_feature[:, :, 1:, :].reshape(b, -1, dim)
        
        # v2: use learnable query do cross attention with each view dino feature
        init_view_query_feat = self.vision_query.repeat(batch_size, 1, 1, 1, 1).reshape(-1, seq_len-1, dim)  # [1, 12, 3, 256, 1024] -> [bs, 12, 3, 256, 1024]
        spatial_view_feat = self.vision_decoder(init_view_query_feat, all_frames_camera_feature[:, :, :, 1:, :].reshape(-1, seq_len-1, dim))
            
        spatial_view_feat = spatial_view_feat.reshape(b, num_frames, -1, dim)  # [b, num_frames, 768, 1024]

        # =============================== keyval trans =======================================
        status_encoding = self._status_encoding(all_frames_status_feature)  # [bs, frame, 8] -> [bs, 12, 256]
        keyval = torch.cat([spatial_view_feat, status_encoding[:, :, None]], dim=-2)  # [bs, 12, 768+1, 256]
        keyval = keyval.clone() + self._keyval_embedding.weight[None, None, ...]

        keyval_final = keyval   # [bs, 12, 768+1 , 256]


        ego_query = self._query_embedding.weight[None, None, ...].repeat(batch_size, num_frames, 1, 1)   # [bs, num_mode * num_poses, 256]

        # 多模态引导，与 forward_train 保持一致
        if self._config.num_mode:
            ##使用anchor去引导多模态轨迹
            if self._num_mode == 20:
                # diffusiondrive的20模态
                mode_ref = self.waypoint_mode_ref.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, num_mode, num_poses, 2]
            elif self._num_mode == 18:
                mode_ref = self.waypoint_mode_ref.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)[..., :2]  # [bs, 3, 6, 8, 2]
            elif self._num_mode == 3:
                # 使用左中右三种命令来指引模态
                mode_ref = self.waypoint_mode_ref.weight # [num_mode, 256]
                if self._config.use_cmd_embed:
                    mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, self._num_poses, 1)  # [bs, num_mode, num_poses, 256]
                else:
                    mode_ref = mode_ref.reshape(self._num_mode, self._num_poses, self._config.tf_d_model)  # [num_mode, num_poses, 256]
                    mode_ref = mode_ref.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, num_mode, num_poses, 256]           
            elif self._num_mode == 4:
                # 使用左中右三种命令来指引模态
                mode_ref = self.waypoint_mode_ref.weight # [num_mode, 256]
                if self._config.use_cmd_embed:
                    mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, self._num_poses, 1)  # [bs, num_mode, num_poses, 256]
                else:
                    mode_ref = mode_ref.reshape(self._num_mode, self._num_poses, self._config.tf_d_model)  # [num_mode, num_poses, 256]
                    mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(1).repeat(batch_size, num_frames, 1, 1, 1)  # [bs, num_frames, num_mode, num_poses, 256]
            else:
                mode_ref = self.waypoint_mode_ref.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, num_mode, num_poses, 2]
            
            if self._num_mode in [20, 18]:
                mode_ref_feat_sin = self.gen_sineembed_for_position(mode_ref)  # [bs, num_mode, num_poses, 256]
                mode_ref_feat = self._mode_embedding(mode_ref_feat_sin)  # [bs, num_mode, num_poses, 256]
            else:
                mode_ref_feat = self._mode_embedding(mode_ref)  # [bs, num_frames, num_mode, num_poses, 256]
            ego_query = ego_query.clone() + mode_ref_feat.reshape(batch_size, num_frames, -1, mode_ref_feat.shape[-1]).clone()



        
        ##==============================轨迹特征提取========================================
        ego_query = ego_query.reshape(-1, ego_query.shape[-2], ego_query.shape[-1])
        keyval_final = keyval_final.reshape(-1, keyval_final.shape[-2], keyval_final.shape[-1])
        ego_query_out = self._tf_decoder(ego_query, keyval_final)

        ##==============================轨迹特征提取========================================

       

        ##==============================控制命令预测==============================================
        cmd_query = self._cmd_pred_query.weight[None, None, ...].repeat(batch_size, num_frames, 1, 1)  # [bs, num_frames, 1, 256]
        cmd_query = cmd_query.reshape(-1, cmd_query.shape[-2], cmd_query.shape[-1])
        cmd_query_out = self._cmd_head_decoder(cmd_query, keyval_final)  #

        cmd_logits = self._cmd_mlp(cmd_query_out).squeeze(1)  # [bs*num_frames, 4]
        # 取一个除了3外，最大概率的类别作为预测值
        # cmd_pred = torch.argmax(cmd_logits[:, :3], dim=-1)  # [bs]
        cmd_pred = torch.argmax(cmd_logits, dim=-1)  # [bs*num_frames]

        # wm
        if self.use_wm:
            trajectory = {}
            trajectory['first_traj'] = self._trajectory_head(ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...], cmd=cmd[:, 1])

            trajectory['cmd_logits'] = cmd_logits.reshape(batch_size, num_frames, -1)[:, 1, ...].squeeze(1)
            trajectory['cmd_pred'] = cmd_pred.reshape(batch_size, num_frames)[:, 1]
            trajectory['cmd_gt'] = cmd.reshape(batch_size, num_frames)[:, 1]
            
            wm_keyval = spatial_view_feat[:, :2, ...]
            # auto-regressive from frame-5 to frame-12 (8 frames)
            for current_frame in range(2, num_frames):
                n_tokens_4d = current_frame * self._config.num_views * 16 * 16
                attention_mask = self.temporal_world_model.get_attention_mask(n_tokens_4d, self._config.num_views * 16 * 16).to(device)
                current_keyval = wm_keyval[:, :current_frame, ...].reshape(batch_size, -1, dim)
                current_query = self._wm_query_embedding.weight[None, :current_frame*768, :].repeat(batch_size, 1, 1).reshape(batch_size, -1, dim)
                wm_next_latent = self.temporal_world_model.forward(
                                                                query=current_query,
                                                                keyval=current_keyval,
                                                                time_frames=current_frame,
                                                                views=self._config.num_views,
                                                                height=16,
                                                                width=16,
                                                                tgt_mask=attention_mask,
                                                                memory_mask=attention_mask
                                                            )
                wm_next_latent = wm_next_latent.reshape(batch_size, current_frame, self._config.num_views, -1, dim)
                wm_keyval = wm_keyval.reshape(batch_size, current_frame, self._config.num_views, -1, dim)
                wm_keyval = torch.cat([wm_keyval, wm_next_latent[:, -1, ...].unsqueeze(1)], dim=1)

            trajectory['wm_next_latent']=wm_next_latent
            
            # refine
            refine_query = ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...]
            refine_keyval = wm_next_latent.reshape(batch_size, num_frames-1, -1, dim)[:, -1, ...]
            refine_query_out = ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...] + self.refine_block(refine_query, refine_keyval)
            
            # TODO: comupute the refined trajectory with residual
            trajectory['refined_traj'] = self.refine_traj_head(refine_query_out, cmd=cmd[:, 1])
            for key in trajectory['first_traj']:
                trajectory['refined_traj'][key] += trajectory['first_traj'][key]
        return trajectory

    def forward_train(self, features) -> Dict[str, torch.Tensor]:
        camera_feature = features['camera_feature']
        num_frames = self._config.num_frames
        all_frames_camera_feature = torch.stack(
            [
                features['camera_feature_prev_3'],
                # features['camera_feature_prev_2'],
                # features['camera_feature_prev_1'],
                features['camera_feature'],
                # features['camera_feature_next_1'],
                # features['camera_feature_next_2'],
                # features['camera_feature_next_3'],
                features['camera_feature_next_4'],
                # features['camera_feature_next_5'],
                # features['camera_feature_next_6'],
                # features['camera_feature_next_7'],
                features['camera_feature_next_8'],
            ], dim=1
        )
        b, n, seq_len, dim = camera_feature.shape  # [bs, num_frames, 257, 1024]
        dim = self._config.tf_d_model

        status_feature = features['status_feature']
        all_frames_status_feature = torch.stack(
            [
                features['status_feature_prev_3'],
                # features['status_feature_prev_2'],
                # features['status_feature_prev_1'],
                status_feature,
                # features['status_feature_next_1'],
                # features['status_feature_next_2'],
                # features['status_feature_next_3'],
                features['status_feature_next_4'],
                # features['status_feature_next_5'],
                # features['status_feature_next_6'],
                # features['status_feature_next_7'],
                features['status_feature_next_8'],
            ], dim=1
        )

        cmd = all_frames_status_feature[..., :2]  # [bs, num_frames, 4] one-hot
        cmd = torch.argmax(cmd, dim=-1)  # [bs, num_frames] int


        batch_size = status_feature.shape[0]
        device= camera_feature.device

        # img_feat = self.image_encoder(camera_feature)[-1]
        # img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)
        # img_feat = self.image_fc(img_feat.clone()) # 512 -> 256 # [bs, 8*32, 256]
        
        # # v1: concat 3 views in sequence length dimension  [bs, 3, 256, 1024] -> [bs, 768, 1024]
        # img_feat = camera_feature[:, :, 1:, :].reshape(b, -1, dim)
        
        # v2: use learnable query do cross attention with each view dino feature
        init_view_query_feat = self.vision_query.repeat(batch_size, 1, 1, 1, 1).reshape(-1, seq_len-1, dim)  # [1, 12, 3, 256, 1024] -> [bs, 12, 3, 256, 1024]
        spatial_view_feat = self.vision_decoder(init_view_query_feat, all_frames_camera_feature[:, :, :, 1:, :].reshape(-1, seq_len-1, dim))
            
        spatial_view_feat = spatial_view_feat.reshape(b, num_frames, -1, dim)  # [b, num_frames, 768, 1024]

        # =============================== keyval trans =======================================
        status_encoding = self._status_encoding(all_frames_status_feature)  # [bs, frame, 8] -> [bs, 12, 256]
        keyval = torch.cat([spatial_view_feat, status_encoding[:, :, None]], dim=-2)  # [bs, 12, 768+1, 256]
        keyval = keyval.clone() + self._keyval_embedding.weight[None, None, ...]

        keyval_final = keyval   # [bs, 12, 768+1 , 256]


        ego_query = self._query_embedding.weight[None, None, ...].repeat(batch_size, num_frames, 1, 1)   # [bs, num_mode * num_poses, 256]

        # 多模态引导，与 forward_train 保持一致
        if self._config.num_mode:
            ##使用anchor去引导多模态轨迹
            if self._num_mode == 20:
                # diffusiondrive的20模态
                mode_ref = self.waypoint_mode_ref.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, num_mode, num_poses, 2]
            elif self._num_mode == 18:
                mode_ref = self.waypoint_mode_ref.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)[..., :2]  # [bs, 3, 6, 8, 2]
            elif self._num_mode == 3:
                # 使用左中右三种命令来指引模态
                mode_ref = self.waypoint_mode_ref.weight # [num_mode, 256]
                if self._config.use_cmd_embed:
                    mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, self._num_poses, 1)  # [bs, num_mode, num_poses, 256]
                else:
                    mode_ref = mode_ref.reshape(self._num_mode, self._num_poses, self._config.tf_d_model)  # [num_mode, num_poses, 256]
                    mode_ref = mode_ref.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, num_mode, num_poses, 256]           
            elif self._num_mode == 4:
                # 使用左中右三种命令来指引模态
                mode_ref = self.waypoint_mode_ref.weight # [num_mode, 256]
                if self._config.use_cmd_embed:
                    mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, self._num_poses, 1)  # [bs, num_mode, num_poses, 256]
                else:
                    mode_ref = mode_ref.reshape(self._num_mode, self._num_poses, self._config.tf_d_model)  # [num_mode, num_poses, 256]
                    mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(1).repeat(batch_size, num_frames, 1, 1, 1)  # [bs, num_frames, num_mode, num_poses, 256]
            else:
                mode_ref = self.waypoint_mode_ref.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, num_mode, num_poses, 2]
            
            if self._num_mode in [20, 18]:
                mode_ref_feat_sin = self.gen_sineembed_for_position(mode_ref)  # [bs, num_mode, num_poses, 256]
                mode_ref_feat = self._mode_embedding(mode_ref_feat_sin)  # [bs, num_mode, num_poses, 256]
            else:
                mode_ref_feat = self._mode_embedding(mode_ref)  # [bs, num_frames, num_mode, num_poses, 256]
            ego_query = ego_query.clone() + mode_ref_feat.reshape(batch_size, num_frames, -1, mode_ref_feat.shape[-1]).clone()



        
        ##==============================轨迹特征提取========================================
        ego_query = ego_query.reshape(-1, ego_query.shape[-2], ego_query.shape[-1])
        keyval_final = keyval_final.reshape(-1, keyval_final.shape[-2], keyval_final.shape[-1])
        ego_query_out = self._tf_decoder(ego_query, keyval_final)

        ##==============================轨迹特征提取========================================

       

        ##==============================控制命令预测==============================================
        cmd_query = self._cmd_pred_query.weight[None, None, ...].repeat(batch_size, num_frames, 1, 1)  # [bs, num_frames, 1, 256]
        cmd_query = cmd_query.reshape(-1, cmd_query.shape[-2], cmd_query.shape[-1])
        cmd_query_out = self._cmd_head_decoder(cmd_query, keyval_final)  #

        cmd_logits = self._cmd_mlp(cmd_query_out).squeeze(1)  # [bs*num_frames, 4]
        # 取一个除了3外，最大概率的类别作为预测值
        # cmd_pred = torch.argmax(cmd_logits[:, :3], dim=-1)  # [bs]
        cmd_pred = torch.argmax(cmd_logits, dim=-1)  # [bs*num_frames]

        # wm
        if self.use_wm:
            trajectory = {}
            trajectory['first_traj'] = self._trajectory_head(ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 2, ...], cmd=cmd[:, 2])

            trajectory['cmd_logits'] = cmd_logits.reshape(batch_size, num_frames, -1)[:, 2, ...].squeeze(1)
            trajectory['cmd_pred'] = cmd_pred.reshape(batch_size, num_frames)[:, 2]
            trajectory['cmd_gt'] = cmd.reshape(batch_size, num_frames)[:, 2]

            wm_keyval = spatial_view_feat[:, :-1, ...].reshape(batch_size, -1, dim)
            wm_query = self._wm_query_embedding.weight[None, ...].repeat(batch_size, 1, 1).reshape(batch_size, -1, dim)

            n_tokens_4d = (num_frames-1) * self._config.num_views * 16 * 16
            attention_mask = self.temporal_world_model.get_attention_mask(n_tokens_4d, self._config.num_views * 16 * 16).to(device)
            wm_next_latent = self.temporal_world_model.forward(
                                                            query=wm_query,
                                                            keyval=wm_keyval,
                                                            time_frames=num_frames-1,
                                                            views=self._config.num_views,
                                                            height=16,
                                                            width=16,
                                                            tgt_mask=attention_mask,
                                                            memory_mask=attention_mask
                                                        )
            trajectory['wm_next_latent']=wm_next_latent.reshape(batch_size, num_frames-1, self._config.num_views, -1, dim)
            
            # refine
            refine_query = ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...]
            refine_keyval = wm_next_latent.reshape(batch_size, num_frames-1, -1, dim)[:, -1, ...]
            refine_query_out = ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...] + self.refine_block(refine_query, refine_keyval)
            
            # TODO: comupute the refined trajectory with residual
            trajectory['refined_traj'] = self.refine_traj_head(refine_query_out, cmd=cmd[:, 1])
            for key in trajectory['first_traj']:
                trajectory['refined_traj'][key] += trajectory['first_traj'][key]
        return trajectory
    
    # the loss function for world model
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        logging_prefix: str = None,
    ) -> torch.Tensor:

        def get_traj_loss(traj_dic, targets):
            all_trajs = traj_dic.get("all_trajectories")  # [B, M, T, 3]
            gt = targets["trajectory"]  # [B, T, 3]
            B, M = all_trajs.shape[0], all_trajs.shape[1]

            traj_gt_dist = torch.norm(all_trajs[..., :2] - gt[:, None, :, :2], p=1, dim=-1)  # [B,M,T]
            fde_dist = traj_gt_dist[:, :, -1]  # [B,M]
            best_mode = torch.argmin(fde_dist, dim=-1)  # [B]
            best_traj = all_trajs[torch.arange(B), best_mode]
            trajectory_loss = torch.nn.functional.l1_loss(best_traj, gt)

            return trajectory_loss

        loss_dict = {}
        # ========================= 多模态监督 =========================
        if self._config.num_mode and logging_prefix == 'train':
            
            first_traj_dict = predictions['first_traj']
            refined_traj_dict = predictions['refined_traj']

            first_traj_loss = get_traj_loss(first_traj_dict, targets)
            refined_traj_loss = get_traj_loss(refined_traj_dict, targets)
            loss_dict["first_traj_loss"] = first_traj_loss * self.traj_loss_weight
            loss_dict["refined_traj_loss"] = refined_traj_loss * self.traj_loss_weight

            ## 控制命令预测loss
            if "cmd_logits" in predictions:
                cmd = predictions['cmd_gt']
                cmd_loss = torch.nn.functional.cross_entropy(predictions["cmd_logits"], cmd)
                loss_dict["cmd_loss"] = cmd_loss * self._config.traj_cmd_loss_weight

        else:
            first_traj_dict = predictions['first_traj']
            refined_traj_dict = predictions['refined_traj']

            first_traj_loss = get_traj_loss(first_traj_dict, targets)
            refined_traj_loss = get_traj_loss(refined_traj_dict, targets)
            loss_dict["first_traj_loss"] = first_traj_loss * self.traj_loss_weight
            loss_dict["refined_traj_loss"] = refined_traj_loss * self.traj_loss_weight

        # 世界模型损失
        if 'wm_next_latent' in predictions:
            wm_loss_a = torch.nn.functional.mse_loss(predictions["wm_next_latent"], predictions["next_latent"].detach())
            loss_dict["wm_loss"] = wm_loss_a * self.wm_loss_weight

        return loss_dict

class TrajectoryHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, num_mode: int = None):
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

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

    def forward(self, object_queries, pred_cmd=None, cmd=None) -> Dict[str, torch.Tensor]:
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
            elif self._num_mode == 3 and pred_cmd is not None:
                # cmd==3 表示未指定命令，针对这部分样本使用 pred_cmd，否则使用 gt cmd
                use_pred = (cmd == 3)
                # 如果存在命令3，则打印警告
                # if use_pred.any():
                #     print("Warning: Using predicted command for some samples in TrajectoryHead.")
                best_traj_idx = torch.where(use_pred, pred_cmd, cmd)
                batch_indices = torch.arange(poses.shape[0], device=poses.device)
                best_traj = poses[batch_indices, best_traj_idx]
                return {
                    "trajectory": best_traj,
                    "cls_logits": cls_score,
                    "all_trajectories": poses,
                }
            elif self._num_mode == 4 and pred_cmd is None:
                ## 直接使用gt的cmd作为指引
                best_traj_idx = cmd.reshape(-1)
                batch_indices = torch.arange(poses.shape[0], device=poses.device)
                best_traj = poses[batch_indices, best_traj_idx]
                return {
                    "trajectory": best_traj,
                    "cls_logits": cls_score,
                    "all_trajectories": poses,
                }

        else:
            poses = self._mlp(object_queries)  # [bs, num_poses, 3]
            poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
            return {"trajectory": poses}

