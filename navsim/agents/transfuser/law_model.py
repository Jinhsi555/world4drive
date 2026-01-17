from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import timm
import time

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from navsim.common.enums import StateSE2Index

import torchvision.models as models
import torch.nn.functional as F
import torchvision.models as models

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

class LAWModel(nn.Module):
    def __init__(self, 
            config: TransfuserConfig,
            ):
        super().__init__()

        #pretrained here
        self.image_encoder = timm.create_model(
                    config.image_architecture, pretrained=False, features_only=True
                )
        self.image_encoder.load_state_dict(torch.load(
            '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/ckpts/resnet34-333f7ec4.pth', 
            map_location='cpu', 
            weights_only=False
            ),strict=False)


        self.image_fc = nn.Linear(512, 256)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        num_poses = config.trajectory_sampling.num_poses
        #TODO: petr position_embedding
        # num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 20*12 + 1
        num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 256+1
        # num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 256+1
        # num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 120+1
    
        self._keyval_embedding = nn.Embedding(
            num_keyval, config.tf_d_model
        )  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(num_poses, config.tf_d_model)
        
        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._trajectory_head = TrajectoryHead(num_poses, config.tf_d_ffn, config.tf_d_model)
        
        # add world model here
        self.use_wm = True
        self.use_wm_training = True
        self.wm_loss_weight=0.2
        self.traj_loss_weight=1.0
        self.stride=32
        self.use_all_mb = config.use_all_mb if hasattr(config, 'use_all_mb') else False

        if self.use_wm:
            num_wm_query = num_keyval
            self._wm_query_embedding = nn.Embedding(num_wm_query, config.tf_d_model)
            wm_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.tf_d_model,
                nhead=config.tf_num_head,
                dim_feedforward=config.tf_d_ffn,
                dropout=config.tf_dropout,
                batch_first=True,
            )
            self._wm_decoder = nn.TransformerDecoder(wm_decoder_layer, config.tf_num_layers) # input: Bz, num_token, d_model

    def forward_test(self, features) -> Dict[str, torch.Tensor]:
        camera_feature = features['camera_feature']
        


        status_feature = features['status_feature']

        batch_size = status_feature.shape[0]
        device= camera_feature.device

        # # =============================== img backbone =======================================

        img_feat = self.image_encoder(camera_feature)[-1]   # [bs, 512, 8, 32]
        img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)    # [bs, 8*32, 512]
        img_feat = self.image_fc(img_feat.clone()) # 512 -> 256

        
        # =============================== keyval trans=======================================
        status_encoding = self._status_encoding(status_feature) # [bs, 256]

        keyval = torch.cat([img_feat, status_encoding[:, None]], dim=1) # [bs, 256+1, 256]
        keyval = keyval.clone() + self._keyval_embedding.weight[None, ...]  # 每一维都有一个query

        keyval_final = keyval   # [bs, 64, 257+256 , 256]

        # traj decoder
        temp_query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)

        ##==============================轨迹特征提取========================================
        
        query_out = self._tf_decoder(temp_query, keyval_final)

        ##==============================轨迹特征提取========================================


        trajectory = self._trajectory_head(query_out)


        # 如果rl，则输出方差
        # if self.config.training_mode == "ft" or self.config.training_mode == 'bcsl':
        #     traj_var = self._var_head(query_out)
        #     trajectory['trajectory_var'] = traj_var

        return trajectory

    def forward_train(self, features) -> Dict[str, torch.Tensor]:
        camera_feature = features['camera_feature']

        status_feature = features['status_feature']

        batch_size = status_feature.shape[0]
        device= camera_feature.device

        img_feat = self.image_encoder(camera_feature)[-1]
        img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)
        img_feat = self.image_fc(img_feat.clone()) # 512 -> 256 # [bs, 8*32, 256]


        # =============================== keyval trans=======================================
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.cat([img_feat, status_encoding[:, None]], dim=1)
        keyval = keyval.clone() + self._keyval_embedding.weight[None, ...]

        keyval_final = keyval   # [bs, 64, 257+256 , 256]


      
        temp_query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)   # [bs, num_poses, 256]

        
        ##==============================轨迹特征提取========================================
        
        temp_query_out = self._tf_decoder(temp_query, keyval_final)

        ##==============================轨迹特征提取========================================

        query_out = self._tf_decoder(temp_query, keyval_final)

        trajectory = self._trajectory_head(query_out)

        # # 如果rl，则输出方差
        # if self.config.training_mode == "ft" or self.config.training_mode == 'bcsl':
        #     traj_var = self._var_head(query_out)
        #     trajectory['trajectory_var'] = traj_var
        #     trajectory['keyval_final'] = keyval_final

        if self.use_wm:
            wm_keyval = keyval_final
            wm_target = keyval_final
            wm_query = self._wm_query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
            wm_next_latent = self._wm_decoder(wm_query, wm_keyval)
            trajectory['wm_next_latent']=wm_next_latent
            trajectory['cur_latent']=wm_target
            return trajectory
        else:
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
        trajectory_loss = torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])
        loss_dict["traj_loss"] = trajectory_loss * self.traj_loss_weight
        if self.use_wm_training and 'wm_next_latent' in predictions:
            wm_loss_a = torch.nn.functional.mse_loss(predictions["wm_next_latent"], predictions["cur_latent"].detach())
            loss_dict["wm_loss"] = wm_loss_a * self.wm_loss_weight 

        return loss_dict

class TrajectoryHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, StateSE2Index.size()),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return {"trajectory": poses}


    