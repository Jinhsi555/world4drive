from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import timm
import pickle
import gzip
from dataclasses import dataclass
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from navsim.common.enums import StateSE2Index

import torchvision.models as models
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from typing import Tuple, Dict, Optional
from copy import deepcopy

import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

from torch.linalg import inv

from navsim.agents.vggt.models.vggt import VGGT
from navsim.agents.vggt.utils.load_fn import load_and_preprocess_images
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

class AdaptiveTransform(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用自适应池化调整空间尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 32))
        # 1x1卷积调整通道数
        self.conv = nn.Conv2d(2048, 256, kernel_size=1)
        
    def forward(self, x):
        # x: [bs, 9, 37, 2048]
        x = x.permute(0, 3, 1, 2)  # [bs, 2048, 9, 37]
        
        # 先调整空间尺寸
        x = self.adaptive_pool(x)  # [bs, 2048, 8, 32]
        
        # 再调整通道数
        x = self.conv(x)  # [bs, 256, 8, 32]
        
        x = x.permute(0, 2, 3, 1)  # [bs, 8, 32, 256]
        return x
@dataclass
class VGGTEmbeddingConfig:
    input_dim: int
    output_dim: int
    patch_size: int = 14  

class VGGTEmbedding(nn.Module):
    def __init__(
        self,
        config: VGGTEmbeddingConfig,
        vggt: nn.Module = None
    ) -> None:
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.patch_size = config.patch_size
        self.adaptive_transform = AdaptiveTransform()
        self.token_in_dim = self.input_dim
        self.vggt = vggt  # 保存vggt实例

        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_in_dim, self.token_in_dim),
            nn.GELU(),
            nn.Linear(self.token_in_dim, self.output_dim),
        )

    def forward(
        self, aggregated_tokens_list: List[torch.Tensor], patch_start_idx: int, images_shape: Tuple, media_type: str
    ) -> torch.Tensor:
        # tokens shape: (Batch, Frame, Token_num, vggt_dim)
        tokens = aggregated_tokens_list[-1][:, :, patch_start_idx:] # [bs, 1, num_token, vggt_dim]
        bz, _, num_tokens, vgg_dim = tokens.shape
        bz, _, c, h, w = images_shape
        num_patch_h, num_patch_w = h // self.patch_size, w // self.patch_size
        # if media_type == "images":
        #     pass  
     
        # tokens shape: (Batch, sequence, NUM_PATCH_H, NUM_PATCH_W, output_dim) = (B, 1, 9, 37, 2048)
        tokens = tokens.reshape(bz, num_patch_h, num_patch_w, vgg_dim)  #[bz, 1, 9, 37, 2048]-->[bz, 9, 37, 2048]
        # x = self.adaptive_transform(tokens) # [bz, 8, 32, 256]
        ## 不使用pooling效果似乎更好
        return tokens

    def get_vgg_embeds(self, media_input: torch.Tensor, media_type: str):
        """
        Get the vgg features for images and videos.
        Shape of media_input: (B, F, C, H, W) where F=1 for images
        Media_type: "images" or "video"
        """
        vgg_aggregator = self.vggt.aggregator
        with torch.no_grad():
            with torch.cuda.amp.autocast():     
                output_list, patch_start_idx = vgg_aggregator.forward(media_input)
        # -- shape of vgg_embeds: (Batch, 8, 32, 256)
        vgg_embeds = self.forward(output_list, patch_start_idx, media_input.shape, media_type)

        # mlp层融合
        vgg_embeds = self.token_mlp(vgg_embeds)
        
        return vgg_embeds

class LawModel(nn.Module):
    def __init__(self, 
            config: TransfuserConfig,
            ):
        super().__init__()

        #pretrained here
        self.image_encoder = timm.create_model(
                    config.image_architecture, pretrained=False, features_only=True
                )
        self.image_encoder.load_state_dict(torch.load('/lpai/volumes/ad-vla-bd-ga/ypx/ckpts/resnet34-333f7ec4.pth'),strict=False)

        # # 初始化部分
        self.config = config

        self.vggt = VGGT()
        # # 选择是否导入pretrain model
        self.vggt.load_state_dict(torch.load("/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/checkpoints"))
        # # 选择是否冻结参数
        self.vggt.eval()
        self.vggt.requires_grad_(False)

        # # ft过程
        if self.config.training_mode == 'ft' or self.config.training_mode == 'bcsl':
            # 初始化方差预测网络
            self._var_head = TrajectoryVariance(config.tf_d_model)

            self._yaw_head = TrajectoryYaw(8*2)

            # 冻结imgbackbone网络
            self.image_encoder.requires_grad_(False)


        self.vggt_embedding_config = VGGTEmbeddingConfig(
            input_dim=config.tf_d_model,  # 256
            output_dim=config.tf_d_model,  # 256
            patch_size=14,  # 

        )
        self.vggt_embedding = VGGTEmbedding(
            config=self.vggt_embedding_config,
            vggt=self.vggt  # 传入vggt实例
        )

        self.image_fc = nn.Linear(512, 256)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        num_poses = config.trajectory_sampling.num_poses
        #TODO: petr position_embedding
        # num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 20*12 + 1
        if config.use_depth:
            num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 264+1
        else:
            num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 256+1
    
        self._keyval_embedding = nn.Embedding(
            num_keyval, config.tf_d_model
        )  # 8x8 feature grid + trajectory
        self._vggt_keyval_embedding = nn.Embedding(256, config.tf_d_model)
        self.num_poses = config.trajectory_sampling.num_poses
        self.num_spa_poses = 50  # 关注的spa点数
        self.num_navi_poses = 1
        self._query_embedding = nn.Embedding(self.num_poses, config.tf_d_model)

        # spatial query
        self._spa_query_embedding = nn.Embedding(self.num_spa_poses, config.tf_d_model)

        # navi query
        self._navi_query_embedding = nn.Embedding(self.num_navi_poses, config.tf_d_model)

        ##===================================仅用1维特征相加的版本====================================
        # self._query_embedding = nn.Embedding(1, config.tf_d_model)

        # # spatial query
        # self._spa_query_embedding = nn.Embedding(1, config.tf_d_model)

        # # navi query
        # self._navi_query_embedding = nn.Embedding(1, config.tf_d_model)
        ##====================================仅用1维特征相加的版本====================================       
        
        ##=====================================轨迹特征提取层============================================
        spa_tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._spa_tf_decoder = nn.TransformerDecoder(spa_tf_decoder_layer, config.tf_num_layers)

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)

        navi_tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._navi_tf_decoder = nn.TransformerDecoder(navi_tf_decoder_layer, config.tf_num_layers)
        ##=====================================轨迹特征提取层============================================

        ##=====================================轨迹特征交互层============================================
        # 定义self-attention层
        self_attn_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.global_self_attn = nn.TransformerDecoder(self_attn_layer, 1)

        ##=====================================轨迹特征交互层============================================


        self._trajectory_head = TrajectoryHead(num_poses, config.tf_d_ffn, config.tf_d_model)
        
        # add world model here
        self.use_wm = False
        self.use_wm_training = False
        self.wm_loss_weight=0.2
        self.spa_traj_loss_weight=1.0
        self.traj_loss_weight=1.0
        self.navi_traj_loss_weight=2e-4
        self.stride=32
        self.use_depth=config.use_depth
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

        # # 获取vggt特征
        media_type = "images"
        image_input = features['camera_feature']
        # 需要添加一维序列维
        media_input = self.preprocess_images_batch_cuda(image_input).unsqueeze(1)   # [bz, 1, 3, 126, 518]
        vgg_embeds = self.vggt_embedding.get_vgg_embeds(media_input, media_type)    # [bz, 8, 32, 256]
        vgg_embeds = vgg_embeds.flatten(1, 2)   # [bz, 256, 256]
        vgg_embeds = vgg_embeds + self._vggt_keyval_embedding.weight[None, ...]
        # add vgg feat to img
        # img_feat +=  vgg_embeds
        
        # =============================== keyval trans=======================================
        status_encoding = self._status_encoding(status_feature) # [bs, 256]

        keyval = torch.cat([img_feat, status_encoding[:, None]], dim=1) # [bs, 256+1, 256]
        keyval = keyval.clone() + self._keyval_embedding.weight[None, ...]  # 每一维都有一个query

        # vgg 特征融合
        keyval = torch.cat([keyval, vgg_embeds], dim=-2)    # [bz, 256+1+256, 256]

        keyval_final = keyval   # [bs, 64, 257+256 , 256]

        # traj decoder
        spa_query = self._spa_query_embedding.weight[None, ...].repeat(batch_size, 1, 1)

        temp_query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)

        navi_query = self._navi_query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        ##==============================轨迹特征提取========================================
        spa_query_out = self._spa_tf_decoder(spa_query, keyval_final)
        
        temp_query_out = self._tf_decoder(temp_query, keyval_final)

        navi_query_out = self._navi_tf_decoder(navi_query, keyval_final)
        ##==============================轨迹特征提取========================================

        #=============================轨迹特征交互==========================================
        # 将三个query堆叠
        plan_queries_stacked = torch.cat([spa_query_out, temp_query_out, navi_query_out], dim=1)  # [bs, num_spa + 8 + 1, 256]

        # 全局self-attention (使用自己作为memory)
        global_features = self.global_self_attn(plan_queries_stacked, plan_queries_stacked)  # [bs, num_spa + 8 + 1, 256]

        # 分离全局特征
        spa_global = global_features[:, 0:self.num_spa_poses, :]  # [1, 1, 256]
        temp_global = global_features[:, self.num_spa_poses:self.num_spa_poses+self.num_poses, :]  # [1, 1, 256]
        navi_global = global_features[:, self.num_spa_poses+self.num_poses:, :]  # [1, 1, 256]

        #=============================轨迹特征交互==========================================

        trajectory = self._trajectory_head(spa_global, temp_global, navi_global)

        ## 保留gt

        # 如果rl，则输出方差
        if self.config.training_mode == "ft" or self.config.training_mode == 'bcsl':
            traj_var = self._var_head(temp_global)
            trajectory['trajectory_var'] = traj_var

        return trajectory

    def forward_train(self, features) -> Dict[str, torch.Tensor]:
        camera_feature = features['camera_feature']

        status_feature = features['status_feature']

        batch_size = status_feature.shape[0]
        device= camera_feature.device

        img_feat = self.image_encoder(camera_feature)[-1]
        img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)
        img_feat = self.image_fc(img_feat.clone()) # 512 -> 256 # [bs, 8*32, 256]

        # # 获取vggt特征
        media_type = "images"
        image_input = features['camera_feature']
        # 需要添加一维序列维
        media_input = self.preprocess_images_batch_cuda(image_input).unsqueeze(1)   # [bz, 1, 3, 126, 518]
        vgg_embeds = self.vggt_embedding.get_vgg_embeds(media_input, media_type)    # [bz, 8, 32, 256]
        vgg_embeds = vgg_embeds.flatten(1, 2)   # [bz, 256, 256]
        vgg_embeds = vgg_embeds + self._vggt_keyval_embedding.weight[None, ...]
        # add vgg feat to img
        # img_feat +=  vgg_embeds

        # =============================== keyval trans=======================================
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.cat([img_feat, status_encoding[:, None]], dim=1)
        keyval = keyval.clone() + self._keyval_embedding.weight[None, ...]

        # vgg 特征融合
        keyval = torch.cat([keyval, vgg_embeds], dim=-2)    # [bz, 256+1+256, 256]

        keyval_final = keyval   # [bs, 64, 257+256 , 256]


        # traj decoder
        spa_query = self._spa_query_embedding.weight[None, ...].repeat(batch_size, 1, 1)    # [bs, num_spa, 256]

        temp_query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)   # [bs, num_poses, 256]

        navi_query = self._navi_query_embedding.weight[None, ...].repeat(batch_size, 1, 1)  # [bs, 1, 256]
        ##==============================轨迹特征提取========================================
        spa_query_out = self._spa_tf_decoder(spa_query, keyval_final)
        
        temp_query_out = self._tf_decoder(temp_query, keyval_final)

        navi_query_out = self._navi_tf_decoder(navi_query, keyval_final)
        ##==============================轨迹特征提取========================================

        #=============================轨迹特征交互==========================================
        # 将三个query堆叠
        plan_queries_stacked = torch.cat([spa_query_out, temp_query_out, navi_query_out], dim=1)  # [bs, num_spa + 8 + 1, 256]

        # 全局self-attention (使用自己作为memory)
        global_features = self.global_self_attn(plan_queries_stacked, plan_queries_stacked)  # [bs, num_spa + 8 + 1, 256]

        # 分离全局特征
        spa_global = global_features[:, 0:self.num_spa_poses, :]  # [1, 1, 256]
        temp_global = global_features[:, self.num_spa_poses:self.num_spa_poses+self.num_poses, :]  # [1, 1, 256]
        navi_global = global_features[:, self.num_spa_poses+self.num_poses:, :]  # [1, 1, 256]

        #=============================轨迹特征交互==========================================

        trajectory = self._trajectory_head(spa_global, temp_global, navi_global)
        
        # 如果rl，则输出方差
        if self.config.training_mode == "ft" or self.config.training_mode == 'bcsl':
            traj_var = self._var_head(temp_global)
            trajectory['trajectory_var'] = traj_var
            trajectory['keyval_final'] = keyval_final

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
    ) -> torch.Tensor:
        loss_dict = {}

        # spa loss
        resampled_spatial_cur_waypoint, resampled_spatial_gt_waypoint, gt_spatial_mask = self._trajectory_head.process_trajectories(predictions["spa_trajectory"], targets["trajectory"], interval=2.0)
        resampled_spatial_cur_waypoint = resampled_spatial_cur_waypoint * gt_spatial_mask
        resampled_spatial_gt_waypoint = resampled_spatial_gt_waypoint * gt_spatial_mask
        spa_trajectory_loss = torch.nn.functional.l1_loss(resampled_spatial_cur_waypoint, resampled_spatial_gt_waypoint, reduction='sum')
        spa_trajectory_loss = spa_trajectory_loss / (gt_spatial_mask.sum() + 1e-6)

        loss_dict["spa_traj_loss"] = spa_trajectory_loss * self.spa_traj_loss_weight

        # temp loss
        trajectory_loss = torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

        loss_dict["traj_loss"] = trajectory_loss * self.traj_loss_weight
        
        # navi loss
        # dist = torch.norm(predictions["trajectory"][:, -1, :2] - targets["trajectory"][:, -1, :2], p=2, dim=-1)
        # if dist > 5:
        #     loss_navi_waypoint = self.loss_plan_reg(navi_cur_waypoint, gt_end_point) * 0.01
        # else:
        #     loss_navi_waypoint = self.loss_plan_reg(navi_cur_waypoint, gt_end_point) * 5e-4

        navi_trajectory_loss = torch.nn.functional.l1_loss(predictions["trajectory"][:, -1, :], targets["trajectory"][:, -1, :])

        loss_dict["navi_traj_loss"] = navi_trajectory_loss * self.navi_traj_loss_weight

        if self.use_wm_training and 'wm_next_latent' in predictions:
            wm_loss_a = torch.nn.functional.mse_loss(predictions["wm_next_latent"], predictions["cur_latent"].detach())
            loss_dict["wm_loss"] = wm_loss_a * self.wm_loss_weight 

        return loss_dict

    def preprocess_images_batch_cuda(self, images, mode="crop"):
        """
        GPU optimized version
        """
        # 确保在 GPU 上
        device = images.device
        if device.type != 'cuda':
            images = images.cuda()
        
        bs, c, height, width = images.shape
        target_size = 518
        
        # 使用 torch.cuda.amp 进行混合精度计算
        with torch.cuda.amp.autocast():
            if mode == "pad":
                if width >= height:
                    new_width = target_size
                    new_height = ((height * target_size // width + 7) // 14) * 14
                else:
                    new_height = target_size
                    new_width = ((width * target_size // height + 7) // 14) * 14
            else:
                new_width = target_size
                new_height = ((height * target_size // width + 7) // 14) * 14
            
            # 使用 CUDA 优化的插值
            images = torch.nn.functional.interpolate(
                images, 
                size=(new_height, new_width), 
                mode='bilinear',
                align_corners=False,
                antialias=False  # 关闭抗锯齿以提速
            )
        
        # 后续操作保持在原精度
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            images = images[:, :, start_y : start_y + target_size, :]
        
        elif mode == "pad" and (images.shape[2] < target_size or images.shape[3] < target_size):
            # 使用更高效的 padding
            pad_h = max(0, target_size - images.shape[2])
            pad_w = max(0, target_size - images.shape[3])
            
            if pad_h > 0 or pad_w > 0:
                # 创建目标大小的张量并复制
                padded = torch.ones((bs, c, target_size, target_size), 
                                dtype=images.dtype, device=images.device)
                
                h_offset = pad_h // 2
                w_offset = pad_w // 2
                
                padded[:, :, h_offset:h_offset+images.shape[2], 
                    w_offset:w_offset+images.shape[3]] = images
                images = padded
        
        return images.to(device)

class TrajectoryHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self._spa_num_poses = 50
        self._navi_num_poses = 1

        self._spa_mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, StateSE2Index.size()),   # 3
        )

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, StateSE2Index.size()),   # 3
        )

        self._navi_mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, StateSE2Index.size()),   # 3
        )

    # def forward(self, object_queries) -> Dict[str, torch.Tensor]:
    #     poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())    # [bz, 8, 3]
    #     poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
    #     return {"trajectory": poses}

    def forward(self, spa_query, temp_query, navi_query) -> Dict[str, torch.Tensor]:

        spa_poses = self._spa_mlp(spa_query).reshape(-1, self._spa_num_poses, StateSE2Index.size()) # [bz, num_spa, 3]
        spa_poses[..., StateSE2Index.HEADING] = spa_poses[..., StateSE2Index.HEADING].tanh() * np.pi

        poses = self._mlp(temp_query).reshape(-1, self._num_poses, StateSE2Index.size())    # [bz, 8, 3]
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi

        navi_poses = self._navi_mlp(navi_query).reshape(-1, self._navi_num_poses, StateSE2Index.size())     # [bz, 1, 3]
        navi_poses[..., StateSE2Index.HEADING] = navi_poses[..., StateSE2Index.HEADING].tanh() * np.pi
        
        return {"trajectory": poses,
                "spa_trajectory": spa_poses,
                "navi_trajectory": navi_poses}

    
    def circular_interp_batch(self, angle0: torch.Tensor, angle1: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        """
        批量循环插值角度，处理角度的周期性
        
        Args:
            angle0: 起始角度 [...]
            angle1: 结束角度 [...]
            ratio: 插值比例 [...]
        
        Returns:
            插值后的角度 [...]
        """
        # 使用complex数表示角度，避免多次atan2计算
        z0 = torch.complex(torch.cos(angle0), torch.sin(angle0))
        z1 = torch.complex(torch.cos(angle1), torch.sin(angle1))
        
        # 计算最短路径的角度差
        z_diff = z1 / z0
        angle_diff = torch.angle(z_diff)
        
        # 插值并返回结果
        result_angle = angle0 + ratio * angle_diff
        return torch.remainder(result_angle + torch.pi, 2 * torch.pi) - torch.pi

    def resample_traj_tensor(self, traj: torch.Tensor, interval: float = 2.0, num_points: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对轨迹按给定间隔重采样，固定采样num_points个点
        
        Args:
            traj: [bz, N, 3] 的绝对轨迹（x, y, yaw）
            interval: 采样间隔
            num_points: 采样点数
        
        Returns:
            resampled_traj: [bz, num_points, 3] 的轨迹序列
            mask: [bz, num_points] 有效点的mask
        """
        bz, N, _ = traj.shape
        device = traj.device
        
        # 准备绝对坐标轨迹，添加原点
        abs_traj = torch.cat([traj.new_zeros(bz, 1, 3), traj], dim=1)  # [bz, N+1, 3]
        
        # 计算累计距离（基于x,y）
        diffs = torch.diff(abs_traj[:, :, :2], dim=1)  # [bz, N, 2]
        distances = torch.norm(diffs, dim=2)  # [bz, N]
        cum_dist = torch.cat([distances.new_zeros(bz, 1), 
                            distances.cumsum(dim=1)], dim=1)  # [bz, N+1]
        
        # 创建采样位置并计算有效mask
        sample_positions = torch.arange(num_points, device=device, dtype=traj.dtype) * interval
        sample_positions = sample_positions[None, :].expand(bz, -1)  # [bz, num_points]
        
        total_lengths = cum_dist[:, -1:]  # [bz, 1]
        mask = sample_positions <= total_lengths  # [bz, num_points]
        
        # 限制采样位置不超过轨迹总长度
        clamped_positions = torch.minimum(sample_positions, total_lengths)
        
        # 使用searchsorted找到插值索引
        idxs = torch.searchsorted(cum_dist, clamped_positions, right=False)  # [bz, num_points]
        idxs = idxs.clamp(1, N)
        
        # 准备批量索引
        batch_idx = torch.arange(bz, device=device)[:, None].expand(-1, num_points)
        idx0, idx1 = idxs - 1, idxs
        
        # 获取插值所需的点和距离
        points0 = abs_traj[batch_idx, idx0]  # [bz, num_points, 3]
        points1 = abs_traj[batch_idx, idx1]  # [bz, num_points, 3]
        
        dist0 = cum_dist[batch_idx, idx0]  # [bz, num_points]
        dist1 = cum_dist[batch_idx, idx1]  # [bz, num_points]
        
        # 计算插值比例，避免除零
        dist_diff = (dist1 - dist0).clamp(min=1e-6)
        ratio = ((clamped_positions - dist0) / dist_diff).clamp(0, 1)  # [bz, num_points]
        
        # 对x,y进行线性插值
        ratio_expanded = ratio[:, :, None]  # [bz, num_points, 1]
        new_xy = torch.lerp(points0[:, :, :2], points1[:, :, :2], ratio_expanded)  # [bz, num_points, 2]
        
        # 对yaw进行循环插值
        new_yaw = self.circular_interp_batch(points0[:, :, 2], points1[:, :, 2], ratio)
        
        # 组合结果
        new_traj = torch.cat([new_xy, new_yaw[:, :, None]], dim=2)  # [bz, num_points, 3]
        
        # 应用mask，将超出范围的点设置为0
        new_traj = new_traj * mask[:, :, None]
        
        return new_traj, mask

    def process_trajectories(self, pred: torch.Tensor, gt: torch.Tensor, interval: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        处理预测和真实轨迹，确保它们具有相同的采样点
        
        Args:
            pred: [B, 50, 3] 模型输出的轨迹序列
            gt: [B, N, 3] 的原始轨迹
            interval: 采样间隔
        
        Returns:
            processed_pred: [B, 50, 3] 处理后的预测轨迹
            processed_gt: [B, 50, 3] 重采样后的真实轨迹
            mask: [B, 50, 1] 有效点的mask
        """
        bz, num_points = pred.shape[:2]
        
        # 确保gt的batch size与pred一致
        if gt.shape[0] != bz:
            gt = gt.expand(bz, -1, -1) if gt.shape[0] == 1 else gt[:bz]
        
        # 对gt进行重采样
        gt_resampled, mask = self.resample_traj_tensor(gt, interval, num_points)
        
        # 返回处理后的结果
        return pred, gt_resampled, mask[:, :, None]
    
class TrajectoryVariance(nn.Module):
    """方差网络，输出轨迹的标准差"""
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(TrajectoryVariance, self).__init__()
        self.xy_fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 3) # 3*2
            )
        # self.yaw_fc = nn.Sequential(
        #         nn.Linear(input_dim, hidden_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(hidden_dim, 1) # 3*2
        # )
        
        # 初始化最后一层，使初始标准差较小
        # nn.init.uniform_(self.fc_std.weight, -3e-3, 3e-3)
        # nn.init.uniform_(self.fc_std.bias, -3e-3, 3e-3)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.xy_fc(state) # 3 x,y,co_xy

        return out
    
class TrajectoryYaw(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 最终 yaw 输出
        self.yaw_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 因为要拼接 state_feat + img_feat
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 8)
        )

    def forward(self, state: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        """
        state: [N, B, ...]
        img_feat: [B, 256, 256]  # [batch, channel, spatial]
        return: [N, B, 8]
        """
        N, B = state.shape[:2]
        state = state.flatten(-2, -1)  # [N, B, input_dim]

        # 提取 img_feat 全局特征 [B, 256]
        img_feat_global = img_feat.mean(dim=-2)  # GAP over spatial -> [B, 256]

        # 对所有 N 个时间步一次性编码
        state_feat = self.encoder(state.view(N * B, -1))  # [N*B, hidden_dim]

        # 把 img_feat_global 扩展到 [N*B, hidden_dim]
        img_feat_repeat = img_feat_global.unsqueeze(0).expand(N, -1, -1).reshape(N * B, -1)

        # 拼接特征
        feat = torch.cat([state_feat, img_feat_repeat], dim=-1)  # [N*B, hidden_dim*2]

        # 输出 yaw
        yaw = self.yaw_fc(feat).tanh() * np.pi  # [N*B, 8]

        return yaw.view(N, B, 8)
    
class TrajectoryGaussianPolicy:
    """轨迹高斯策略，支持微调预训练的Actor"""
    def __init__(
        self, 
        n_samples: int = 10,
    ):
        self.n_samples = n_samples
    
    def get_distribution(self, mean_traj: torch.Tensor, variance: torch.Tensor) -> dist.Normal:
        """
        构建多元高斯分布
        
        Args:
            variance_vector: [B, T, 6] 包含方差和协方差的向量
                            [var_x, var_y, var_phi, cov_xy, cov_xphi, cov_yphi]
            mean: [B, T, 3] 均值向量，如果为None则默认为0
        
        Returns:
            MultivariateNormal分布对象
        """
        # B, T, _ = mean_traj.shape
        
        # L = torch.zeros(B, T, 3, 3, device=mean_traj.device)
        # indices = torch.tril_indices(3, 3)
        # L[:, :, indices[0], indices[1]] = variance
        # diag = torch.diagonal(L, dim1=-2, dim2=-1)
        # diag_positive = F.softplus(diag)
        # L = L - torch.diag_embed(diag) + torch.diag_embed(diag_positive)

        
        # return dist.MultivariateNormal(
        #     loc=mean_traj.view(-1, 3),
        #     scale_tril=L.view(-1, 3, 3),
        #     validate_args=False)

        # 二元
        B, T, _ = mean_traj.shape
        
        L = torch.zeros(B, T, 2, 2, device=mean_traj.device)
        indices = torch.tril_indices(2, 2)
        L[:, :, indices[0], indices[1]] = variance
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag_positive = F.softplus(diag)
        L = L - torch.diag_embed(diag) + torch.diag_embed(diag_positive)

        
        return dist.MultivariateNormal(
            loc=mean_traj.view(-1, 2),
            scale_tril=L.view(-1, 2, 2),
            validate_args=False)

        ##===================================独立3维高斯分布================================================ 
        
        # 独立的3维高斯
        # return dist.Independent(dist.Normal(mean_traj, torch.sqrt(variance)), 1)

            
    def sample_trajectories(self, mean_traj: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        # bs, T, _ = mean_traj.shape
        # distribution = self.get_distribution(mean_traj, variance)
        
        # samples = []
        # for _ in range(self.n_samples):
        #     sample = distribution.rsample()  # [bs*8, 3]
        #     sample = sample.view(bs, T, 3)  # 重塑为 [bs, 8, 3]
        #     samples.append(sample)
        
        # return torch.stack(samples, dim=0)  # [n_samples, bs, 8, 3]

        # 二元
        bs, T, _ = mean_traj.shape
        distribution = self.get_distribution(mean_traj, variance)
        
        samples = []
        for _ in range(self.n_samples):
            sample = distribution.rsample()  # [bs*8, 3]
            sample = sample.view(bs, T, 2)  # 重塑为 [bs, 8, 3]
            samples.append(sample)
        
        return torch.stack(samples, dim=0)  # [n_samples, bs, 8, 2]
    
    def log_prob(self, mean_traj: torch.Tensor, variance: torch.Tensor, 
                 sampled_traj: torch.Tensor) -> torch.Tensor:
        """
        计算采样轨迹的对数概率
        mean_traj: [bs, 8, 3] - 轨迹均值
        variance: [bs, 8, 3] - 轨迹方差
        sampled_traj: [bs, 8, 3] - 要计算概率的轨迹
        returns: [bs] - 每个批次的对数概率
        """
        distribution = self.get_distribution(mean_traj, variance)
        # 计算每个维度的log_prob，然后在时间步和坐标维度上求和
        log_probs = distribution.log_prob(sampled_traj)  # [bs, 8, 3]
        return log_probs.sum(dim=[-2, -1])  # [bs]
    
    def log_prob_batch(self, mean_traj: torch.Tensor, variance: torch.Tensor, 
                    sampled_trajs: torch.Tensor) -> torch.Tensor:
        """
        批量计算多条采样轨迹的对数概率
        mean_traj: [bs, 8, 3]
        variance: [bs, 8, 6]  # 注意这里应该是6维（包含协方差）
        sampled_trajs: [n_samples, bs, 8, 3]
        returns: [n_samples, bs]
        """
        # n_samples, bs, T, _ = sampled_trajs.shape
        # log_probs = []
        
        # for i in range(n_samples):
        #     # 逐个计算每个样本的 log_prob
        #     distribution = self.get_distribution(mean_traj, variance)
        #     sample_flat = sampled_trajs[i].view(-1, 3)  # [bs*8, 3]
        #     log_prob = distribution.log_prob(sample_flat)  # [bs*8]
        #     # log_prob = log_prob.view(bs, T).sum(dim=1)  # [bs]
        #     log_prob = log_prob.view(bs, T).mean(dim=1)  # [bs]
        #     log_probs.append(log_prob)
        
        # return torch.stack(log_probs, dim=0)  # [n_samples, bs]

        # 二元
        n_samples, bs, T, _ = sampled_trajs.shape
        log_probs = []
        
        for i in range(n_samples):
            # 逐个计算每个样本的 log_prob
            distribution = self.get_distribution(mean_traj, variance)
            sample_flat = sampled_trajs[i].view(-1, 2)  # [bs*8, 3]
            log_prob = distribution.log_prob(sample_flat)  # [bs*8]
            # log_prob = log_prob.view(bs, T).sum(dim=1)  # [bs]
            log_prob = log_prob.view(bs, T).mean(dim=1)  # [bs]
            log_probs.append(log_prob)
        
        return torch.stack(log_probs, dim=0)  # [n_samples, bs]
    
        ##=====================================不展平==================================
        # n_samples, bs, T, D = sampled_trajs.shape
        
        # # 扩展mean和std
        # if isinstance(variance, torch.Tensor):
        #     std = torch.sqrt(torch.clamp(variance, min=1e-3))
        #     std_expanded = std.unsqueeze(0).expand(n_samples, -1, -1, -1)
        # else:
        #     std_expanded = math.sqrt(variance)
        
        # mean_expanded = mean_traj.unsqueeze(0).expand(n_samples, -1, -1, -1)
        
        # # 创建扩展的分布
        # distribution = dist.Independent(
        #     dist.Normal(mean_expanded, std_expanded), 1
        # )
        
        # # 直接计算所有样本的log_prob
        # log_probs = distribution.log_prob(sampled_trajs)  # [n_samples, bs, 8]
        # return log_probs.sum(dim=-1)  # [n_samples, bs]
    
