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

def resize_K(h,w, K, new_width, new_height):
    """
    对图像进行上下对称裁剪后，再resize，同时更新相机内参矩阵

    参数:
    image: 原始图像 (numpy 数组)
    K: 原始相机内参矩阵，形状 (3, 3)
    crop_pixels: 上下各裁剪的像素数
    new_width: resize后的图像宽度
    new_height: resize后的图像高度

    返回:
    resized_image: 裁剪并resize后的图像
    K_new: 更新后的相机内参矩阵
    """

    # 2. Resize: 计算x和y方向的缩放因子
    s_x = new_width / w
    s_y = new_height / h

    K[0, 0] *= s_x  # fx
    K[1, 1] *= s_y  # fy
    K[0, 2] *= s_x  # cx
    K[1, 2] *= s_y  # cy

    return K

def get_locations_reso(h, w, device, stride, reso):
        """
        Position embedding for image pixels.
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (H, W, 2)
        """
        
        # 计算x方向的位移
        shifts_x = torch.arange(
            0, stride*w, step=stride // reso,
            dtype=torch.float32, device=device
        )
        # 计算y方向的位移
        shifts_y = torch.arange(
            0, h * stride, step=stride // reso,
            dtype=torch.float32, device=device
        ) 
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_y  = shift_y + stride // reso // 2
        shift_x  = shift_x + stride // reso // 2
        coord = torch.stack([shift_x, shift_y, torch.ones_like(shift_x)], dim=0).reshape(3, -1)
        
        return coord

class Ego3DPositionEmbeddingMLP(nn.Module):
    """Absolute pos embedding, learned.
    https://github.com/kwea123/nerf_pl/blob/52aeb387da64a9ad9a0f914ea9b049ffc598b20c/models/nerf.py#L4
    """

    def __init__(self, in_channels=3, num_pos_feats=768, n_freqs=8, logscale=True):
        super(Ego3DPositionEmbeddingMLP, self).__init__()
        self.n_freqs = n_freqs
        self.freq_out_channels = in_channels * (2 * n_freqs + 1)
        if logscale:
            freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (n_freqs - 1), n_freqs)
        
        center = torch.tensor([0., 0., 2.]).repeat(in_channels // 3)
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.register_buffer("center", center, persistent=False)

        self.position_embedding_head = nn.Sequential(
            nn.Linear(self.freq_out_channels, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """init with small weights to maintain stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)

    @torch.no_grad()
    def frequency_encoding(self, xyz):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        x \in [-2, 2]
        y \in [-2, 2]
        z \in [0., 4]
        Inputs:
            x: (b n m)
        Outputs:
            out: (b n o)
        """
        xyz_n = ((xyz - self.center) / 2.0).to(self.freq_bands.dtype)
        xyz_feq = xyz_n.unsqueeze(-1) * self.freq_bands  # (b n m 1)
        sin_xyz, cos_xyz = torch.sin(xyz_feq), torch.cos(xyz_feq)  # (b n m nf)
        encoding = torch.cat([xyz_n.unsqueeze(-1), sin_xyz, cos_xyz], -1).reshape(*xyz.shape[:2], -1)
        return encoding

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, N, F)."""
        # TODO: encoding with 3D position
        freq_encoding = self.frequency_encoding(xyz)
        position_embedding = self.position_embedding_head(freq_encoding)
        return position_embedding

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

class LawModel(nn.Module):
    def __init__(self, 
            config: TransfuserConfig,
            ):
        super().__init__()
        # self.image_encoder = timm.create_model(
        #     config.image_architecture, pretrained=False, features_only=True
        # )
        # self.image_encoder = ResNet34Backbone(config=config,pretrained=True)
        # self._backbone = TransfuserBackbone(config)
        #pretrained here
        self.image_encoder = timm.create_model(
                    config.image_architecture, pretrained=False, features_only=True
                )
        self.image_encoder.load_state_dict(torch.load('/home/zyp/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth'),strict=False)
        if config.use_depth:
            self.position_embedding_3d = Ego3DPositionEmbeddingMLP(
                2**2 * 3, num_pos_feats=config.tf_d_model, n_freqs=8
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

    def backproject_patch(self, K: torch.Tensor, lidar2cam: torch.Tensor, depth: torch.Tensor, patch_size=32, reso=2) -> torch.Tensor:
        """
        Backproject depth map to 3D points in camera coordinate.
        Args:
            K: camera intrinsic matrix (b 3 3)
            depth: depth map (b 1 h w)
            pixel_offset: offset to the pixel coordinate
        """
        # __import__("ipdb").set_trace()
        b = len(depth)
        c, h, w = depth[0].shape
        hp, wp = h // patch_size, w // patch_size
        sub_hp = sub_wp = reso
        device = depth.device
        patch_depth = torch.nn.functional.interpolate(depth, size=(hp * reso, wp * reso), mode="area").reshape(b, c, -1)
        coords = get_locations_reso(hp, wp, device, self.stride, reso)
        p_cam = (inv(K.float()) @ coords.float()) * patch_depth  # (b 3 3) @ (3 hw) -> (b 3 hw) * (b 1 hw) -> (b 3 hw)
        p_cam = torch.cat([p_cam, torch.ones_like(p_cam[:, 0:1, :])], dim=1)
        p_lidar = (inv(lidar2cam) @ p_cam)[:, 0:3, :]
        patch_p_lidar = p_lidar.reshape(b, 3, hp, sub_hp, wp, sub_wp).permute(0, 2, 4, 3, 5, 1).reshape(b, hp * wp, -1)
        return patch_p_lidar

    def get_depth_embed(self,K,lidar2cam,depth,device,stride,repo):
        # lidar2img = lidar2img.detach()
        # p11_start_time=time.time()
        lidar2cam = lidar2cam.detach()
        # cam2lidar = lidar2cam.inverse().detach()
        # K = torch.matmul(lidar2img, cam2lidar)[:, 0:3, 0:3].detach()
        xyz = self.backproject_patch(
            K, lidar2cam, depth, patch_size=self.stride, reso=2
        )
        # p11_end_time=time.time()
        img_pos3d = self.position_embedding_3d(xyz)
        # p12_end_time=time.time()
        # p11_time=p11_end_time-p11_start_time
        # p12_time=p12_end_time-p11_end_time
        # print("depth_embed_time_p11:",p11_time)
        # print("depth_embed_time_p12:",p12_time)
        return img_pos3d

    def forward_test(self, features) -> Dict[str, torch.Tensor]:
        camera_feature = features['camera_feature']
        # camera_feature_l=F.interpolate(camera_feature[:,:,:,:272], size=(256, 288), mode='bilinear', align_corners=False)
        # camera_feature_r=F.interpolate(camera_feature[:,:,:,-272:], size=(256, 288), mode='bilinear', align_corners=False)
        # camera_feature=torch.cat([camera_feature_l,camera_feature[:,:,:,272:-272],camera_feature_r],dim=3)

        status_feature = features['status_feature']

        batch_size = status_feature.shape[0]
        device= camera_feature.device

        # camera_feature=camera_feature[...,272:-272]
        # img_feat = self.image_encoder(camera_feature)[-1]
        # _,c,h,w=img_feat.shape
        # # =============================== depth embedding =======================================
        # K_left=resize_K(256,272,features['K_left'],288,256)
        # K_right=resize_K(256,272,features['K_right'],288,256)
        # depth_l=F.interpolate(features['depth_l'].unsqueeze(1), size=(256, 288), mode='nearest')
        # depth_r=F.interpolate(features['depth_r'].unsqueeze(1), size=(256, 288), mode='nearest')
        # left_pose=self.get_depth_embed(K_left.float(),features['lidar2cam_left'].float(),
        #                                 depth_l.float(),device,stride=self.stride,repo=2)
        # right_pose=self.get_depth_embed(K_right.float(),features['lidar2cam_right'].float(),
        #                                 depth_r.float(),device,stride=self.stride,repo=2)
        # front_pose=self.get_depth_embed(features['K_front'].float(),features['lidar2cam_front'].float(),
        #                                 features['depth_f'].float().unsqueeze(1),device,stride=self.stride,repo=2)
        
        # left_pose=left_pose.reshape(batch_size,8,9,256)
        # right_pose=right_pose.reshape(batch_size,8,9,256)
        # front_pose=front_pose.reshape(batch_size,8,15,256)
        # img_pos3d=torch.cat([left_pose,front_pose,right_pose],dim=2)
        
        # img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)
        # img_feat = self.image_fc(img_feat.clone()) # 512 -> 256
        # img_feat=img_feat.reshape(batch_size,h,w,256)
        # img_feat+=img_pos3d
        # img_feat=img_feat.reshape(batch_size,-1,256)

        # # =============================== img backbone =======================================
        # backbone_start_time=time.time()
        img_feat = self.image_encoder(camera_feature)[-1]
        img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)
        img_feat = self.image_fc(img_feat.clone()) # 512 -> 256
        # backbone_end_time=time.time()
        # backbone_time=backbone_end_time-backbone_start_time

        # =============================== add depth embedding v2=======================================
        if self.use_depth:
            # depth_embed_start_time=time.time()
            # left_pose=self.get_depth_embed(features['K_l'],features['lidar2img_l'],
            #                                 features['depth_l'][:,None,:,:],device,stride=self.stride,repo=2)
            # right_pose=self.get_depth_embed(features['K_r'],features['lidar2img_r'],
            #                                 features['depth_r'][:,None,:,:],device,stride=self.stride,repo=2)
            # front_pose=self.get_depth_embed(features['K_f'],features['lidar2img_f'],
            #                                 features['depth_f'][:,None,:,:],device,stride=self.stride,repo=2)
            # left_pose=left_pose.reshape(batch_size,8,9,256)
            # right_pose=right_pose.reshape(batch_size,8,9,256)
            # front_pose=front_pose.reshape(batch_size,8,15,256)
            # img_pos3d=torch.cat([left_pose,front_pose,right_pose],dim=2)
            # img_feat+=img_pos3d.reshape(batch_size,-1,256)
            # depth_embed_end_time=time.time()
            # depth_embed_time=depth_embed_end_time-depth_embed_start_time

            img_pos_l=self.position_embedding_3d(features['left_xyz'].squeeze())
            img_pos_f=self.position_embedding_3d(features['front_xyz'].squeeze())
            img_pos_r=self.position_embedding_3d(features['right_xyz'].squeeze())
            img_pos3d=torch.cat([img_pos_l,img_pos_f,img_pos_r],dim=1)
            img_feat+=img_pos3d

        # =============================== keyval trans=======================================
        # keyval_start_time=time.time()
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.cat([img_feat, status_encoding[:, None]], dim=1)
        keyval = keyval.clone() + self._keyval_embedding.weight[None, ...]

        keyval_final = keyval
        # keyval_end_time=time.time()
        # keyval_time=keyval_end_time-keyval_start_time

        # traj decoder
        # traj_start_time=time.time()
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval_final)
        trajectory = self._trajectory_head(query_out)
        # traj_end_time=time.time()
        # traj_time=traj_end_time-traj_start_time

        # print("backbone_time:",backbone_time)
        # print("depth_embed_time:",depth_embed_time)
        # print("keyval_time:",keyval_time)
        # print("traj_time:",traj_time)

        return trajectory

    def forward_train(self, features) -> Dict[str, torch.Tensor]:
        camera_feature = features['camera_feature']
        # camera_feature_l=F.interpolate(camera_feature[:,:,:,:272], size=(256, 288), mode='bilinear', align_corners=False)
        # camera_feature_r=F.interpolate(camera_feature[:,:,:,-272:], size=(256, 288), mode='bilinear', align_corners=False)
        # camera_feature=torch.cat([camera_feature_l,camera_feature[:,:,:,272:-272],camera_feature_r],dim=3)

        status_feature = features['status_feature']

        batch_size = status_feature.shape[0]
        device= camera_feature.device

        # # camera_feature=camera_feature[...,272:-272]
        # img_feat = self.image_encoder(camera_feature)[-1]
        # _,c,h,w=img_feat.shape
        # # =============================== depth embedding =======================================
        # K_left=resize_K(256,272,features['K_left'],288,256)
        # K_right=resize_K(256,272,features['K_right'],288,256)
        # depth_l=F.interpolate(features['depth_l'].unsqueeze(1), size=(256, 288), mode='nearest')
        # depth_r=F.interpolate(features['depth_r'].unsqueeze(1), size=(256, 288), mode='nearest')
        # left_pose=self.get_depth_embed(K_left.float(),features['lidar2cam_left'].float(),
        #                                 depth_l.float(),device,stride=self.stride,repo=2)
        # right_pose=self.get_depth_embed(K_right.float(),features['lidar2cam_right'].float(),
        #                                 depth_r.float(),device,stride=self.stride,repo=2)
        # front_pose=self.get_depth_embed(features['K_front'].float(),features['lidar2cam_front'].float(),
        #                                 features['depth_f'].float().unsqueeze(1),device,stride=self.stride,repo=2)
        
        # left_pose=left_pose.reshape(batch_size,8,9,256)
        # right_pose=right_pose.reshape(batch_size,8,9,256)
        # front_pose=front_pose.reshape(batch_size,8,15,256)
        # img_pos3d=torch.cat([left_pose,front_pose,right_pose],dim=2)
        
        # img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)
        # img_feat = self.image_fc(img_feat.clone()) # 512 -> 256
        # img_feat=img_feat.reshape(batch_size,h,w,256)
        # img_feat+=img_pos3d
        # img_feat=img_feat.reshape(batch_size,-1,256)

        # # =============================== depth embedding v2 =======================================
        # backbone_start_time=time.time()
        img_feat = self.image_encoder(camera_feature)[-1]
        img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)
        img_feat = self.image_fc(img_feat.clone()) # 512 -> 256
        # backbone_end_time=time.time()
        # backbone_time=backbone_end_time-backbone_start_time

        # =============================== add depth embedding v2=======================================
        if self.use_depth:
            # depth_embed_time1=time.time()
            # left_pose=self.get_depth_embed(features['K_l'],features['lidar2img_l'],
            #                                 features['depth_l'][:,None,:,:],device,stride=self.stride,repo=2)
            # right_pose=self.get_depth_embed(features['K_r'],features['lidar2img_r'],
            #                                 features['depth_r'][:,None,:,:],device,stride=self.stride,repo=2)
            # front_pose=self.get_depth_embed(features['K_f'],features['lidar2img_f'],
            #                                 features['depth_f'][:,None,:,:],device,stride=self.stride,repo=2)
            # depth_embed_time2=time.time()
            # left_pose=left_pose.reshape(batch_size,8,9,256)
            # right_pose=right_pose.reshape(batch_size,8,9,256)
            # front_pose=front_pose.reshape(batch_size,8,15,256)
            # img_pos3d=torch.cat([left_pose,front_pose,right_pose],dim=2)
            # img_feat+=img_pos3d.reshape(batch_size,-1,256)
            # depth_embed_time3=time.time()

            # depth_embed_time_p1=depth_embed_time2-depth_embed_time1
            # depth_embed_time_p2=depth_embed_time3-depth_embed_time2
            img_pos_l=self.position_embedding_3d(features['left_xyz'])
            img_pos_f=self.position_embedding_3d(features['front_xyz'])
            img_pos_r=self.position_embedding_3d(features['right_xyz'])
            img_pos3d=torch.cat([img_pos_l,img_pos_f,img_pos_r],dim=1)
            img_feat+=img_pos3d

        # =============================== keyval trans=======================================
        # keyval_start_time=time.time()
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.cat([img_feat, status_encoding[:, None]], dim=1)
        keyval = keyval.clone() + self._keyval_embedding.weight[None, ...]

        keyval_final = keyval
        # keyval_end_time=time.time()
        # keyval_time=keyval_end_time-keyval_start_time

        # traj decoder
        # traj_start_time=time.time()
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval_final)
        trajectory = self._trajectory_head(query_out)
        # traj_end_time=time.time()
        # traj_time=traj_end_time-traj_start_time

        # print("backbone_time:",backbone_time)
        # print("depth_embed_time_p1:",depth_embed_time_p1)
        # print("depth_embed_time_p2:",depth_embed_time_p2)
        # print("keyval_time:",keyval_time)
        # print("traj_time:",traj_time)

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


    