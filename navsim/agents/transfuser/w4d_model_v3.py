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
            '/vepfs-mlp2/c20250502/haoce/wlb/world4drive/ckpts/resnet34-333f7ec4.pth', 
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
        elif self._num_mode == 3:
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
        else:
            self._trajectory_head = TrajectoryHead(self._num_poses, config.tf_d_ffn, config.tf_d_model)


        # 多模态词表导入
        if self._config.use_vocab_trajs:
            vocab_trajs_dict_path = self._config.vocab_trajs_dict_path
            vocab_trajs_path = self._config.vocab_trajs_path
            vocab_trajs_dict = pickle.load(open(vocab_trajs_dict_path, 'rb'))
            vocab_trajs = np.load(vocab_trajs_path, allow_pickle=True).item()




        # loss weight
        self.wm_loss_weight=0.2
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
        self.use_wm = getattr(config, 'use_wm', False)
        self.use_wm_training = getattr(config, 'use_wm_training', False)

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

    def forward_test(self, features) -> Dict[str, torch.Tensor]:
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
        keyval_final = keyval  # [bs, 256+1, 256]

        # =============================== query =============================================
        ego_query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)  # [bs, num_mode * num_poses, 256] 或 [bs, num_poses, 256]

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
            else:
                mode_ref = self.waypoint_mode_ref.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, num_mode, num_poses, 2]
                
            if self._num_mode in [20, 18]:
                mode_ref_feat_sin = self.gen_sineembed_for_position(mode_ref)  # [bs, num_mode, num_poses, 256]
                mode_ref_feat = self._mode_embedding(mode_ref_feat_sin)  # [bs, num_mode, num_poses, 256]
            else:
                mode_ref_feat = self._mode_embedding(mode_ref)  # [bs, num_mode, num_poses, 256]
            ego_query = ego_query.clone() + mode_ref_feat.reshape(batch_size, -1, mode_ref_feat.shape[-1]).clone()

        # =============================== 轨迹特征提取 =======================================
        ego_query_out = self._tf_decoder(ego_query, keyval_final)

        ##==============================控制命令预测==============================================
        cmd_query = self._cmd_pred_query.weight[None, ...].repeat(batch_size, 1, 1)  # [bs, 1, 256]
        cmd_query_out = self._cmd_head_decoder(cmd_query, keyval_final)  #

        cmd_logits = self._cmd_mlp(cmd_query_out).squeeze(1)  # [bs, 4]
        cmd_pred = torch.argmax(cmd_logits, dim=-1)  # [bs]

        # =============================== 轨迹输出 ===========================================
        trajectory = self._trajectory_head(ego_query_out, cmd_pred, cmd)
      


        trajectory['cmd_logits'] = cmd_logits
        trajectory['cmd_pred'] = cmd_pred
        trajectory['cmd_gt'] = cmd
        
        return trajectory

    def forward_train(self, features) -> Dict[str, torch.Tensor]:
        camera_feature = features['camera_feature']

        status_feature = features['status_feature']

        cmd = status_feature[..., :4]  # [bs, 4] one-hot
        cmd = torch.argmax(cmd, dim=-1)  # [bs] int


        batch_size = status_feature.shape[0]
        device= camera_feature.device

        img_feat = self.image_encoder(camera_feature)[-1]
        img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)
        img_feat = self.image_fc(img_feat.clone()) # 512 -> 256 # [bs, 8*32, 256]


        # =============================== keyval trans=======================================
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.cat([img_feat, status_encoding[:, None]], dim=1)
        keyval = keyval.clone() + self._keyval_embedding.weight[None, ...]

        keyval_final = keyval   # [bs, 64, 257 , 256]


        ego_query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)   # [bs, num_mode * num_poses, 256]

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
            else:
                mode_ref = self.waypoint_mode_ref.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, num_mode, num_poses, 2]
                
            if self._num_mode in [20, 18]:
                mode_ref_feat_sin = self.gen_sineembed_for_position(mode_ref)  # [bs, num_mode, num_poses, 256]
                mode_ref_feat = self._mode_embedding(mode_ref_feat_sin)  # [bs, num_mode, num_poses, 256]
            else:
                mode_ref_feat = self._mode_embedding(mode_ref)  # [bs, num_mode, num_poses, 256]
            ego_query = ego_query.clone() + mode_ref_feat.reshape(batch_size, -1, mode_ref_feat.shape[-1]).clone()



        
        ##==============================轨迹特征提取========================================

        ego_query_out = self._tf_decoder(ego_query, keyval_final)

        ##==============================轨迹特征提取========================================

       

        ##==============================控制命令预测==============================================
        cmd_query = self._cmd_pred_query.weight[None, ...].repeat(batch_size, 1, 1)  # [bs, 1, 256]
        cmd_query_out = self._cmd_head_decoder(cmd_query, keyval_final)  #

        cmd_logits = self._cmd_mlp(cmd_query_out).squeeze(1)  # [bs, 4]
        cmd_pred = torch.argmax(cmd_logits, dim=-1)  # [bs]


        trajectory = self._trajectory_head(ego_query_out, cmd_pred, cmd)

        trajectory['cmd_logits'] = cmd_logits
        trajectory['cmd_pred'] = cmd_pred
        trajectory['cmd_gt'] = cmd
       

        # # 如果rl，则输出方差
        # if self.config.training_mode == "ft" or self.config.training_mode == 'bcsl':
        #     traj_var = self._var_head(ego_query_out)
        #     trajectory['trajectory_var'] = traj_var
        #     trajectory['keyval_final'] = keyval_final

        # TODO: wm 架构
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
        # ========================= 多模态监督 =========================
        if self._config.num_mode and logging_prefix == 'train':
            all_trajs = predictions.get("all_trajectories")  # [B, M, T, 3]
            gt = targets["trajectory"]  # [B, T, 3]
            B, M = all_trajs.shape[0], all_trajs.shape[1]

            # 计算 per-mode 误差 (ADE/FDE)
            diff = all_trajs - gt[:, None]  # [B,M,T,3]
            l1_per_t = diff.abs().mean(-1)  # [B,M,T]
            ade = l1_per_t.mean(-1)  # [B,M]
            fde = l1_per_t[:, :, -1]  # [B,M]

            if self.soft_resp:  # 软责任
                # 选择加权依据（FDE 或 ADE）
                dist_for_weight = fde if self.use_fde_for_weight else ade  # [B,M]
                # Top-K 过滤
                if self.soft_resp_topk and self.soft_resp_topk < M:
                    # 取距离最小的K个索引
                    topk_val, topk_idx = torch.topk(-dist_for_weight, k=self.soft_resp_topk, dim=1)  # 负号=最小
                    mask = torch.zeros_like(dist_for_weight, dtype=torch.bool)
                    mask.scatter_(1, topk_idx, True)
                    # 对未入选的模式给予一个很大的距离(或直接置 -inf logit)
                    large = 1e6
                    dist_masked = dist_for_weight.clone()
                    dist_masked[~mask] = large
                    weights = torch.softmax(-dist_masked / max(self.soft_resp_tau, 1e-6), dim=1)
                else:
                    weights = torch.softmax(-dist_for_weight / max(self.soft_resp_tau, 1e-6), dim=1)  # [B,M]

                # 回归损失: 每模式 L1 平均 (也可用 l1_per_t.mean(-1))
                per_mode_traj_loss = l1_per_t.mean(-1)  # [B,M]
                trajectory_loss = (per_mode_traj_loss * weights).sum(-1).mean()

                # 分类软标签损失
                if "cls_logits" in predictions:
                    log_probs = F.log_softmax(predictions["cls_logits"], dim=-1)
                else:
                    log_probs = None

                if log_probs is not None:
                    cls_loss = -(weights * log_probs).sum(-1).mean()
                    loss_dict["cls_loss"] = cls_loss * self.traj_cls_loss_weight
                    if self.cls_entropy_weight > 0:
                        probs = log_probs.exp()
                        entropy = -(probs * log_probs).sum(-1).mean()
                        # 希望保持一定熵 → 最大化熵 → 加上 -entropy 系数为负, 这里直接减去熵
                        loss_dict["cls_entropy"] = -entropy * self.cls_entropy_weight
            # elif self._config.num_mode and logging_prefix == 'train' and self._config.use_vocab_trajs:
            #     tokens = features['token']
            #     N = 19
            #     for token in tokens:
            #         cur_vocab_trajs_dict = vocab_trajs_dict[token]  
            #         # 根据pdms分数，取top N
            #         top_n_idx = cur_vocab_trajs_dict['pdm_score'].topk(N, dim=0).indices
            #         top_n_trajs = vocab_trajs[top_n_idx]  # np (N, 40, 3)
            #     pass
            else:
                # 原 hard winner-take-all 方式
                traj_gt_dist = torch.norm(all_trajs[..., :2] - gt[:, None, :, :2], p=1, dim=-1)  # [B,M,T]
                fde_dist = traj_gt_dist[:, :, -1]  # [B,M]
                best_mode = torch.argmin(fde_dist, dim=-1)  # [B]
                best_traj = all_trajs[torch.arange(B), best_mode]
                trajectory_loss = torch.nn.functional.l1_loss(best_traj, gt)

                # 分类 hard label
                if "cls_logits" in predictions:
                    cls_loss = torch.nn.functional.cross_entropy(predictions["cls_logits"], best_mode)
                    loss_dict["cls_loss"] = cls_loss * self.traj_cls_loss_weight

            loss_dict["traj_loss"] = trajectory_loss * self.traj_loss_weight

            ## 控制命令预测loss
            if "cmd_logits" in predictions:
                cmd = predictions['cmd_gt']
                cmd_loss = torch.nn.functional.cross_entropy(predictions["cmd_logits"], cmd)
                loss_dict["cmd_loss"] = cmd_loss * self._config.traj_cmd_loss_weight

            # 多样性损失（终点排斥）—— 与软/硬责任无关
            if self.diversity_loss_weight > 0:
                endpoints = all_trajs[..., -1, :2]  # [B,M,2]
                pair_dist = torch.cdist(endpoints, endpoints)  # [B,M,M]
                m = pair_dist.shape[1]
                diag_mask = 1 - torch.eye(m, device=pair_dist.device)
                penalty = F.relu(self.diversity_margin - pair_dist) * diag_mask
                tri_mask = torch.triu(torch.ones_like(penalty), diagonal=1)
                penalty = penalty * tri_mask
                num_pairs = m * (m - 1) / 2
                diversity_loss = penalty.sum() / (B * num_pairs + 1e-6)
                loss_dict["diversity_loss"] = diversity_loss * self.diversity_loss_weight
        else:
            # 单模态或非训练阶段
            if "trajectory" in predictions:
                trajectory_loss = torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])
                loss_dict["traj_loss"] = trajectory_loss * self.traj_loss_weight

        # 世界模型损失
        if self.use_wm_training and 'wm_next_latent' in predictions:
            wm_loss_a = torch.nn.functional.mse_loss(predictions["wm_next_latent"], predictions["cur_latent"].detach())
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
                if use_pred.any():
                    print("Warning: Using predicted command for some samples in TrajectoryHead.")
                    pred_cmd[pred_cmd == 3] = 1
                best_traj_idx = torch.where(use_pred, pred_cmd, cmd)
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

