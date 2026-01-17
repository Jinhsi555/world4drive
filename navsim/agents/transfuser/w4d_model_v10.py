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
            '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/ckpts/resnet34-333f7ec4.pth', 
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

        
        # 编码多模态引导信息为嵌入
        self._mode_embedding = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )

        ############################################多模态相关初始化########################################

        # 控制命令预测头,不需要这个头了，直接使用轨迹预测结果来推断控制命令
        # self._cmd_pred_query = nn.Embedding(1, config.tf_d_model)  # single query

        # self._cmd_head_decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=config.tf_d_model,
        #     nhead=config.tf_num_head,
        #     dim_feedforward=config.tf_d_ffn,
        #     dropout=config.tf_dropout,
        #     batch_first=True,
        # )
        # self._cmd_head_decoder = nn.TransformerDecoder(self._cmd_head_decoder_layer, 1)

        # self._cmd_mlp = nn.Sequential(
        #     nn.Linear(config.tf_d_model, config.tf_d_ffn),
        #     nn.ReLU(),
        #     nn.Linear(config.tf_d_ffn, 3),  # 3 classes 训练集中只有三种命令,测试集当中有第四种命令
        # )


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
        self.wm_loss_weight= config.wm_loss_weight if hasattr(config, 'wm_loss_weight') else 0.2
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

            

        self.stride=32
        self.use_all_mb = config.use_all_mb if hasattr(config, 'use_all_mb') else False

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
    def wm_prediction(self, cur_img_feat, cur_traj):
        batch_size = cur_img_feat.shape[0]
        cur_traj = cur_traj.reshape(batch_size, 1, -1)  # [bs, 1, num_poses*3]
        traj_token = self.traj_encoder(cur_traj)  # [bs, 1, d_model]
        wm_next_latent = torch.zeros_like(cur_img_feat)  # [bs, num_token, d_model]
        input_tokens = torch.cat([cur_img_feat, traj_token], dim=1)  # [bs, num_token+1, d_model]
        wm_query = self._wm_query_embedding.weight[None, ...].repeat(batch_size, 1, 1)  # [bs, num_token, d_model]
        wm_next_latent = self._wm_decoder(wm_query, input_tokens)  # [bs, num_token, d_model]
        return wm_next_latent
    
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



        # AR过程进行逐步残差预测,即每一步的query都与前一步的预测结果相关联，
        if self._config.use_ar and self._num_mode == 4:
            kv = keyval_final
            all_cmd_temp_traj = []
            all_temp_traj = []
            for i in range(1, self._num_poses + 1):
                # 构造第 i 步 query 与对应的 mode_ref
                q = getattr(self, f'waypoint_query_ar_{i}').weight.unsqueeze(0).repeat(batch_size, 1, 1)
                m = getattr(self, f'waypoint_mode_ref_ar_{i}').weight.unsqueeze(0).repeat(batch_size, 1, 1)
                m_feat = self._mode_embedding(m)
                ego_q = q + m_feat

                # Transformer 解码并生成所有模态轨迹
                decoder = getattr(self, f'_tf_decoder_ar_{i}')
                out = decoder(ego_q, kv)
                last_trajectory = self._trajectory_head.forward_ar(out, cmd=cmd)
                all_traj = last_trajectory['all_trajectories']  # [bs, num_mode*i, 8, 3]


                # 将本步预测嵌入后追加到 kv，用于下一步
                # traj_encoder_ar = getattr(self, f'traj_encoder_ar_{i}')
                # embed = traj_encoder_ar(all_traj).flatten(1, 2)    # [bs, num_mode*i*num_poses, d_model]
                embed = self._traj_encoder_ar(all_traj).flatten(1, 2)
                kv = torch.cat([kv, embed], dim=1)
                all_cmd_temp_traj.append(all_traj)
                all_temp_traj.append(last_trajectory['trajectory'])
            
            trajectory = last_trajectory
            trajectory['all_cmd_temp_trajectories'] = all_cmd_temp_traj  # list of [bs, num_mode*i, 8, 3]
            trajectory['all_temp_trajectories'] = all_temp_traj  # list of [bs, i, 3]
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
        
        if self._config.num_mode and logging_prefix == 'train':
            all_trajs = predictions.get("all_trajectories")  # [B, M, T, 3]
            gt = targets["trajectory"]  # [B, T, 3]
            B, M = all_trajs.shape[0], all_trajs.shape[1]

            # # 原 hard winner-take-all 方式
            # traj_gt_dist = torch.norm(all_trajs[..., :2] - gt[:, None, :, :2], p=1, dim=-1)  # [B,M,T]
            # fde_dist = traj_gt_dist[:, :, -1]  # [B,M]
            # best_mode = torch.argmin(fde_dist, dim=-1)  # [B]
            # best_traj = all_trajs[torch.arange(B), best_mode]
            # trajectory_loss = torch.nn.functional.l1_loss(best_traj, gt)

            # # 分类 hard label
            # if "cls_logits" in predictions:
            #     cls_loss = torch.nn.functional.cross_entropy(predictions["cls_logits"], best_mode)
            #     loss_dict["cls_loss"] = cls_loss * self.traj_cls_loss_weight
            
            if not self._config.use_ar:
                trajectory_loss = torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])
                loss_dict["traj_loss"] = trajectory_loss * self.traj_loss_weight

            ## 控制命令预测loss
            # if "cmd_logits" in predictions:
            #     cmd = predictions['cmd_gt']
            #     cmd_loss = torch.nn.functional.cross_entropy(predictions["cmd_logits"], cmd)
            #     loss_dict["cmd_loss"] = cmd_loss * self._config.traj_cmd_loss_weight

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
            if "trajectory" in predictions and not self._config.use_ar:
                trajectory_loss = torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])
                loss_dict["traj_loss"] = trajectory_loss * self.traj_loss_weight

        # 世界模型损失
        if self._config.use_wm and 'next_latent' in predictions:
            wm_loss = torch.nn.functional.mse_loss(predictions["pred_next_latent"], predictions["next_latent"].detach())
            loss_dict["wm_loss"] = wm_loss * self.wm_loss_weight

        if self._config.use_ar:
            # AR 逐步监督损失
            all_temp_trajectories = predictions.get('all_temp_trajectories', [])  # list of [bs, num_mode*i, 8, 3]
           
            gt = targets["trajectory"]  # [B, T, 3]
            for i, temp_traj in enumerate(all_temp_trajectories):
                # 只计算前面7步
                if i >= 7:
                    break
                T_i = i + 1
                gt_i = gt[:, :T_i, :]  # [B,T_i,3]
                loss_ar = torch.nn.functional.l1_loss(temp_traj, gt_i)
                loss_dict[f'ar_loss_step_{T_i}'] = loss_ar * self.traj_loss_weight

            trajectory_loss = torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])
            loss_dict["traj_loss"] = trajectory_loss * self.traj_loss_weight

            

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
                use_pred = (cmd == 3)
                # 如果存在命令3，则打印警告
                if use_pred.any():
                    print("Warning: Using predicted command for some samples in TrajectoryHead.")
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

