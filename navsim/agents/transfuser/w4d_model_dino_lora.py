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
# from navsim.agents.transfuser.utils import RotaryPositionEmbedding, BridgeAttentionTransformer, timed, apply_rope

import torchvision.models as models
import torch.nn.functional as F
import torchvision.models as models
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
import timm
from torch.linalg import inv

from transformers import AutoImageProcessor, AutoModel
from transformers.models.dinov2 import Dinov2Model
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from peft import LoraConfig, get_peft_model

class W4DModel(nn.Module):
    def __init__(self, 
            config: TransfuserConfig,
            ):
        super().__init__()

        self._config = config

        # learnable scene query
        self.scene_embeds = nn.Parameter(torch.randn(1, config.num_view, config.num_scene_query_token, config.tf_d_model)*1e-6, requires_grad=True)
        
        # define dino-LoRA encoder for vision query
        self.dino_encoder = AutoModel.from_pretrained("/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-small")
        self.processor = AutoImageProcessor.from_pretrained("/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-small")
        self.processor.crop_size = {'height': 224, 'width': 896}

        def _apply_lora_to_dino_encoder(encoder):
            lora_config = LoraConfig(
                r=self._config.lora_rank,
                lora_alpha=self._config.lora_alpha,
                target_modules=self._config.lora_target_modules,
                lora_dropout=self._config.lora_dropout,
            )
            lora_encoder = get_peft_model(encoder, lora_config)
            print("✅ LoRA applied to dino encoder.")

            for name, param in lora_encoder.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # List the LoRA adapter applied to the backbone
            for name, module in lora_encoder.named_modules():
                if "lora" in name:
                    print(f"    - {name}")

            # Print trainable parameter statistics
            # print("Trainable parameters in LoRA backbone:")
            trainable_params = sum(p.numel() for p in lora_encoder.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in lora_encoder.parameters())
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")

            return lora_encoder
        
        self.lora_dino_encoder = _apply_lora_to_dino_encoder(self.dino_encoder)
        
        # # geometry learnable query
        # self.geometry_query = nn.Parameter(
        #     torch.randn(1, config.num_view, config.num_scene_query_token, config.tf_d_model),
        #     requires_grad=True,
        # )
        
        # self.geometry_decoder = nn.ModuleList([
        #     nn.TransformerDecoderLayer(
        #         d_model=config.tf_d_model,
        #         nhead=config.tf_num_head,
        #         dim_feedforward=config.tf_d_ffn,
        #         dropout=config.tf_dropout,
        #         batch_first=True,
        #     ) for _ in range(config.num_view)
        # ])

        # self.geometry_projector = nn.Sequential(
        #     nn.Linear(config.tf_d_model, 2*config.tf_d_model),
        #     nn.ReLU(),
        #     nn.Linear(2*config.tf_d_model, 2*config.tf_d_model),
        # )

        # self.geometry_loss_weight = 0.2

        # refine net
        # self.refine_traj_decoder = BridgeAttentionTransformer(config)

        # self.image_fc = nn.Linear(512, 256)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._num_poses = config.trajectory_sampling.num_poses

        #TODO: petr position_embedding
        num_keyval = config.num_scene_query_token+1

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
            self._trajectory_head_for_refine = TrajectoryHead(self._num_poses, config.tf_d_ffn, config.tf_d_model, config.num_mode)
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

               
        # world model
        self.stride=32
        self.use_all_mb = config.use_all_mb if hasattr(config, 'use_all_mb') else False

        if self._config.use_wm:
            num_wm_query = num_keyval-1  # 16
            self._wm_query_embedding = nn.Embedding(num_wm_query, config.tf_d_model)
            wm_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.tf_d_model,
                nhead=config.tf_num_head,
                dim_feedforward=config.tf_d_ffn,
                dropout=config.tf_dropout,
                batch_first=True,
            )
            self._wm_decoder = nn.TransformerDecoder(wm_decoder_layer, config.tf_num_layers) # input: Bz, num_token, d_model
    
    def get_dino_features_with_scene_query(self, model: Dinov2Model, inputs: dict, scene_query: torch.Tensor):
        output_hidden_states = inputs.get("output_hidden_states", None)
        pixel_values = inputs.get("pixel_values", None)

        if output_hidden_states is None:
            output_hidden_states = model.config.output_hidden_states

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = model.embeddings(pixel_values, None)
        embedding_output = embedding_output.reshape(pixel_values.shape[0], -1, self._config.tf_d_model)
        embedding_output = torch.cat([scene_query, embedding_output], dim=1)

        encoder_outputs: BaseModelOutput = model.encoder(
            embedding_output, None, output_hidden_states=output_hidden_states
        )
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = model.layernorm(sequence_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
        )

    def forward_test(self, features) -> Dict[str, torch.Tensor]:
        # dino_feature, geometry_feature = features['camera_feature']
        image_feature = features['camera_feature']
        inputs = self.processor(images=image_feature, return_tensors="pt").to('cuda')

        status_feature = features['status_feature']

        cmd = status_feature[..., :4]  # [bs, 4] one-hot
        cmd = torch.argmax(cmd, dim=-1)  # [bs] int


        batch_size = status_feature.shape[0]
        device= self.scene_embeds.device

        # img_feat = self.image_encoder(camera_feature)[-1]
        # img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)
        # img_feat = self.image_fc(img_feat.clone()) # 512 -> 256 # [bs, 8*32, 256]
        
        # # v1: concat 3 views in sequence length dimension  [bs, 3, 256, 1024] -> [bs, 768, 1024]
        # img_feat = camera_feature[:, :, 1:, :].reshape(b, -1, dim)
        

        # ==================== Dino Encoder with scene query =================
        # v2: use learnable query do cross attention with each view dino feature
        init_scene_query = self.scene_embeds.repeat(batch_size, 1, 1, 1)  # [1, 3, 16, 384] -> [bs, 3, 16, 384]
        b, n, seq_len, dim = init_scene_query.shape  # [b, 3, 16, 384]
        init_scene_query = init_scene_query.reshape(-1, seq_len, dim)
        image_scene_query = self.get_dino_features_with_scene_query(self.lora_dino_encoder, inputs, init_scene_query).last_hidden_state  # [bs*1, 16+1024, 384]
        image_scene_query = image_scene_query.reshape(b, n, -1, dim)[..., :self._config.num_scene_query_token, :]  # [bs, 1, 16, 384]
            
        scene_query = image_scene_query.reshape(b, -1, dim)  # [b, 16, 1024]

        # # geometry query to interact with dino feature of each view
        # init_geometry_query = self.geometry_query.repeat(batch_size, 1, 1, 1)  # [1, 3, 519, 1024] -> [bs, 3, 519, 1024]
        # geometry_feat = torch.zeros_like(init_geometry_query)
        # for i in range(self._config.num_view):
        #     geometry_query = init_geometry_query[:, i]  # [bs, 519, 1024]
        #     camera_kv = dino_feature[:, i, 1:]  # [bs, seq_len, dim]
        #     geometry_feat[:, i] = self.geometry_decoder[i](geometry_query, camera_kv)
        
        # geometry_feat_for_refine = geometry_feat[:, :, 7:, :].reshape(b, -1, dim)  # [b, 3*512, 1024]
        # geometry_feat = self.geometry_projector(geometry_feat)  # [b, 3*519, 1024] -> [b, 3*519, 2048]

        # =============================== keyval trans =======================================
        status_encoding = self._status_encoding(status_feature)  # [bs, 1024]
        keyval = torch.cat([scene_query, status_encoding[:, None]], dim=1)  # [bs, 16+1, 256]
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
            elif self._num_mode == 4:
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
        # 取一个除了3外，最大概率的类别作为预测值
        # cmd_pred = torch.argmax(cmd_logits[:, :3], dim=-1)  # [bs]
        cmd_pred = torch.argmax(cmd_logits, dim=-1)  # [bs]

        # =============================== 轨迹输出 ===========================================
        trajectory = {}
        trajectory['first_trajectory'] = self._trajectory_head(ego_query_out, cmd=cmd)
      
        trajectory['cmd_logits'] = cmd_logits
        trajectory['cmd_pred'] = cmd_pred
        trajectory['cmd_gt'] = cmd

        # # record the geometry feature and gt
        # trajectory['geometry_predict'] = geometry_feat
        # trajectory['geometry_gt'] = geometry_feature[:, 0, ...]

        if self._config.use_wm:
            wm_keyval = keyval_final
            wm_target = keyval_final
            wm_query = self._wm_query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
            wm_next_latent = self._wm_decoder(wm_query, wm_keyval)
            trajectory['wm_next_latent']=wm_next_latent
            trajectory['cur_latent']=wm_target

            # refine the trajectory
            # refined_traj_feat = self.refine_traj_decoder(ego_query_out, wm_next_latent, geometry_feat_for_refine)
            # refined_traj = self._trajectory_head_for_refine(refined_traj_feat, cmd=cmd)
            # trajectory['refined_trajectory'] = refined_traj

            return trajectory
        else:
            return trajectory

    # @timed
    def forward_train(self, features) -> Dict[str, torch.Tensor]:
        # unpack the camera_feature to get dino feature and geometry feature
        # dino_feature, geometry_feature = features['camera_feature']
        image_feature = features['camera_feature']
        inputs = self.processor(images=image_feature, return_tensors="pt").to('cuda')

        status_feature = features['status_feature']

        cmd = status_feature[..., :4]  # [bs, 4] one-hot
        cmd = torch.argmax(cmd, dim=-1)  # [bs] int


        batch_size = status_feature.shape[0]
        device= self.scene_embeds.device

        # img_feat = self.image_encoder(camera_feature)[-1]
        # img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)
        # img_feat = self.image_fc(img_feat.clone()) # 512 -> 256 # [bs, 8*32, 256]
        
        # # v1: concat 3 views in sequence length dimension  [bs, 3, 256, 1024] -> [bs, 768, 1024]
        # img_feat = camera_feature[:, :, 1:, :].reshape(b, -1, dim)
        

        # ==================== Dino Encoder with scene query =================
        # v2: use learnable query do cross attention with each view dino feature
        init_scene_query = self.scene_embeds.repeat(batch_size, 1, 1, 1)  # [1, 3, 16, 384] -> [bs, 3, 16, 384]
        b, n, seq_len, dim = init_scene_query.shape  # [b, 3, 16, 384]
        init_scene_query = init_scene_query.reshape(-1, seq_len, dim)
        image_scene_query = self.get_dino_features_with_scene_query(self.lora_dino_encoder, inputs, init_scene_query).last_hidden_state  # [bs*1, 16+1024, 384]
        image_scene_query = image_scene_query.reshape(b, n, -1, dim)[..., :self._config.num_scene_query_token, :]  # [bs, 1, 16, 384]
            
        scene_query = image_scene_query.reshape(b, -1, dim)  # [b, 16, 1024]

        # # geometry query to interact with dino feature of each view
        # start_time = time.time()

        # init_geometry_query = self.geometry_query.repeat(batch_size, 1, 1, 1)  # [1, 3, 519, 1024] -> [bs, 3, 519, 1024]
        # geometry_feat = torch.zeros_like(init_geometry_query)
        # for i in range(self._config.num_view):
        #     geometry_query = init_geometry_query[:, i]  # [bs, 519, 1024]
        #     camera_kv = dino_feature[:, i]  # [bs, 519, 1024]
        #     geometry_feat[:, i] = self.geometry_decoder[i](geometry_query, camera_kv)
        
        # geometry_feat_for_refine = geometry_feat[:, :, 7:, :].reshape(b, -1, dim)  # [b, 3*512, 1024]
        # geometry_feat = self.geometry_projector(geometry_feat)  # [b, 3, 519, 1024] -> [b, 3, 519, 2048]

        # print(f"geometry_decoder time: {time.time() - start_time}")

        # =============================== keyval trans =======================================
        start_time = time.time()

        status_encoding = self._status_encoding(status_feature)  # [bs, 384]
        keyval = torch.cat([scene_query, status_encoding[:, None]], dim=1)  # [bs, 16+1, 384]
        keyval = keyval.clone() + self._keyval_embedding.weight[None, ...]
        keyval_final = keyval  # [bs, 16+1, 384]

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
                mode_ref = self.waypoint_mode_ref.weight  # [num_mode, 256]
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
                    mode_ref = mode_ref.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, num_mode, num_poses, 256]
            else:
                mode_ref = self.waypoint_mode_ref.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, num_mode, num_poses, 2]
            
            if self._num_mode in [20, 18]:
                mode_ref_feat_sin = self.gen_sineembed_for_position(mode_ref)  # [bs, num_mode, num_poses, 256]
                mode_ref_feat = self._mode_embedding(mode_ref_feat_sin)  # [bs, num_mode, num_poses, 256]
            else:
                mode_ref_feat = self._mode_embedding(mode_ref)  # [bs, num_mode, num_poses, 256]
            ego_query = ego_query.clone() + mode_ref_feat.reshape(batch_size, -1, mode_ref_feat.shape[-1]).clone()



        
        ##============================== 轨迹特征提取 ======================================

        ego_query_out = self._tf_decoder(ego_query, keyval_final)

        print(f"ego_query_decoder time: {time.time() - start_time}")
        ##============================== 轨迹特征提取 ======================================

       

        ##============================== 控制命令预测 ======================================
        start_time = time.time()

        cmd_query = self._cmd_pred_query.weight[None, ...].repeat(batch_size, 1, 1)  # [bs, 1, 256]
        cmd_query_out = self._cmd_head_decoder(cmd_query, keyval_final)  #

        cmd_logits = self._cmd_mlp(cmd_query_out).squeeze(1)  # [bs, 4]
        # 取一个除了3外，最大概率的类别作为预测值
        # cmd_pred = torch.argmax(cmd_logits[:, :3], dim=-1)  # [bs]
        cmd_pred = torch.argmax(cmd_logits, dim=-1)  # [bs]

        print(f"cmd_head_decoder time: {time.time() - start_time}")

        start_time = time.time()
        trajectory = {}
        trajectory['first_trajectory'] = self._trajectory_head(ego_query_out, cmd=cmd)

        print(f"first trajectory_head time: {time.time() - start_time}")

        trajectory['cmd_logits'] = cmd_logits
        trajectory['cmd_pred'] = cmd_pred
        trajectory['cmd_gt'] = cmd

        # # record the geometry feature and gt
        # trajectory['geometry_predict'] = geometry_feat
        # trajectory['geometry_gt'] = geometry_feature[:, 0, ...]  # [bs, 1, n, 519, 2048] -> [bs, n, 519, 2048]
       

        # # 如果rl，则输出方差
        # if self.config.training_mode == "ft" or self.config.training_mode == 'bcsl':
        #     traj_var = self._var_head(ego_query_out)
        #     trajectory['trajectory_var'] = traj_var
        #     trajectory['keyval_final'] = keyval_final

        if self._config.use_wm:
            start_time = time.time()

            wm_keyval = keyval_final
            wm_target = keyval_final
            wm_query = self._wm_query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
            wm_next_latent = self._wm_decoder(wm_query, wm_keyval)
            trajectory['wm_next_latent']=wm_next_latent
            trajectory['cur_latent']=wm_target

            print(f"wm_decoder time: {time.time() - start_time}")
            
            # refine the trajectory
            # start_time = time.time()

            # refined_traj_feat = self.refine_traj_decoder(ego_query_out, wm_next_latent, geometry_feat_for_refine)
            # refined_traj = self._trajectory_head(refined_traj_feat, cmd=cmd)
            # trajectory['refined_trajectory'] = refined_traj

            # print(f"refined trajectory_head time: {time.time() - start_time}")
            return trajectory
        else:
            return trajectory
    
    def compute_traj_loss(self, trajectories, gt):
        loss_dict = {}
        all_trajs = trajectories.get("all_trajectories")  # [B, M, T, 3]
        B, M = all_trajs.shape[0], all_trajs.shape[1]
        
        # 原 hard winner-take-all 方式
        traj_gt_dist = torch.norm(all_trajs[..., :2] - gt[:, None, :, :2], p=1, dim=-1)  # [B,M,T]
        fde_dist = traj_gt_dist[:, :, -1]  # [B,M]
        best_mode = torch.argmin(fde_dist, dim=-1)  # [B]
        best_traj = all_trajs[torch.arange(B), best_mode]
        trajectory_loss = torch.nn.functional.l1_loss(best_traj, gt)

        # 分类 hard label
        if "cls_logits" in trajectories:
            cls_loss = torch.nn.functional.cross_entropy(trajectories["cls_logits"], best_mode)
            loss_dict["cls_loss"] = cls_loss * self.traj_cls_loss_weight

        loss_dict["traj_loss"] = trajectory_loss * self.traj_loss_weight
        return loss_dict

    # the loss function for world model
    # @timed
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        logging_prefix: str = None,
    ) -> torch.Tensor:
        loss_dict = {}
        # ========================= 多模态监督 =========================
        if self._config.num_mode:
            # first trajectory
            first_trajectory = predictions['first_trajectory']
            # refined_trajectory = predictions['refined_trajectory']
            gt = targets["trajectory"]  # [B, T, 3]

            # first trajectory loss
            first_traj_loss_dict = self.compute_traj_loss(first_trajectory, gt)
            # refined_traj_loss_dict = self.compute_traj_loss(refined_trajectory, gt)

            loss_dict.update(
                {
                    "first_traj_loss": first_traj_loss_dict["traj_loss"],
                    # "refined_traj_loss": refined_traj_loss_dict["traj_loss"],
                    # "first_cls_loss": first_traj_loss_dict["cls_loss"],
                    # "refined_cls_loss": refined_traj_loss_dict["cls_loss"],
                }
            )

            ## 控制命令预测loss
            if "cmd_logits" in predictions:
                cmd = predictions['cmd_gt']
                cmd_loss = torch.nn.functional.cross_entropy(predictions["cmd_logits"], cmd)
                # loss_dict["cmd_loss"] = cmd_loss * self._config.traj_cmd_loss_weight

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
        if self._config.use_wm_training and 'wm_next_latent' in predictions:
            wm_loss_a = torch.nn.functional.mse_loss(predictions["wm_next_latent"], predictions["next_latent"])
            loss_dict["wm_loss"] = wm_loss_a * self.wm_loss_weight

        # # geometry feature loss
        # geometry_loss = torch.nn.functional.l1_loss(predictions["geometry_predict"], predictions["geometry_gt"])
        # loss_dict["geometry_loss"] = geometry_loss * self.geometry_loss_weight

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

    # @timed
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
                best_traj_idx = cmd
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

