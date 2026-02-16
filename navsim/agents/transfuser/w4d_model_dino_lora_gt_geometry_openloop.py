import copy
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import timm
import time
import math  # 添加math以支持gen_sineembed_for_position中使用
from PIL import Image

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from navsim.common.enums import StateSE2Index
from navsim.agents.transfuser.temporal_world_model import TemporalWorldModel
from navsim.agents.transfuser.utils.bridge_attention import BridgeAttentionTransformer

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

        self.is_eval = config.is_eval

        # learnable scene query
        self.scene_embeds = nn.Parameter(torch.randn(1, config.num_frames, config.num_views, config.num_scene_query_token, config.dino_d_model)*1e-6, requires_grad=True)
        
        # define dino-LoRA encoder for vision query
        self.dino_encoder = AutoModel.from_pretrained("/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-small").to(self.scene_embeds.device)
        self.dino_processor = AutoImageProcessor.from_pretrained("/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-small")
        self.dino_processor.crop_size = {'height': 224, 'width': 448}

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
        
        # online dino encoder with LoRA
        self.lora_dino_encoder = _apply_lora_to_dino_encoder(self.dino_encoder)
        
        # target dino encoder for alignment
        self.target_dino_encoder = copy.deepcopy(self.lora_dino_encoder)
        for param in self.target_dino_encoder.parameters():
            param.requires_grad = False
        
        self.dino_projector = nn.Sequential(
            nn.Linear(config.dino_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )

        # geometry learnable query
        self.geometry_query = nn.Parameter(
            torch.randn(1, config.num_views, config.num_geometry_feature_token, config.gtf_d_model),
            requires_grad=True,
        )
        
        self.geometry_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.gtf_d_model,
                nhead=config.gtf_num_head,
                dim_feedforward=config.gtf_d_ffn,
                dropout=config.gtf_dropout,
                batch_first=True,
            ), config.gtf_num_layers
        )

        self.geometry_projector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )

        # self.image_fc = nn.Linear(512, 256)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._num_poses = config.trajectory_sampling.num_poses
        self._num_mode = config.num_mode if hasattr(config, 'num_mode') else 4

        #TODO: petr position_embedding
        num_keyval = config.num_views * config.num_scene_query_token + 1
        self.num_keyval = num_keyval

        ############################################多模态相关初始化########################################
        # 增加多模态轨迹嵌入
        # 注意：self._num_mode 已在前面初始化 
        if self._num_mode == 4:
            if self._config.use_cmd_embed:
                # 使用左中右三种命令来指引模态 + 命令嵌入
                self.waypoint_mode_ref = nn.Embedding(self._num_mode, config.tf_d_model)  # 4 commands
            else:
                # 使用左中右三种命令来指引模态
                self.waypoint_mode_ref = nn.Embedding(self._num_mode * self._num_poses, config.tf_d_model)  # [num_mode * num_poses, d_model]

        # 编码多模态引导信息为嵌入
        self._mode_embedding = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        
        # refine_ego_query: 独立的精化轨迹查询embedding，与ego_query分开优化
        if self._num_mode == 4:
            if self._config.use_cmd_embed:
                self._refine_query_embedding = nn.Embedding(self._num_mode, config.tf_d_model)  # [4, d_model]
            else:
                self._refine_query_embedding = nn.Embedding(self._num_mode * self._num_poses, config.tf_d_model)  # [num_mode * num_poses, d_model]

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

            self.refine_block = nn.Sequential(*[
                BridgeAttentionTransformer(config) for _ in range(config.tf_num_layers)
            ])

            self.refine_traj_head = TrajectoryHead(self._num_poses, config.tf_d_ffn, config.tf_d_model, config.num_mode)
        else:
            self._trajectory_head = TrajectoryHead(self._num_poses, config.tf_d_ffn, config.tf_d_model)

        # loss weight
        self.wm_loss_weight = config.wm_loss_weight
        self.traj_loss_weight = config.traj_loss_weight if hasattr(config, 'traj_loss_weight') else 1.0
        self.traj_cls_loss_weight = config.traj_cls_loss_weight if hasattr(config, 'traj_cls_loss_weight') else 0.5
               
        # world model
        self.use_wm = self._config.use_wm
        if self.use_wm:
            num_wm_query = (self._config.num_frames-1) * num_keyval
            self._wm_query_embedding = nn.Embedding(num_wm_query, config.tf_d_model)
            self.temporal_world_model = TemporalWorldModel(config, use_4d_rope=False)
    
    def get_dino_features_with_scene_query(self, model: Dinov2Model, inputs: torch.Tensor, scene_query: torch.Tensor):
        pixel_values = inputs

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = model.embeddings(pixel_values, None)
        embedding_output = torch.cat([scene_query, embedding_output], dim=1)

        encoder_outputs: BaseModelOutput = model.encoder(
            embedding_output, None, None
        )
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = model.layernorm(sequence_output)
        sequence_output = self.dino_projector(sequence_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: float):
        for target_param, online_param in zip(self.target_dino_encoder.parameters(), self.lora_dino_encoder.parameters()):
            target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)

    def forward_test(self, features) -> Dict[str, torch.Tensor]:
        def get_dino_input_image(image_feature):
            batch_size, num_views, height, width, channels = image_feature.shape
            inputs = self.dino_processor(images=image_feature.reshape(batch_size * num_views, height, width, channels), return_tensors="pt").to('cuda')
            return inputs.pixel_values.reshape(batch_size, num_views, channels, *inputs.pixel_values.shape[-2:])

        all_frames_dino_input = torch.stack(
            [
                get_dino_input_image(features['camera_feature_prev_3']['dino_feature']),
                get_dino_input_image(features['camera_feature']['dino_feature']),
                # get_dino_input_image(features['camera_feature_next_4']['dino_feature']),
                get_dino_input_image(features['camera_feature_next_8']['dino_feature']),
            ], dim=1
        )

        # only use current frame's geometry feature
        geometry_feature_gt = features['camera_feature']['geometry_feature'].squeeze(1)  # [bs, num_views, 519, 2048]

        all_frames_status_feature = torch.stack(
            [
                features['status_feature_prev_3'],
                features['status_feature'],
                # features['status_feature_next_4'],
                features['status_feature_next_8'],
            ], dim=1
        )

        cmd = all_frames_status_feature[..., :4]  # [bs, num_frames, 4] one-hot
        cmd = torch.argmax(cmd, dim=-1)  # [bs, num_frames] int

        batch_size, num_frames, num_views, _, _, _ = all_frames_dino_input.shape
        device = self.scene_embeds.device

        # ==================== Dino Encoder with scene query =================
        init_scene_query = self.scene_embeds.repeat(batch_size, 1, 1, 1, 1)
        batch_size, num_frames, num_views, num_query_tokens, dino_dim = init_scene_query.shape
        init_scene_query = init_scene_query.reshape(batch_size * num_frames * num_views, num_query_tokens, dino_dim)

        image_scene_query = self.get_dino_features_with_scene_query(
            self.lora_dino_encoder, 
            all_frames_dino_input.reshape(-1, *all_frames_dino_input.shape[-3:]), 
            init_scene_query
        ).last_hidden_state

        _, _, dim = image_scene_query.shape
        scene_query = image_scene_query.reshape(batch_size, num_frames, num_views, -1, dim)[..., :self._config.num_scene_query_token, :]
        scene_query = scene_query.reshape(batch_size, num_frames, -1, dim)
        status_encoding = self._status_encoding(all_frames_status_feature)
        keyval = torch.cat([status_encoding[:, :, None], scene_query], dim=-2)
        keyval = keyval.clone() + self._keyval_embedding.weight[None, None, ...]
        keyval_final = keyval

        if not self.is_eval:
            # target_dino_encoder 提取未来 dino feature 作为 gt
            with torch.no_grad():
                # detach init_scene_query，断开与 self.scene_embeds 的计算图连接
                init_scene_query_detached = init_scene_query.detach()
                
                gt_scene_query = self.get_dino_features_with_scene_query(
                    self.target_dino_encoder, 
                    all_frames_dino_input.reshape(-1, *all_frames_dino_input.shape[-3:]), 
                    init_scene_query_detached  # ✅ 使用 detach 后的 query
                ).last_hidden_state
                
                gt_scene_query = gt_scene_query.reshape(batch_size, num_frames, num_views, -1, dim)[..., :self._config.num_scene_query_token, :]
                gt_scene_query = gt_scene_query.reshape(batch_size, num_frames, -1, dim)
                
                # detach status_encoding，断开与主编码器的计算图连接
                status_encoding_detached = status_encoding.detach()
                
                gt_keyval = torch.cat([status_encoding_detached[:, :, None], gt_scene_query], dim=-2)
                gt_keyval = gt_keyval.clone() + self._keyval_embedding.weight[None, None, ...]
                gt_keyval_final = gt_keyval

        # =============================== keyval trans =======================================
        # start_time = time.time()

        # ego_query: 第一阶段主轨迹查询
        ego_query = self._query_embedding.weight[None, None, ...].repeat(batch_size, num_frames, 1, 1)
        # refine_ego_query: 独立的精化轨迹查询，与forward_train_curriculum保持一致
        refine_ego_query = self._refine_query_embedding.weight[None, None, ...].repeat(batch_size, num_frames, 1, 1)

        # 多模态引导，与 forward_train_curriculum 保持一致
        if self._config.num_mode:
            # 使用左中右三种命令来指引模态
            mode_ref = self.waypoint_mode_ref.weight # [num_mode, 256]
            if self._config.use_cmd_embed:
                mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, self._num_poses, 1)  # [bs, num_mode, num_poses, 256]
            else:
                mode_ref = mode_ref.reshape(self._num_mode, self._num_poses, self._config.tf_d_model)  # [num_mode, num_poses, 256]
                mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(1).repeat(batch_size, num_frames, 1, 1, 1)  # [bs, num_frames, num_mode, num_poses, 256]

            mode_ref_feat = self._mode_embedding(mode_ref)  # [bs, num_frames, num_mode, num_poses, 256]
            # 主轨迹查询 + 模态特征
            ego_query = ego_query.clone() + mode_ref_feat.reshape(batch_size, num_frames, -1, mode_ref_feat.shape[-1]).clone()

        ## ============================== 轨迹特征提取 ======================================
        ego_query_out = self._tf_decoder(
            ego_query.reshape(-1, *ego_query.shape[-2:]),
            keyval_final.reshape(-1, *keyval_final.shape[-2:])
        )

        # print(f"ego_query_decoder time: {time.time() - start_time}")
        ##============================== 轨迹特征提取 ======================================

       

        ##============================== 控制命令预测 ======================================
        # start_time = time.time()

        cmd_query = self._cmd_pred_query.weight[None, None, ...].repeat(batch_size, num_frames, 1, 1)  # [bs, num_frames, 1, 256]
        cmd_query_out = self._cmd_head_decoder(
            cmd_query.reshape(-1, *cmd_query.shape[-2:]),
            keyval_final.reshape(-1, *keyval_final.shape[-2:])
        )  # [bs, num_frames, 1, 256]

        cmd_logits = self._cmd_mlp(cmd_query_out).squeeze(1)  # [bs*num_frames, 4]
        # 取一个除了3外，最大概率的类别作为预测值
        # cmd_pred = torch.argmax(cmd_logits[:, :3], dim=-1)  # [bs]
        cmd_pred = torch.argmax(cmd_logits, dim=-1)  # [bs*num_frames]

        # print(f"cmd_head_decoder time: {time.time() - start_time}")

        # # predict geometry feature from dino_feature
        # dino_full_feature = image_scene_query.reshape(batch_size, num_frames, num_views, -1, dim)[:, 1, :, self._config.num_scene_query_token:, :]  # [bs, 3, 513, 256]
        # init_geometry_query = self.geometry_query.repeat(batch_size, 1, 1, 1)
        # geometry_feature_pred = self.geometry_decoder(init_geometry_query.reshape(batch_size, -1, dim), dino_full_feature.reshape(batch_size, -1, dim)).reshape(batch_size, num_views, -1, dim)
        geometry_feature_align = self.geometry_projector(geometry_feature_gt)  # [bs, num_views, 519, 256]

        # wm
        if self.use_wm:
            trajectory = {}
            trajectory['first_traj'] = self._trajectory_head(ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...], cmd=cmd[:, 1])

            trajectory['cmd_logits'] = cmd_logits.reshape(batch_size, num_frames, -1)[:, 1, ...].squeeze(1)
            trajectory['cmd_pred'] = cmd_pred.reshape(batch_size, num_frames)[:, 1]
            trajectory['cmd_gt'] = cmd.reshape(batch_size, num_frames)[:, 1]
            # trajectory['geometry_feature_align'] = geometry_feature_align
            # trajectory['geometry_feature_gt'] = geometry_feature_gt

            # 原自回归WM已移除，改为Teacher Forcing模式，与forward_train_curriculum完全一致
            wm_keyval = keyval_final[:, :-1, ...].reshape(batch_size, -1, dim)
            wm_query = self._wm_query_embedding.weight[None, ...].repeat(batch_size, 1, 1).reshape(batch_size, -1, dim)

            n_tokens_3d = (num_frames-1)*(self._config.num_views*self._config.num_scene_query_token+1)
            attention_mask = self.temporal_world_model.get_attention_mask(n_tokens_3d, (self._config.num_views*self._config.num_scene_query_token+1)).to(device)
            wm_next_latent = self.temporal_world_model.forward(
                query=wm_query,
                keyval=wm_keyval,
                time_frames=num_frames-1,
                height=self._config.num_views,
                width=self._config.num_scene_query_token,
                tgt_mask=attention_mask,
                memory_mask=attention_mask
            )
            trajectory['wm_next_latent'] = wm_next_latent.reshape(batch_size, num_frames-1, -1, dim)
            if not self.is_eval:
                trajectory['gt_next_latent'] = gt_keyval_final[:, 1:, ...]
            
            # refine: 与forward_train_curriculum保持一致
            # 提取主轨迹特征用于残差连接
            ego_query_out_for_residual = ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...]
            
            # refine_query 独立初始化，通过refine_block
            refine_query = refine_ego_query.reshape(batch_size, num_frames, -1, dim)[:, 1, ...]
            wm_next_latent = trajectory['wm_next_latent'].reshape(batch_size, num_frames-1, -1, dim)[:, -1, ...]
            for layer in self.refine_block:
                refine_query = layer(refine_query, geometry_feature_align.reshape(batch_size, -1, dim), wm_next_latent)
            
            # 残差连接：在refine基础上加上一阶段的主轨迹特征
            refined_traj_features = ego_query_out_for_residual + refine_query
            trajectory['refined_traj'] = self.refine_traj_head(refined_traj_features, cmd=cmd[:, 1])

        return trajectory

    # @timed
    def forward_train(self, features) -> Dict[str, torch.Tensor]:
        # unpack the camera_feature to get dino feature and geometry feature
        # dino_feature, geometry_feature = features['camera_feature']
        def get_dino_input_image(image_feature):
            batch_size, num_views, height, width, channels = image_feature.shape
            inputs = self.dino_processor(images=image_feature.reshape(batch_size * num_views, height, width, channels), return_tensors="pt").to('cuda')
            return inputs.pixel_values.reshape(batch_size, num_views, channels, *inputs.pixel_values.shape[-2:])

        all_frames_dino_input = torch.stack(
            [
                get_dino_input_image(features['camera_feature_prev_3']['dino_feature']),
                # get_dino_input_image(features['camera_feature_prev_2']['dino_feature']),
                # get_dino_input_image(features['camera_feature_prev_1']['dino_feature']),
                get_dino_input_image(features['camera_feature']['dino_feature']),
                # get_dino_input_image(features['camera_feature_next_1']['dino_feature']),
                # get_dino_input_image(features['camera_feature_next_2']['dino_feature']),
                # get_dino_input_image(features['camera_feature_next_3']['dino_feature']),
                get_dino_input_image(features['camera_feature_next_4']['dino_feature']),
                # get_dino_input_image(features['camera_feature_next_5']['dino_feature']),
                # get_dino_input_image(features['camera_feature_next_6']['dino_feature']),
                # get_dino_input_image(features['camera_feature_next_7']['dino_feature']),
                get_dino_input_image(features['camera_feature_next_8']['dino_feature']),
            ], dim=1
        )

        all_frames_status_feature = torch.stack(
            [
                features['status_feature_prev_3'],
                # features['status_feature_prev_2'],
                # features['status_feature_prev_1'],
                features['status_feature'],
                # features['status_feature_next_1'],
                # features['status_feature_next_2'],
                # features['status_feature_next_3'],
                features['status_feature_next_4'],
                # features['status_feature_next_5'],
                # features['status_feature_next_6'],–
                # features['status_feature_next_7'],
                features['status_feature_next_8'],
            ], dim=1
        )

        cmd = all_frames_status_feature[..., :4]  # [bs, num_frames, 4] one-hot
        cmd = torch.argmax(cmd, dim=-1)  # [bs, num_frames] int


        batch_size, num_frames, num_views, _, _, _ = all_frames_dino_input.shape
        device= self.scene_embeds.device

        # ==================== Dino Encoder with scene query =================
        # v2: use learnable query do cross attention with each view dino feature

        # start_time = time.time()
        # lora_dino_encoder 提取历史 dino feature 作为 keyval
        init_scene_query = self.scene_embeds.repeat(batch_size, 1, 1, 1, 1)  # [1, num_frames, num_views, 16, 384] -> [bs, num_frames, num_views, 16, 384]
        batch_size, num_frames, num_views, num_query_tokens, dino_dim = init_scene_query.shape  # [bs, 4, 3, 16, 384]
        init_scene_query = init_scene_query.reshape(batch_size * num_frames, num_views * num_query_tokens, dino_dim)  # [bs*4, 3*16, 384]

        image_scene_query = self.get_dino_features_with_scene_query(
            self.lora_dino_encoder, 
            all_frames_dino_input.reshape(-1, *all_frames_dino_input.shape[-3:]), 
            init_scene_query
        ).last_hidden_state  # [bs*4, 48+512*3, 256]

        _, _, dim = image_scene_query.shape
        image_scene_query = image_scene_query.reshape(batch_size, num_frames, num_views, -1, dim)[..., :self._config.num_scene_query_token, :]  # [bs, 4, 3, 16, 256]
            
        scene_query = image_scene_query.reshape(batch_size, num_frames, -1, dim)  # [bs, 4, 3*16, 256]
        status_encoding = self._status_encoding(all_frames_status_feature)  # [bs, num_frame, 8] -> [bs, num_frame, 256]
        keyval = torch.cat([status_encoding[:, :, None], scene_query], dim=-2)  # [bs, 4, 1+48, 256]
        keyval = keyval.clone() + self._keyval_embedding.weight[None, None, ...]
        keyval_final = keyval  # [bs, 4, 48+1, 256]

        # print(f"dino_encoder time: {time.time() - start_time}")

        # target_dino_encoder 提取未来 dino feature 作为 gt
        # start_time = time.time()
        with torch.no_grad():
            # detach init_scene_query，断开与 self.scene_embeds 的计算图连接
            init_scene_query_detached = init_scene_query.detach()
            
            gt_scene_query = self.get_dino_features_with_scene_query(
                self.target_dino_encoder, 
                all_frames_dino_input.reshape(-1, *all_frames_dino_input.shape[-3:]), 
                init_scene_query_detached  # ✅ 使用 detach 后的 query
            ).last_hidden_state
            
            gt_scene_query = gt_scene_query.reshape(batch_size, num_frames, num_views, -1, dim)[..., :self._config.num_scene_query_token, :]
            gt_scene_query = gt_scene_query.reshape(batch_size, num_frames, -1, dim)
            
            # detach status_encoding，断开与主编码器的计算图连接
            status_encoding_detached = status_encoding.detach()
            
            gt_keyval = torch.cat([status_encoding_detached[:, :, None], gt_scene_query], dim=-2)
            gt_keyval = gt_keyval.clone() + self._keyval_embedding.weight[None, None, ...]
            gt_keyval_final = gt_keyval

        # print(f"target_dino_encoder time: {time.time() - start_time}")

        # # geometry query to interact with dino feature of each view
        # start_time = time.time()

        # init_geometry_query = self.geometry_query.repeat(batch_size, 1, 1, 1)  # [1, 3, 519, 1024] -> [bs, 3, 519, 1024]
        # geometry_feat = torch.zeros_like(init_geometry_query)
        # for i in range(self._config.num_views):
        #     geometry_query = init_geometry_query[:, i]  # [bs, 519, 1024]
        #     camera_kv = dino_feature[:, i]  # [bs, 519, 1024]
        #     geometry_feat[:, i] = self.geometry_decoder[i](geometry_query, camera_kv)
        
        # geometry_feat_for_refine = geometry_feat[:, :, 7:, :].reshape(b, -1, dim)  # [b, 3*512, 1024]
        # geometry_feat = self.geometry_projector(geometry_feat)  # [b, 3, 519, 1024] -> [b, 3, 519, 2048]

        # print(f"geometry_decoder time: {time.time() - start_time}")

        # =============================== keyval trans =======================================
        # start_time = time.time()

        ego_query = self._query_embedding.weight[None, None, ...].repeat(batch_size, num_frames, 1, 1)   # [bs, num_frames, num_mode * num_poses, 256]

        # 多模态引导，与 forward_train 保持一致
        if self._config.num_mode:
            # 使用左中右三种命令来指引模态
            mode_ref = self.waypoint_mode_ref.weight # [num_mode, 256]
            if self._config.use_cmd_embed:
                mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, self._num_poses, 1)  # [bs, num_mode, num_poses, 256]
            else:
                mode_ref = mode_ref.reshape(self._num_mode, self._num_poses, self._config.tf_d_model)  # [num_mode, num_poses, 256]
                mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(1).repeat(batch_size, num_frames, 1, 1, 1)  # [bs, num_frames, num_mode, num_poses, 256]

            mode_ref_feat = self._mode_embedding(mode_ref)  # [bs, num_frames, num_mode, num_poses, 256]
            ego_query = ego_query.clone() + mode_ref_feat.reshape(batch_size, num_frames, -1, mode_ref_feat.shape[-1]).clone()

        ## ============================== 轨迹特征提取 ======================================
        ego_query_out = self._tf_decoder(
            ego_query.reshape(-1, *ego_query.shape[-2:]),
            keyval_final.reshape(-1, *keyval_final.shape[-2:])
        )

        # print(f"ego_query_decoder time: {time.time() - start_time}")
        ##============================== 轨迹特征提取 ======================================

       

        ##============================== 控制命令预测 ======================================
        # start_time = time.time()

        cmd_query = self._cmd_pred_query.weight[None, None, ...].repeat(batch_size, num_frames, 1, 1)  # [bs, num_frames, 1, 256]
        cmd_query_out = self._cmd_head_decoder(
            cmd_query.reshape(-1, *cmd_query.shape[-2:]),
            keyval_final.reshape(-1, *keyval_final.shape[-2:])
        )  # [bs, num_frames, 1, 256]

        cmd_logits = self._cmd_mlp(cmd_query_out).squeeze(1)  # [bs*num_frames, 4]
        # 取一个除了3外，最大概率的类别作为预测值
        # cmd_pred = torch.argmax(cmd_logits[:, :3], dim=-1)  # [bs]
        cmd_pred = torch.argmax(cmd_logits, dim=-1)  # [bs*num_frames]

        # print(f"cmd_head_decoder time: {time.time() - start_time}")

        # # record the geometry feature and gt
        # trajectory['geometry_predict'] = geometry_feat
        # trajectory['geometry_gt'] = geometry_feature[:, 0, ...]  # [bs, 1, n, 519, 2048] -> [bs, n, 519, 2048]

        # wm
        if self.use_wm:
            # start_time = time.time()
            trajectory = {}
            trajectory['first_traj'] = self._trajectory_head(ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...], cmd=cmd[:, 1])

            trajectory['cmd_logits'] = cmd_logits.reshape(batch_size, num_frames, -1)[:, 1, ...].squeeze(1)
            trajectory['cmd_pred'] = cmd_pred.reshape(batch_size, num_frames)[:, 1]
            trajectory['cmd_gt'] = cmd.reshape(batch_size, num_frames)[:, 1]

            wm_keyval = keyval_final[:, :-1, ...].reshape(batch_size, -1, dim)
            wm_query = self._wm_query_embedding.weight[None, ...].repeat(batch_size, 1, 1).reshape(batch_size, -1, dim)

            n_tokens_3d = (num_frames-1)*(self._config.num_views*self._config.num_scene_query_token+1)
            attention_mask = self.temporal_world_model.get_attention_mask(n_tokens_3d, (self._config.num_views*self._config.num_scene_query_token+1)).to(device)
            wm_next_latent = self.temporal_world_model.forward(
                                                            query=wm_query,
                                                            keyval=wm_keyval,
                                                            time_frames=num_frames-1,
                                                            height=self._config.num_views,
                                                            width=self._config.num_scene_query_token,
                                                            tgt_mask=attention_mask,
                                                            memory_mask=attention_mask
                                                        )
            trajectory['wm_next_latent'] = wm_next_latent.reshape(batch_size, num_frames-1, -1, dim)
            trajectory['gt_next_latent'] = gt_keyval_final[:, 1:, ...]

            # print(f"wm_forward time: {time.time() - start_time}")
            
            # refine
            refine_query = ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...]
            refine_keyval = wm_next_latent.reshape(batch_size, num_frames-1, -1, dim)[:, -1, ...]
            refine_query_out = ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...] + self.refine_block(refine_query, refine_keyval)
            
            # TODO: comupute the refined trajectory with residual
            trajectory['refined_traj'] = self.refine_traj_head(refine_query_out, cmd=cmd[:, 1])
            # for key in trajectory['first_traj']:
            #     trajectory['refined_traj'][key] += trajectory['first_traj'][key]
        return trajectory

    def forward_train_curriculum(self, features, ar_ratio: float = 0.0, global_step: int = 0):
        """
        课程学习训练方法，支持 teacher forcing 和自回归模式的混合
        
        Args:
            features: 输入特征
            ar_ratio: 自回归比例 (0.0 = 纯 teacher forcing, 1.0 = 纯自回归)
            global_step: 全局训练步数，用于 DDP 多卡训练时的确定性决策
        """
        # no need to use autoregressive
        use_autoregressive = False
        
        def get_dino_input_image(image_feature):
            batch_size, num_views, height, width, channels = image_feature.shape
            inputs = self.dino_processor(images=image_feature.reshape(batch_size * num_views, height, width, channels), return_tensors="pt").to('cuda')
            return inputs.pixel_values.reshape(batch_size, num_views, channels, *inputs.pixel_values.shape[-2:])

        all_frames_dino_input = torch.stack(
            [
                get_dino_input_image(features['camera_feature_prev_3']['dino_feature']),
                get_dino_input_image(features['camera_feature']['dino_feature']),
                # get_dino_input_image(features['camera_feature_next_4']['dino_feature']),
                get_dino_input_image(features['camera_feature_next_8']['dino_feature']),
            ], dim=1
        )

        # only use current frame's geometry feature
        geometry_feature_gt = features['camera_feature']['geometry_feature'].squeeze(1)  # [bs, num_views, 519, 2048]

        all_frames_status_feature = torch.stack(
            [
                features['status_feature_prev_3'],
                features['status_feature'],
                # features['status_feature_next_4'],
                features['status_feature_next_8'],
            ], dim=1
        )

        cmd = all_frames_status_feature[..., :4]  # [bs, num_frames, 4] one-hot
        cmd = torch.argmax(cmd, dim=-1)  # [bs, num_frames] int

        batch_size, num_frames, num_views, _, _, _ = all_frames_dino_input.shape
        device = self.scene_embeds.device

        # ==================== Dino Encoder with scene query =================
        init_scene_query = self.scene_embeds.repeat(batch_size, 1, 1, 1, 1)
        batch_size, num_frames, num_views, num_query_tokens, dino_dim = init_scene_query.shape
        init_scene_query = init_scene_query.reshape(batch_size * num_frames * num_views, num_query_tokens, dino_dim)

        image_scene_query = self.get_dino_features_with_scene_query(
            self.lora_dino_encoder, 
            all_frames_dino_input.reshape(-1, *all_frames_dino_input.shape[-3:]), 
            init_scene_query
        ).last_hidden_state

        _, _, dim = image_scene_query.shape
        scene_query = image_scene_query.reshape(batch_size, num_frames, num_views, -1, dim)[..., :self._config.num_scene_query_token, :]
        scene_query = scene_query.reshape(batch_size, num_frames, -1, dim)
        status_encoding = self._status_encoding(all_frames_status_feature)
        keyval = torch.cat([status_encoding[:, :, None], scene_query], dim=-2)
        keyval = keyval.clone() + self._keyval_embedding.weight[None, None, ...]
        keyval_final = keyval

        # target_dino_encoder 提取未来 dino feature 作为 gt
        with torch.no_grad():
            init_scene_query_detached = init_scene_query.detach()
            gt_scene_query = self.get_dino_features_with_scene_query(
                self.target_dino_encoder, 
                all_frames_dino_input.reshape(-1, *all_frames_dino_input.shape[-3:]), 
                init_scene_query_detached
            ).last_hidden_state
            gt_scene_query = gt_scene_query.reshape(batch_size, num_frames, num_views, -1, dim)[..., :self._config.num_scene_query_token, :]
            gt_scene_query = gt_scene_query.reshape(batch_size, num_frames, -1, dim)
            status_encoding_detached = status_encoding.detach()
            gt_keyval = torch.cat([status_encoding_detached[:, :, None], gt_scene_query], dim=-2)
            gt_keyval = gt_keyval.clone() + self._keyval_embedding.weight[None, None, ...]
            gt_keyval_final = gt_keyval

        # =============================== ego query trans =======================================
        # ego_query: 第一阶段主轨迹查询
        ego_query = self._query_embedding.weight[None, None, ...].repeat(batch_size, num_frames, 1, 1)
        # refine_ego_query: 独立的精化轨迹查询，NOT derived from ego_query
        refine_ego_query = self._refine_query_embedding.weight[None, None, ...].repeat(batch_size, num_frames, 1, 1)

        if self._config.num_mode:
            mode_ref = self.waypoint_mode_ref.weight
            if self._config.use_cmd_embed:
                mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, self._num_poses, 1)
            else:
                mode_ref = mode_ref.reshape(self._num_mode, self._num_poses, self._config.tf_d_model)
                mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(1).repeat(batch_size, num_frames, 1, 1, 1)

            mode_ref_feat = self._mode_embedding(mode_ref)
            # 主轨迹查询 + 模态特征
            ego_query = ego_query.clone() + mode_ref_feat.reshape(batch_size, num_frames, -1, mode_ref_feat.shape[-1]).clone()

        ego_query_out = self._tf_decoder(
            ego_query.reshape(-1, *ego_query.shape[-2:]),
            keyval_final.reshape(-1, *keyval_final.shape[-2:])
        )

        cmd_query = self._cmd_pred_query.weight[None, None, ...].repeat(batch_size, num_frames, 1, 1)
        cmd_query_out = self._cmd_head_decoder(
            cmd_query.reshape(-1, *cmd_query.shape[-2:]),
            keyval_final.reshape(-1, *keyval_final.shape[-2:])
        )
        cmd_logits = self._cmd_mlp(cmd_query_out).squeeze(1)
        cmd_pred = torch.argmax(cmd_logits, dim=-1)

        # # predict geometry feature from dino_feature
        # dino_full_feature = image_scene_query.reshape(batch_size, num_frames, num_views, -1, dim)[:, 1, :, self._config.num_scene_query_token:, :]  # [bs, 3, 513, 256]
        # init_geometry_query = self.geometry_query.repeat(batch_size, 1, 1, 1)
        # geometry_feature_pred = self.geometry_decoder(init_geometry_query.reshape(batch_size, -1, dim), dino_full_feature.reshape(batch_size, -1, dim)).reshape(batch_size, num_views, -1, dim)

        # use gt geometry feature to refine the first trajectory
        geometry_feature_align = self.geometry_projector(geometry_feature_gt)  # [bs, num_views, 519, 256]

        # wm - 基于 ar_ratio 选择不同的训练模式
        if self.use_wm:
            trajectory = {}
            trajectory['first_traj'] = self._trajectory_head(ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...], cmd=cmd[:, 1])
            trajectory['cmd_logits'] = cmd_logits.reshape(batch_size, num_frames, -1)[:, 1, ...].squeeze(1)
            trajectory['cmd_pred'] = cmd_pred.reshape(batch_size, num_frames)[:, 1]
            trajectory['cmd_gt'] = cmd.reshape(batch_size, num_frames)[:, 1]
            # trajectory['geometry_feature_align'] = geometry_feature_align
            # trajectory['geometry_feature_gt'] = geometry_feature_gt

            if not use_autoregressive:
                # ========== Teacher Forcing 模式 (forward_train 原始逻辑) ==========
                wm_keyval = keyval_final[:, :-1, ...].reshape(batch_size, -1, dim)
                wm_query = self._wm_query_embedding.weight[None, ...].repeat(batch_size, 1, 1).reshape(batch_size, -1, dim)

                n_tokens_3d = (num_frames-1)*(self._config.num_views*self._config.num_scene_query_token+1)
                attention_mask = self.temporal_world_model.get_attention_mask(n_tokens_3d, (self._config.num_views*self._config.num_scene_query_token+1)).to(device)
                wm_next_latent = self.temporal_world_model.forward(
                    query=wm_query,
                    keyval=wm_keyval,
                    time_frames=num_frames-1,
                    height=self._config.num_views,
                    width=self._config.num_scene_query_token,
                    tgt_mask=attention_mask,
                    memory_mask=attention_mask
                )
                trajectory['wm_next_latent'] = wm_next_latent.reshape(batch_size, num_frames-1, -1, dim)
            else:
                # ========== 自回归模式 (forward_test 逻辑) ==========
                wm_keyval = keyval_final[:, :2, ...].reshape(batch_size, -1, dim)
                
                # auto-regressive from frame-2 to frame-(num_frames-1)
                for current_frame in range(2, num_frames):
                    n_tokens_3d = current_frame*(self._config.num_views*self._config.num_scene_query_token+1)
                    attention_mask = self.temporal_world_model.get_attention_mask(n_tokens_3d, (self._config.num_views*self._config.num_scene_query_token+1)).to(device)
                    current_keyval = wm_keyval.reshape(batch_size, -1, dim)
                    current_query = self._wm_query_embedding.weight[None, :current_frame*(self._config.num_views*self._config.num_scene_query_token+1), :].repeat(batch_size, 1, 1).reshape(batch_size, -1, dim)
                
                    wm_next_latent = self.temporal_world_model.forward(
                        query=current_query,
                        keyval=current_keyval,
                        time_frames=current_frame,
                        height=self._config.num_views,
                        width=self._config.num_scene_query_token,
                        tgt_mask=attention_mask,
                        memory_mask=attention_mask
                    )
                    wm_next_latent = wm_next_latent.reshape(batch_size, current_frame, -1, dim)
                    wm_keyval = wm_keyval.reshape(batch_size, current_frame, -1, dim)
                    wm_keyval = torch.cat([wm_keyval, wm_next_latent[:, -1, ...].unsqueeze(1)], dim=1)

                trajectory['wm_next_latent'] = wm_next_latent
            
            # GT latent for loss calculation
            trajectory['gt_next_latent'] = gt_keyval_final[:, 1:, ...]
            
            # refine: 使用独立的refine_ego_query，但通过残差连接利用一阶段信息
            # 设计思路：独立初始化让refine学习不同特征，残差连接保留主轨迹信息
            # refine_ego_query_out = self._tf_decoder(
            #     refine_ego_query.reshape(-1, *refine_ego_query.shape[-2:]),
            #     keyval_final.reshape(-1, *keyval_final.shape[-2:])
            # )
            
            # 提取主轨迹特征用于残差连接
            ego_query_out_for_residual = ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...]
            
            # refine_query 独立初始化，但最终通过残差与主轨迹连接
            refine_query = refine_ego_query.reshape(batch_size, num_frames, -1, dim)[:, 1, ...]
            wm_next_latent = trajectory['wm_next_latent'].reshape(batch_size, num_frames-1, -1, dim)[:, -1, ...]
            for layer in self.refine_block:
                refine_query = layer(refine_query, geometry_feature_align.reshape(batch_size, -1, dim), wm_next_latent)
            
            # 残差连接：在refine基础上加上一阶段的主轨迹特征
            # 这样既保留了独立优化的能力，又能利用一阶段的信息
            refined_traj_features = ego_query_out_for_residual + refine_query
            trajectory['refined_traj'] = self.refine_traj_head(refined_traj_features, cmd=cmd[:, 1])

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
            # loss_dict["cls_loss"] = cls_loss * self.traj_cls_loss_weight

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
            first_trajectory = predictions['first_traj']
            refined_trajectory = predictions['refined_traj']
            gt = targets["trajectory"]  # [B, T, 3]

            # first trajectory loss
            first_traj_loss_dict = self.compute_traj_loss(first_trajectory, gt)
            refined_traj_loss_dict = self.compute_traj_loss(refined_trajectory, gt)

            loss_dict.update(
                {
                    "first_traj_loss": first_traj_loss_dict["traj_loss"],
                    "refined_traj_loss": refined_traj_loss_dict["traj_loss"],
                    # "first_cls_loss": first_traj_loss_dict["cls_loss"],
                    # "refined_cls_loss": refined_traj_loss_dict["cls_loss"],
                }
            )

            ## 控制命令预测loss
            if "cmd_logits" in predictions:
                cmd = predictions['cmd_gt']
                cmd_loss = torch.nn.functional.cross_entropy(predictions["cmd_logits"], cmd)
                # loss_dict["cmd_loss"] = cmd_loss * self._config.traj_cmd_loss_weight
        else:
            # 单模态或非训练阶段
            if "trajectory" in predictions:
                trajectory_loss = torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])
                loss_dict["traj_loss"] = trajectory_loss * self.traj_loss_weight

        # 世界模型损失
        if 'wm_next_latent' in predictions:
            wm_loss_a = torch.nn.functional.mse_loss(predictions["wm_next_latent"], predictions["gt_next_latent"])
            loss_dict["wm_loss"] = wm_loss_a * self.wm_loss_weight

        # geometry feature loss
        # geometry_loss = torch.nn.functional.mse_loss(predictions["geometry_feature_align"], predictions["geometry_feature_gt"])
        # loss_dict["geometry_loss"] = geometry_loss * self._config.geometry_loss_weight

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

