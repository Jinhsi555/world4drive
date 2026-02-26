import copy
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from transformers.models.dinov2 import Dinov2Model
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from peft import LoraConfig, get_peft_model

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.common.enums import StateSE2Index

from .temporal_world_model import TemporalWorldModel
from .utils.bridge_attention import BridgeAttentionTransformer

class ego_status_decoder(nn.Module):
    def __init__(self, config: TransfuserConfig) -> None:
        super().__init__()
        # supervision by ego status
        self.cmd_decoder = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, 4),
        )

        self.velocity_decoder = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, 2),
        )

        self.acceleration_decoder = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, 2),
        )
    
    def forward(self, x):
        cmd = self.cmd_decoder(x)
        velocity = self.velocity_decoder(x)
        acceleration = self.acceleration_decoder(x)
        return cmd, velocity, acceleration

class W4DModel(nn.Module):
    """W4D 模型，结合 DINOv2 LoRA 编码器和几何查询的 Transformer 轨迹预测模型。

    Args:
        config: TransFuser 配置对象。
    """

    def __init__(
        self,
        config: TransfuserConfig,
    ) -> None:
        super().__init__()

        self._config = config

        self.is_eval = config.is_eval

        # learnable scene query
        self.scene_embeds = nn.Parameter(
            torch.randn(
                1,
                config.num_frames,
                config.num_views,
                config.num_scene_query_token,
                config.dino_d_model,
            )
            * 1e-6,
            requires_grad=True,
        )

        # define dino-LoRA encoder for vision query
        dino_ckpt_path = "/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-base"
        self.dino_encoder = AutoModel.from_pretrained(dino_ckpt_path).to(self.scene_embeds.device)
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_ckpt_path)
        self.dino_processor.crop_size = {"height": 224, "width": 448}

        def _apply_lora_to_dino_encoder(encoder):
            lora_config = LoraConfig(
                r=self._config.lora_rank,
                lora_alpha=self._config.lora_alpha,
                target_modules=self._config.lora_target_modules,
                lora_dropout=self._config.lora_dropout,
                use_rslora=self._config.use_rslora,
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
        # self.lora_dino_encoder = _apply_lora_to_dino_encoder(self.dino_encoder)
        self.lora_dino_encoder = self.dino_encoder
        
        # target dino encoder for alignment
        self.target_dino_encoder = copy.deepcopy(self.lora_dino_encoder)
        for param in self.target_dino_encoder.parameters():
            param.requires_grad = False
        
        self.dino_projector = nn.Sequential(
            nn.Linear(config.dino_d_model, config.tf_d_ffn),
            nn.LayerNorm(config.tf_d_ffn),       # 中间层归一化，稳定激活
            nn.GELU(),
            nn.Dropout(config.tf_dropout),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
            nn.LayerNorm(config.tf_d_model),     # 输出归一化，确保特征稳定
        )

        # geometry learnable query
        self.geometry_query = nn.Parameter(
            torch.randn(1, config.num_frames, config.num_views, config.num_geometry_feature_token, config.gtf_d_model),
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
            nn.Linear(config.gtf_d_model, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
        )

        # self.image_fc = nn.Linear(512, 256)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._num_poses = config.trajectory_sampling.num_poses

        num_keyval = config.num_views * config.num_scene_query_token + 1
        self.num_keyval = num_keyval
        
        ############################################多模态相关初始化########################################
        # 增加多模态轨迹嵌入
        self._num_mode = config.num_mode 
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

        ############################################多模态相关初始化########################################


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
        self.geometry_loss_weight = config.geometry_loss_weight
        self.ego_status_loss_weight = config.ego_status_loss_weight
        self.traj_loss_weight = config.traj_loss_weight if hasattr(config, 'traj_loss_weight') else 1.0
        self.traj_cls_loss_weight = config.traj_cls_loss_weight if hasattr(config, 'traj_cls_loss_weight') else 0.5
               
        # world model
        self.use_wm = self._config.use_wm
        if self.use_wm:
            num_wm_query = (self._config.num_frames-1) * num_keyval
            self._wm_query_embedding = nn.Embedding(num_wm_query, config.tf_d_model)
            self.temporal_world_model = TemporalWorldModel(config, use_4d_rope=False)
            self.ego_status_decoder = ego_status_decoder(config)
    
    def get_dino_features_with_scene_query(
        self,
        model: Dinov2Model,
        inputs: torch.Tensor,
        scene_query: torch.Tensor,
    ) -> BaseModelOutputWithPooling:
        """使用可学习场景查询获取 DINO 特征。

        Args:
            model: DINOv2 模型。
            inputs: 输入图像张量。
            scene_query: 可学习的场景查询嵌入。

        Returns:
            包含池化特征的 BaseModelOutputWithPooling 对象。
        """
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
    def update_target_encoder(self, momentum: float) -> None:
        """更新目标编码器参数（动量更新）。

        Args:
            momentum: 动量系数。
        """
        for target_param, online_param in zip(
            self.target_dino_encoder.parameters(), self.lora_dino_encoder.parameters()
        ):
            target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)

    def forward_test(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
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
                # get_dino_input_image(features['camera_feature_next_2']['dino_feature']),
                get_dino_input_image(features['camera_feature_next_4']['dino_feature']),
                # get_dino_input_image(features['camera_feature_next_6']['dino_feature']),
                get_dino_input_image(features['camera_feature_next_8']['dino_feature']),
            ], dim=1
        )
    
        all_frames_geometry_feature_gt = torch.cat(
            [
                features['camera_feature_prev_3']['geometry_feature'],
                # features['camera_feature_prev_2']['geometry_feature'],
                # features['camera_feature_prev_1']['geometry_feature'],
                features['camera_feature']['geometry_feature'],
                # features['camera_feature_next_2']['geometry_feature'],
                features['camera_feature_next_4']['geometry_feature'],
                # features['camera_feature_next_6']['geometry_feature'],
                features['camera_feature_next_8']['geometry_feature'],
            ], dim=1
        )

        all_frames_status_feature = torch.stack(
            [
                features['status_feature_prev_3'],
                # features['status_feature_prev_2'],
                # features['status_feature_prev_1'],
                features['status_feature'],
                # features['status_feature_next_2'],
                features['status_feature_next_4'],
                # features['status_feature_next_6'],
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

        # predict geometry feature from dino_feature
        dino_full_feature = image_scene_query.reshape(batch_size, num_frames, num_views, -1, dim)[..., self._config.num_scene_query_token+1:, :]  # [bs, num_frames, num_views, 512, 256] (513 token 减去 cls token)
        # init_geometry_query = self.geometry_query.repeat(batch_size, 1, 1, 1, 1)
        # geometry_feature_pred = self.geometry_decoder(dino_full_feature.reshape(batch_size, -1, dim), dino_full_feature.reshape(batch_size, -1, dim)).reshape(batch_size, num_frames, num_views, -1, dim)
        geometry_feature_align = self.geometry_projector(dino_full_feature)  # [bs, num_frames, num_views, 512, 2048]

        trajectory = {}
        trajectory['first_traj'] = self._trajectory_head(ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...], cmd=cmd[:, 1])
        trajectory['geometry_feature_align'] = geometry_feature_align
        trajectory['geometry_feature_gt'] = all_frames_geometry_feature_gt[..., 7:, :]
            
        # wm
        if self.use_wm:
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

        return trajectory

    # @timed
    def forward_train(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
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
                # get_dino_input_image(features['camera_feature_next_2']['dino_feature']),
                get_dino_input_image(features['camera_feature_next_4']['dino_feature']),
                # get_dino_input_image(features['camera_feature_next_6']['dino_feature']),
                get_dino_input_image(features['camera_feature_next_8']['dino_feature']),
            ], dim=1
        )

        # only use current frame's geometry feature
        all_frames_geometry_feature_gt = torch.cat(
            [
                features['camera_feature_prev_3']['geometry_feature'],
                # features['camera_feature_prev_2']['geometry_feature'],
                # features['camera_feature_prev_1']['geometry_feature'],
                features['camera_feature']['geometry_feature'],
                # features['camera_feature_next_2']['geometry_feature'],
                features['camera_feature_next_4']['geometry_feature'],
                # features['camera_feature_next_6']['geometry_feature'],
                features['camera_feature_next_8']['geometry_feature'],
            ], dim=1
        )

        # geometry_feature_gt = features['camera_feature']['geometry_feature'].squeeze(1)  # [bs, num_views, 519, 2048]

        all_frames_status_feature = torch.stack(
            [
                features['status_feature_prev_3'],
                # features['status_feature_prev_2'],
                # features['status_feature_prev_1'],
                features['status_feature'],
                # features['status_feature_next_2'],
                features['status_feature_next_4'],
                # features['status_feature_next_6'],
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
        ego_query = self._query_embedding.weight[None, None, ...].repeat(batch_size, num_frames, 1, 1)

        if self._config.num_mode:
            mode_ref = self.waypoint_mode_ref.weight
            if self._config.use_cmd_embed:
                mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, self._num_poses, 1)
            else:
                mode_ref = mode_ref.reshape(self._num_mode, self._num_poses, self._config.tf_d_model)
                mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(1).repeat(batch_size, num_frames, 1, 1, 1)

            mode_ref_feat = self._mode_embedding(mode_ref)
            ego_query = ego_query.clone() + mode_ref_feat.reshape(batch_size, num_frames, -1, mode_ref_feat.shape[-1]).clone()

        ego_query_out = self._tf_decoder(
            ego_query.reshape(-1, *ego_query.shape[-2:]),
            keyval_final.reshape(-1, *keyval_final.shape[-2:])
        )


        # predict geometry feature from dino_feature
        dino_full_feature = image_scene_query.reshape(batch_size, num_frames, num_views, -1, dim)[..., self._config.num_scene_query_token+1:, :]  # [bs, num_frames, num_views, 512, 256] (513 token 减去 cls token)
        # init_geometry_query = self.geometry_query.repeat(batch_size, 1, 1, 1, 1)
        # geometry_feature_pred = self.geometry_decoder(dino_full_feature.reshape(batch_size, -1, dim), dino_full_feature.reshape(batch_size, -1, dim)).reshape(batch_size, num_frames, num_views, -1, dim)
        geometry_feature_align = self.geometry_projector(dino_full_feature)  # [bs, num_frames, num_views, 512, 2048]

        trajectory = {}
        trajectory['first_traj'] = self._trajectory_head(ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...], cmd=cmd[:, 1])

        trajectory['geometry_feature_align'] = geometry_feature_align
        trajectory['geometry_feature_gt'] = all_frames_geometry_feature_gt[..., 7:, :]

        # wm - 基于 ar_ratio 选择不同的训练模式
        if self.use_wm:
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

        return trajectory

    def forward_train_curriculum(
        self,
        features: Dict[str, torch.Tensor],
        ar_ratio: float = 0.0,
        global_step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        课程学习训练方法，支持 teacher forcing 和自回归模式的混合
        
        Args:
            features: 输入特征
            ar_ratio: 自回归比例 (0.0 = 纯 teacher forcing, 1.0 = 纯自回归)
            global_step: 全局训练步数，用于 DDP 多卡训练时的确定性决策
        """
        # 基于 ar_ratio 和 global_step 做确定性决策
        # global_step 在所有 DDP rank 上是同步的，因此确保所有 GPU 做出相同决策
        if ar_ratio <= 0.0:
            use_autoregressive = False
        elif ar_ratio >= 1.0:
            use_autoregressive = True
        else:
            # 使用 global_step 的哈希值生成确定性伪随机数
            # 这样所有 GPU 在同一个 step 会做出相同的决策
            hash_val = (global_step * 2654435761) % 10000 / 10000.0  # 使用乘法哈希得到 [0, 1)
            use_autoregressive = hash_val < ar_ratio
        
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
                # get_dino_input_image(features['camera_feature_next_2']['dino_feature']),
                get_dino_input_image(features['camera_feature_next_4']['dino_feature']),
                # get_dino_input_image(features['camera_feature_next_6']['dino_feature']),
                get_dino_input_image(features['camera_feature_next_8']['dino_feature']),
            ], dim=1
        )

        # only use current frame's geometry feature
        all_frames_geometry_feature_gt = torch.cat(
            [
                features['camera_feature_prev_3']['geometry_feature'],
                # features['camera_feature_prev_2']['geometry_feature'],
                # features['camera_feature_prev_1']['geometry_feature'],
                features['camera_feature']['geometry_feature'],
                # features['camera_feature_next_2']['geometry_feature'],
                features['camera_feature_next_4']['geometry_feature'],
                # features['camera_feature_next_6']['geometry_feature'],
                features['camera_feature_next_8']['geometry_feature'],
            ], dim=1
        )

        # geometry_feature_gt = features['camera_feature']['geometry_feature'].squeeze(1)  # [bs, num_views, 519, 2048]

        all_frames_status_feature = torch.stack(
            [
                features['status_feature_prev_3'],
                # features['status_feature_prev_2'],
                # features['status_feature_prev_1'],
                features['status_feature'],
                # features['status_feature_next_2'],
                features['status_feature_next_4'],
                # features['status_feature_next_6'],
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
        ego_query = self._query_embedding.weight[None, None, ...].repeat(batch_size, num_frames, 1, 1)

        if self._config.num_mode:
            mode_ref = self.waypoint_mode_ref.weight
            if self._config.use_cmd_embed:
                mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, self._num_poses, 1)
            else:
                mode_ref = mode_ref.reshape(self._num_mode, self._num_poses, self._config.tf_d_model)
                mode_ref = mode_ref.to(device).unsqueeze(0).unsqueeze(1).repeat(batch_size, num_frames, 1, 1, 1)

            mode_ref_feat = self._mode_embedding(mode_ref)
            ego_query = ego_query.clone() + mode_ref_feat.reshape(batch_size, num_frames, -1, mode_ref_feat.shape[-1]).clone()

        ego_query_out = self._tf_decoder(
            ego_query.reshape(-1, *ego_query.shape[-2:]),
            keyval_final.reshape(-1, *keyval_final.shape[-2:])
        )


        # predict geometry feature from dino_feature
        dino_full_feature = image_scene_query.reshape(batch_size, num_frames, num_views, -1, dim)[..., self._config.num_scene_query_token+1:, :]  # [bs, num_frames, num_views, 512, 256] (513 token 减去 cls token)
        # init_geometry_query = self.geometry_query.repeat(batch_size, 1, 1, 1, 1)
        # geometry_feature_pred = self.geometry_decoder(dino_full_feature.reshape(batch_size, -1, dim), dino_full_feature.reshape(batch_size, -1, dim)).reshape(batch_size, num_frames, num_views, -1, dim)
        geometry_feature_align = self.geometry_projector(dino_full_feature)  # [bs, num_frames, num_views, 512, 2048]

        # wm - 基于 ar_ratio 选择不同的训练模式
        if self.use_wm:
            trajectory = {}
            trajectory['first_traj'] = self._trajectory_head(ego_query_out.reshape(batch_size, num_frames, -1, dim)[:, 1, ...], cmd=cmd[:, 1])

            trajectory['geometry_feature_align'] = geometry_feature_align
            trajectory['geometry_feature_gt'] = all_frames_geometry_feature_gt[..., 7:, :]

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
        
            # predicted cmd, velocity, and acceleration
            ego_status_pred = wm_next_latent.reshape(batch_size, num_frames-1, -1, dim)[..., 0, :]
            trajectory['cmd_pred'], trajectory['velocity_pred'], trajectory['acceleration_pred'] = self.ego_status_decoder(ego_status_pred)
            trajectory['cmd_gt'], trajectory['velocity_gt'], trajectory['acceleration_gt'] = cmd[:, 1:], all_frames_status_feature[:, 1:, 4:6], all_frames_status_feature[:, 1:, 6:8]
            
            # GT latent for loss calculation
            trajectory['gt_next_latent'] = gt_keyval_final[:, 1:, ...]

        return trajectory

    def compute_traj_loss(
        self,
        trajectories: Dict[str, torch.Tensor],
        gt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算轨迹损失。

        Args:
            trajectories: 预测轨迹字典。
            gt: 真实轨迹标签。

        Returns:
            损失字典。
        """
        loss_dict: Dict[str, torch.Tensor] = {}
        all_trajs = trajectories.get("all_trajectories")  # [B, M, T, 3]
        B, M = all_trajs.shape[0], all_trajs.shape[1]
        
        # 原 hard winner-take-all 方式
        traj_gt_dist = torch.norm(all_trajs[..., :2] - gt[:, None, :, :2], p=1, dim=-1)  # [B,M,T]
        fde_dist = traj_gt_dist[:, :, -1]  # [B,M]
        best_mode = torch.argmin(fde_dist, dim=-1)  # [B]
        best_traj = all_trajs[torch.arange(B), best_mode]
        trajectory_loss = torch.nn.functional.l1_loss(best_traj, gt)

        # # 分类 hard label
        # if "cls_logits" in trajectories:
        #     cls_loss = torch.nn.functional.cross_entropy(trajectories["cls_logits"], best_mode)
        #     loss_dict["cls_loss"] = cls_loss * self.traj_cls_loss_weight

        loss_dict["traj_loss"] = trajectory_loss * self.traj_loss_weight
        return loss_dict

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        logging_prefix: Optional[str] = None,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        """计算模型损失。

        Args:
            features: 输入特征。
            targets: 目标标签。
            predictions: 模型预测结果。
            logging_prefix: 日志前缀（可选）。

        Returns:
            总损失张量。
        """
        loss_dict: Dict[str, torch.Tensor] = {}
        # ========================= 多模态监督 =========================
        if self._config.num_mode:
            # first trajectory
            first_trajectory = predictions['first_traj']
            # refined_trajectory = predictions['refined_traj']
            gt = targets["trajectory"]  # [B, T, 3]

            # first trajectory loss
            first_traj_loss_dict = self.compute_traj_loss(first_trajectory, gt)
            # refined_traj_loss_dict = self.compute_traj_loss(refined_trajectory, gt)

            loss_dict.update(
                {
                    "first_traj_loss": first_traj_loss_dict["traj_loss"],
                }
            )

            # # 控制命令预测loss
            # if "cmd_logits" in predictions:
            #     cmd = predictions['cmd_gt']
            #     cmd_loss = torch.nn.functional.cross_entropy(predictions["cmd_logits"], cmd)
            #     loss_dict["cmd_loss"] = cmd_loss * self._config.traj_cmd_loss_weight
        else:
            # 单模态或非训练阶段
            if "trajectory" in predictions:
                trajectory_loss = torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])
                loss_dict["traj_loss"] = trajectory_loss * self.traj_loss_weight

        def wm_loss(wm_next_latent, gt_next_latent):
            # wm_pred = F.normalize(wm_next_latent, dim=-1)
            # wm_gt = F.normalize(gt_next_latent, dim=-1)
            wm_loss = F.mse_loss(wm_next_latent, gt_next_latent)
            return wm_loss

        # 世界模型损失
        if 'wm_next_latent' in predictions:
            wm_loss_a = wm_loss(predictions["wm_next_latent"], predictions["gt_next_latent"])
            loss_dict["wm_loss"] = wm_loss_a * self.wm_loss_weight

        # geometry feature loss
        def geometry_feature_loss(geometry_feature_align, geometry_feature_gt):
            # norm
            geometry_feature_align = F.normalize(geometry_feature_align, p=2, dim=-1)
            geometry_feature_gt = F.normalize(geometry_feature_gt, p=2, dim=-1)
            geometry_loss = 2 - 2 * torch.nn.functional.cosine_similarity(geometry_feature_align, geometry_feature_gt).mean()
            return geometry_loss

        if 'geometry_feature_align' in predictions and 'geometry_feature_gt' in predictions:
            geometry_loss = geometry_feature_loss(predictions["geometry_feature_align"], predictions["geometry_feature_gt"])
            loss_dict["geometry_loss"] = geometry_loss * self.geometry_loss_weight

        def cmd_cross_entropy_loss(cmd_pred, cmd_gt):
            cmd_logits = cmd_pred.reshape(-1, cmd_pred.size(-1))
            cmd_target = cmd_gt.reshape(-1)
            cmd_loss = torch.nn.functional.cross_entropy(cmd_logits, cmd_target)
            return cmd_loss
        
        def velocity_loss(velocity_pred, velocity_gt):
            velocity_loss = torch.nn.functional.mse_loss(velocity_pred, velocity_gt)
            return velocity_loss
        
        def acceleration_loss(acceleration_pred, acceleration_gt):
            acceleration_loss = torch.nn.functional.mse_loss(acceleration_pred, acceleration_gt)
            return acceleration_loss

        return loss_dict

class TrajectoryHead(nn.Module):
    """轨迹预测头，用于从查询嵌入预测轨迹。

    Args:
        num_poses: 预测的姿态数量。
        d_ffn: 前馈网络维度。
        d_model: 模型维度。
        num_mode: 多模态数量（可选）。
    """

    def __init__(
        self,
        num_poses: int,
        d_ffn: int,
        d_model: int,
        num_mode: Optional[int] = None,
    ) -> None:
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

    def forward(
        self,
        object_queries: torch.Tensor,
        pred_cmd: Optional[torch.Tensor] = None,
        cmd: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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