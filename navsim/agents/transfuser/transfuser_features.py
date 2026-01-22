from enum import IntEnum
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import numpy.typing as npt
from pathlib import Path
import pickle
import gzip

import torch
from torchvision import transforms

from shapely import affinity
from shapely.geometry import Polygon, LineString

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

from worldmirror.src.models.models.worldmirror import WorldMirror
from worldmirror.src.utils.inference_utils import prepare_images_to_tensor
from transformers import AutoImageProcessor, AutoModel

def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    data_dict_cpu = {k: v.float().detach().cpu() for k, v in data_dict.items()}
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict_cpu, f)
        
class TransfuserFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for TransFuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes feature builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DINO and WorldMirror
        if self._config.cache_mode:
            self.dino_model = AutoModel.from_pretrained('/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-large').to(self.device)
            self.dino_processor = AutoImageProcessor.from_pretrained('/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-large')
            self.dino_processor.crop_size = {'height': 224, 'width': 448}

            self.geometry_model = WorldMirror.from_pretrained("/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/HunyuanWorld-Mirror").to(self.device)
            self.geometry_processor = AutoImageProcessor.from_pretrained('/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-large')
            self.geometry_processor.crop_size = {'height': 224, 'width': 448}
            self.geometry_processor.do_normalize = False

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_feature"
    
    def get_dino_feature(self, camera_inputs: List[np.ndarray]) -> torch.Tensor:
        dino_inputs, _, _ = camera_inputs
        return self.dino_model(**self.dino_processor(images=dino_inputs, return_tensors="pt").to(self.device)).last_hidden_state
    
    def get_geometry_feature(self, camera_inputs: List[np.ndarray]) -> torch.Tensor:
        image_inputs, intrinsics, extrinsics = camera_inputs
        
        views = {}
        imgs = self.geometry_processor(images=image_inputs, return_tensors="pt").to(self.device)['pixel_values']
        views["img"] = imgs.unsqueeze(0)
        views["camera_poses"] = extrinsics
        views["camera_intrs"] = intrinsics

        cond_flags = [1, 0, 1]  # [camera_pose, depth, intrinsics]

        use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if use_amp:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float32
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=bool(use_amp), dtype=amp_dtype):
                priors = self.geometry_model.extract_priors(views)
                geometry_features_list, patch_start_idx = self.geometry_model.visual_geometry_transformer(views["img"], priors, cond_flags=cond_flags)  # list: [4 * hidden_state], patch_start_idx = 7 (camera_token, register_token*4, pose_token, ray_token)
        last_geometry_feature = geometry_features_list[-1]
        return last_geometry_feature

    def compute_features(self, agent_input: AgentInput, scene: Scene, feature_cache_path: Path) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}
        final_features = {}
        save_dict = {}
        
        # Get the token list of the scene to identify the frames
        frame_tokens = [frame.token for frame in scene.frames[:-2]]
        
        final_features["camera_feature"] = frame_tokens[3]
        final_features["camera_feature_prev_1"] = frame_tokens[2]
        final_features["camera_feature_prev_2"] = frame_tokens[1]
        final_features["camera_feature_prev_3"] = frame_tokens[0]
        final_features["camera_feature_next_1"] = frame_tokens[4]
        final_features["camera_feature_next_2"] = frame_tokens[5]
        final_features["camera_feature_next_3"] = frame_tokens[6]
        final_features["camera_feature_next_4"] = frame_tokens[7]
        final_features["camera_feature_next_5"] = frame_tokens[8]
        final_features["camera_feature_next_6"] = frame_tokens[9]
        final_features["camera_feature_next_7"] = frame_tokens[10]
        final_features["camera_feature_next_8"] = frame_tokens[11]
        
        camera_inputs = self._get_camera_feature(agent_input)
        
        final_features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[3].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[3].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[3].ego_acceleration, dtype=torch.float32),
            ],
        )
        # features['camera'],features['intrinsics']=self._get_camera_intrinsics_feature(agent_input)
        # dino_camera_feature = self.dino_processor(images=camera_feature, return_tensors="pt").to(self.device)
        features["camera_feature"] = {
            'dino_feature': self.get_dino_feature(camera_inputs),
            'geometry_feature': self.get_geometry_feature(camera_inputs)
        }

        ## 添加前几帧的信息
        camera_feature_prev_1, camera_feature_prev_2, camera_feature_prev_3 = self._get_camera_feature_prev(agent_input)
        features["camera_feature_prev_1"] = {
            'dino_feature': self.get_dino_feature(camera_feature_prev_1),
            'geometry_feature': self.get_geometry_feature(camera_feature_prev_1)
        }
        features["camera_feature_prev_2"] = {
            'dino_feature': self.get_dino_feature(camera_feature_prev_2),
            'geometry_feature': self.get_geometry_feature(camera_feature_prev_2)
        }
        features["camera_feature_prev_3"] = {
            'dino_feature': self.get_dino_feature(camera_feature_prev_3),
            'geometry_feature': self.get_geometry_feature(camera_feature_prev_3)
        }
        final_features['status_feature_prev_1'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[2].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[2].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[2].ego_acceleration, dtype=torch.float32),
            ],
        )
        final_features['status_feature_prev_2'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[1].ego_acceleration, dtype=torch.float32),
            ],
        )
        final_features['status_feature_prev_3'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[0].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[0].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[0].ego_acceleration, dtype=torch.float32),
            ],
        )

        # 添加后几帧的信息
        camera_feature_next_1, camera_feature_next_2, camera_feature_next_3, camera_feature_next_4, camera_feature_next_5, camera_feature_next_6, camera_feature_next_7, camera_feature_next_8 = self._get_camera_feature_next(agent_input)
        features["camera_feature_next_1"] = {
            'dino_feature': self.get_dino_feature(camera_feature_next_1),
            'geometry_feature': self.get_geometry_feature(camera_feature_next_1)
        }
        features["camera_feature_next_2"] = {
            'dino_feature': self.get_dino_feature(camera_feature_next_2),
            'geometry_feature': self.get_geometry_feature(camera_feature_next_2)
        }
        features["camera_feature_next_3"] = {
            'dino_feature': self.get_dino_feature(camera_feature_next_3),
            'geometry_feature': self.get_geometry_feature(camera_feature_next_3)
        }
        features["camera_feature_next_4"] = {
            'dino_feature': self.get_dino_feature(camera_feature_next_4),
            'geometry_feature': self.get_geometry_feature(camera_feature_next_4)
        }
        features["camera_feature_next_5"] = {
            'dino_feature': self.get_dino_feature(camera_feature_next_5),
            'geometry_feature': self.get_geometry_feature(camera_feature_next_5)
        }
        features["camera_feature_next_6"] = {
            'dino_feature': self.get_dino_feature(camera_feature_next_6),
            'geometry_feature': self.get_geometry_feature(camera_feature_next_6)
        }
        features["camera_feature_next_7"] = {
            'dino_feature': self.get_dino_feature(camera_feature_next_7),
            'geometry_feature': self.get_geometry_feature(camera_feature_next_7)
        }
        features["camera_feature_next_8"] = {
            'dino_feature': self.get_dino_feature(camera_feature_next_8),
            'geometry_feature': self.get_geometry_feature(camera_feature_next_8)
        }
        
        final_features['status_feature_next_1'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[4].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[4].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[4].ego_acceleration, dtype=torch.float32),
            ],
        )
        final_features['status_feature_next_2'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[5].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[5].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[5].ego_acceleration, dtype=torch.float32),
            ],
        )
        final_features['status_feature_next_3'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[6].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[6].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[6].ego_acceleration, dtype=torch.float32),
            ],
        )
        final_features['status_feature_next_4'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[7].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[7].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[7].ego_acceleration, dtype=torch.float32),
            ],
        )
        final_features['status_feature_next_5'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[8].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[8].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[8].ego_acceleration, dtype=torch.float32),
            ],
        )
        final_features['status_feature_next_6'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[9].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[9].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[9].ego_acceleration, dtype=torch.float32),
            ],
        )
        final_features['status_feature_next_7'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[10].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[10].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[10].ego_acceleration, dtype=torch.float32),
            ],
        )
        final_features['status_feature_next_8'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[11].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[11].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[11].ego_acceleration, dtype=torch.float32),
            ],
        )

        # 添加历史轨迹信息
        ## 检查agent_input是否存在属性history_trajectory
        if 'history_trajectory' in dir(agent_input):
            if agent_input.history_trajectory is not None:
                final_features['history_trajectory'] = torch.tensor(agent_input.history_trajectory, dtype=torch.float32)  # [4,3]
                
        for key, value in final_features.items():
            if 'camera_feature' in key:
                save_dict[value] = {
                    'dino_feature': features[key]['dino_feature'],
                    'geometry_feature': features[key]['geometry_feature']
                }
        
        for token_name, cache_feature in save_dict.items():
            save_path = feature_cache_path / f"{token_name}.gz"
            if save_path.exists():
                print(f"跳过已存在的文件: {save_path}")
                continue
            dump_feature_target_to_pickle(save_path, cache_feature)
            
        return final_features

    def adjust_camera_intrinsics(
        self,
        intrinsic_matrix: np.ndarray,
        original_size: tuple,
        resize_size: tuple,
        crop_size: tuple
    ) -> np.ndarray:
        """
        Adjust camera intrinsic matrix after resize and center crop operations.

        Args:
            intrinsic_matrix: Original 3x3 camera intrinsic matrix K
                [[fx, 0,  cx],
                [0,  fy, cy],
                [0,  0,  1]]
            original_size: Tuple of (width, height) of original image
            resize_size: Tuple of (width, height) after resize
            crop_size: Tuple of (width, height) after center crop

        Returns:
            Adjusted 3x3 camera intrinsic matrix K' for the preprocessed image

        Example:
            >>> K = np.array([[1000, 0, 500],
            ...               [0, 1000, 400],
            ...               [0, 0, 1]])
            >>> K_new = adjust_camera_intrinsics(K, (640, 480), (256, 314), (224, 224))
        """
        # Input validation
        assert intrinsic_matrix.shape == (3, 3), "Intrinsic matrix must be 3x3"
        assert len(original_size) == 2, "Original size must be (width, height)"
        assert len(resize_size) == 2, "Resize size must be (width, height)"
        assert len(crop_size) == 2, "Crop size must be (width, height)"

        orig_w, orig_h = original_size
        resize_w, resize_h = resize_size
        crop_w, crop_h = crop_size

        # Step 1: Resize operation
        # Scale factors
        scale_x = resize_w / orig_w
        scale_y = resize_h / orig_h

        # Step 2: Center crop operation
        # Calculate crop offsets (top-left corner of crop region)
        crop_offset_x = (resize_w - crop_w) / 2
        crop_offset_y = (resize_h - crop_h) / 2

        # Extract original intrinsic parameters
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        # Adjust focal lengths (scale with resize)
        fx_new = fx * scale_x
        fy_new = fy * scale_y

        # Adjust principal points (scale with resize, then offset by crop)
        cx_new = cx * scale_x - crop_offset_x
        cy_new = cy * scale_y - crop_offset_y

        # Build new intrinsic matrix
        K_new = np.array([
            [fx_new, 0,       cx_new],
            [0,       fy_new, cy_new],
            [0,       0,      1      ]
        ])

        return K_new
    
    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        # cameras = agent_input.cameras[-1]
        cameras = agent_input.cameras[3]

        # Crop to ensure 4:1 aspect ratio
        l0 = cameras.cam_l0.image
        f0 = cameras.cam_f0.image
        r0 = cameras.cam_r0.image

        image_list = []
        intrinsics_list = []
        extrinsics_list = []
        for camera_token in ['cam_l0', 'cam_f0', 'cam_r0']:
            camera = cameras.__getattribute__(camera_token)
            
            intrinsic = torch.tensor(
                self.adjust_camera_intrinsics(
                    intrinsic_matrix=camera.intrinsics,
                    original_size=(1920, 1080),
                    resize_size=(455, 256),
                    crop_size=(448, 224),
                ),
                device=self.device,
                dtype=torch.float32
            )

            c2w = np.eye(4)
            c2w[:3, :3] = camera.sensor2lidar_rotation
            c2w[:3, 3] = camera.sensor2lidar_translation
            w2c = np.linalg.inv(c2w)
            extrinsic = torch.tensor(w2c, device=self.device, dtype=torch.float32)

            image_list.append(camera.image)
            intrinsics_list.append(intrinsic)
            extrinsics_list.append(extrinsic)

        return (
            np.stack(image_list, axis=0), 
            torch.stack(intrinsics_list, dim=0).unsqueeze(0), 
            torch.stack(extrinsics_list, dim=0).unsqueeze(0), 
        )
    
    def _get_camera_feature_prev(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        
        def proc(cameras):
            image_list = []
            intrinsics_list = []
            extrinsics_list = []
            for camera_token in ['cam_l0', 'cam_f0', 'cam_r0']:
                camera = cameras.__getattribute__(camera_token)
                
                intrinsic = torch.tensor(
                    self.adjust_camera_intrinsics(
                        intrinsic_matrix=camera.intrinsics,
                        original_size=(1920, 1080),
                        resize_size=(455, 256),
                        crop_size=(448, 224),
                    ),
                    device=self.device,
                    dtype=torch.float32
                )

                c2w = np.eye(4)
                c2w[:3, :3] = camera.sensor2lidar_rotation
                c2w[:3, 3] = camera.sensor2lidar_translation
                w2c = np.linalg.inv(c2w)
                extrinsic = torch.tensor(w2c, device=self.device, dtype=torch.float32)

                image_list.append(camera.image)
                intrinsics_list.append(intrinsic)
                extrinsics_list.append(extrinsic)
            
            return (
                np.stack(image_list, axis=0), 
                torch.stack(intrinsics_list, dim=0).unsqueeze(0), 
                torch.stack(extrinsics_list, dim=0).unsqueeze(0), 
            )

        # 依次处理 3,2,1 帧
        return tuple(proc(agent_input.cameras[i]) for i in (2, 1, 0))

    def _get_camera_feature_next(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
    
        def proc(cameras):
            image_list = []
            intrinsics_list = []
            extrinsics_list = []
            for camera_token in ['cam_l0', 'cam_f0', 'cam_r0']:
                camera = cameras.__getattribute__(camera_token)
                
                intrinsic = torch.tensor(
                    self.adjust_camera_intrinsics(
                        intrinsic_matrix=camera.intrinsics,
                        original_size=(1920, 1080),
                        resize_size=(455, 256),
                        crop_size=(448, 224),
                    ),
                    device=self.device,
                    dtype=torch.float32
                )

                c2w = np.eye(4)
                c2w[:3, :3] = camera.sensor2lidar_rotation
                c2w[:3, 3] = camera.sensor2lidar_translation
                w2c = np.linalg.inv(c2w)
                extrinsic = torch.tensor(w2c, device=self.device, dtype=torch.float32)

                image_list.append(camera.image)
                intrinsics_list.append(intrinsic)
                extrinsics_list.append(extrinsic)
            
            return (
                np.stack(image_list, axis=0), 
                torch.stack(intrinsics_list, dim=0).unsqueeze(0), 
                torch.stack(extrinsics_list, dim=0).unsqueeze(0), 
            )

        # 依次处理 +1, +2, +3, +4, +5, +6, +7, +8
        return tuple(proc(agent_input.cameras[i]) for i in (4, 5, 6, 7, 8, 9, 10, 11))

    def crop_and_resize(self,image, K, crop_pixels, new_width, new_height):
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
        # 原图尺寸
        h, w = image.shape[:2]

        # 1. 对图像上下各裁剪crop_pixels个像素
        cropped_image = image[crop_pixels:h-crop_pixels, :]  # 水平方向不裁剪
        # 更新主点的y坐标（c_y）: 原来的 c_y 需要减去裁剪掉的上部像素数
        K_crop = K.copy()
        K_crop[1, 2] -= crop_pixels

        # 裁剪后图像高度
        h_crop = h - 2 * crop_pixels

        # 2. Resize: 计算x和y方向的缩放因子
        s_x = new_width / w
        s_y = new_height / h_crop

        # resize图像
        resized_image = cv2.resize(cropped_image, (new_width, new_height))

        # 3. 更新相机内参矩阵
        K_new = K_crop.copy()
        K_new[0, 0] *= s_x  # fx
        K_new[1, 1] *= s_y  # fy
        K_new[0, 2] *= s_x  # cx
        K_new[1, 2] *= s_y  # cy

        return resized_image, K_new

    def _get_camera_intrinsics_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras[-1]

        f0 = cameras.cam_f0.image
        l0 = cameras.cam_l0.image
        r0 = cameras.cam_r0.image

        f0_intrinsics = cameras.cam_f0.intrinsics
        l0_intrinsics = cameras.cam_l0.intrinsics
        r0_intrinsics = cameras.cam_r0.intrinsics

        images_list = [f0, l0, r0]
        inputs = self.geometry_processor(images=images_list, return_tensors="pt")  # inputs.pixel_values.shape: [3, 3, 224, 224]

        intrinsics_list = [f0_intrinsics, l0_intrinsics, r0_intrinsics]

        crop_pixels = (256 - 224) // 2
        new_width = 224
        new_height = 224
        resized_image, K_new = self.crop_and_resize(f0, cameras.cam_f0.intrinsics, crop_pixels, new_width, new_height)
        
        # r0 = cameras.cam_r0.image[28:-28, 416:-416]

        # stitch l0, f0, r0 images
        # stitched_image = np.concatenate([l0, f0, r0], axis=1)
        # resized_image = cv2.resize(f0, (480, 256))
        # tensor_image = transforms.ToTensor()(resized_image)

        return resized_image,K_new

    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors
        """

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                (self._config.lidar_max_x - self._config.lidar_min_x) * int(self._config.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                (self._config.lidar_max_y - self._config.lidar_min_y) * int(self._config.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return torch.tensor(features)

class World4DriveFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for TransFuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes feature builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config
        self.cache_path = config.agent.cache_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # DINO and WorldMirror
        self.dino_model = AutoModel.from_pretrained('/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-large').to(self.device)
        self.dino_processor = AutoImageProcessor.from_pretrained('/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-large')

        self.geometry_model = WorldMirror.from_pretrained("/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/HunyuanWorld-Mirror").to(self.device)
        self.geometry_processor = AutoImageProcessor.from_pretrained('/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-large')
        self.geometry_processor.do_normalize = False

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        features["camera_feature"] = self._get_camera_feature(agent_input)
        # features["lidar_feature"] = self._get_lidar_feature(agent_input)
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[3].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[3].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[3].ego_acceleration, dtype=torch.float32),
            ],
        )
        # features['camera'],features['intrinsics']=self._get_camera_intrinsics_feature(agent_input)


        ## 添加前几帧的信息
        camera_feature_prev_1, camera_feature_prev_2, camera_feature_prev_3 = self._get_camera_feature_prev(agent_input)
        features["camera_feature_prev_1"] = camera_feature_prev_1
        features["camera_feature_prev_2"] = camera_feature_prev_2
        features["camera_feature_prev_3"] = camera_feature_prev_3
        features['status_feature_prev_1'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[2].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[2].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[2].ego_acceleration, dtype=torch.float32),
            ],
        )
        features['status_feature_prev_2'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[1].ego_acceleration, dtype=torch.float32),
            ],
        )
        features['status_feature_prev_3'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[0].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[0].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[0].ego_acceleration, dtype=torch.float32),
            ],
        )

        # 添加后几帧的信息
        camera_feature_next_1, camera_feature_next_2, camera_feature_next_3, camera_feature_next_4, camera_feature_next_5, camera_feature_next_6, camera_feature_next_7, camera_feature_next_8 = self._get_camera_feature_next(agent_input)
        features["camera_feature_next_1"] = camera_feature_next_1
        features["camera_feature_next_2"] = camera_feature_next_2
        features["camera_feature_next_3"] = camera_feature_next_3
        features["camera_feature_next_4"] = camera_feature_next_4
        features["camera_feature_next_5"] = camera_feature_next_5
        features["camera_feature_next_6"] = camera_feature_next_6
        features["camera_feature_next_7"] = camera_feature_next_7
        features["camera_feature_next_8"] = camera_feature_next_8
        
        features['status_feature_next_1'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[4].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[4].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[4].ego_acceleration, dtype=torch.float32),
            ],
        )
        features['status_feature_next_2'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[5].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[5].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[5].ego_acceleration, dtype=torch.float32),
            ],
        )
        features['status_feature_next_3'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[6].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[6].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[6].ego_acceleration, dtype=torch.float32),
            ],
        )
        features['status_feature_next_4'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[7].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[7].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[7].ego_acceleration, dtype=torch.float32),
            ],
        )
        features['status_feature_next_5'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[8].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[8].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[8].ego_acceleration, dtype=torch.float32),
            ],
        )
        features['status_feature_next_6'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[9].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[9].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[9].ego_acceleration, dtype=torch.float32),
            ],
        )
        features['status_feature_next_7'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[10].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[10].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[10].ego_acceleration, dtype=torch.float32),
            ],
        )
        features['status_feature_next_8'] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[11].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[11].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[11].ego_acceleration, dtype=torch.float32),
            ],
        )   
        # 添加历史轨迹信息
        ## 检查agent_input是否存在属性history_trajectory
        if 'history_trajectory' in dir(agent_input):
            if agent_input.history_trajectory is not None:
                features['history_trajectory'] = torch.tensor(agent_input.history_trajectory, dtype=torch.float32)  # [4,3]
        return features

    def compute_features_cache(self, agent_input: AgentInput, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}
        
        # camera_feature: [cur, prev, future]
        camera_features = {}

        camera_features['camera_features'] = self._get_camera_feature(agent_input)
        camera_feature_prev_1, camera_feature_prev_2, camera_feature_prev_3 = self._get_camera_feature_prev(agent_input)
        camera_feature_next_1, camera_feature_next_2, camera_feature_next_3, camera_feature_next_4, camera_feature_next_5, camera_feature_next_6, camera_feature_next_7, camera_feature_next_8 = self._get_camera_feature_next(agent_input)

        # cache dino and vggt feature
        for frame_token, image in camera_features.items():
            pass 
        
        # status_feature: [cur, prev, future]
        status_feature = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[3].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[3].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[3].ego_acceleration, dtype=torch.float32),
            ],
        )

        # prev
        status_feature_prev_1 = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[2].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[2].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[2].ego_acceleration, dtype=torch.float32),
            ],
        )
        status_feature_prev_2 = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[1].ego_acceleration, dtype=torch.float32),
            ],
        )
        status_feature_prev_3 = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[0].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[0].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[0].ego_acceleration, dtype=torch.float32),
            ],
        )

        # future
        status_feature_next_1 = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[4].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[4].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[4].ego_acceleration, dtype=torch.float32),
            ],
        )
        status_feature_next_2 = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[5].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[5].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[5].ego_acceleration, dtype=torch.float32),
            ],
        )
        status_feature_next_3 = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[6].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[6].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[6].ego_acceleration, dtype=torch.float32),
            ],
        )
        status_feature_next_4 = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[7].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[7].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[7].ego_acceleration, dtype=torch.float32),
            ],
        )
        status_feature_next_5 = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[8].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[8].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[8].ego_acceleration, dtype=torch.float32),
            ],
        )
        status_feature_next_6 = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[9].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[9].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[9].ego_acceleration, dtype=torch.float32),
            ],
        )
        status_feature_next_7 = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[10].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[10].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[10].ego_acceleration, dtype=torch.float32),
            ],
        )
        status_feature_next_8 = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[11].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[11].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[11].ego_acceleration, dtype=torch.float32),
            ],
        )

        # 添加历史轨迹信息
        if 'history_trajectory' in dir(agent_input):
            if agent_input.history_trajectory is not None:
                features['history_trajectory'] = torch.tensor(agent_input.history_trajectory, dtype=torch.float32)  # [4, 3]

        # get final features
        features['']
        return features
        
    # def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
    #     """
    #     Extract stitched camera from AgentInput
    #     :param agent_input: input dataclass
    #     :return: stitched front view image as torch tensor
    #     """
    #     # cameras = agent_input.cameras[-1]
    #     cameras = agent_input.cameras[3]

    #     # Crop to ensure 4:1 aspect ratio
    #     l0 = cameras.cam_l0.image[28:-28, 416:-416]
    #     f0 = cameras.cam_f0.image[28:-28]
    #     r0 = cameras.cam_r0.image[28:-28, 416:-416]

    #     # stitch l0, f0, r0 images
    #     stitched_image = np.concatenate([l0, f0, r0], axis=1)
    #     # resized_image = cv2.resize(f0, (480, 256))
    #     resized_image = cv2.resize(stitched_image, (1024, 256))
    #     tensor_image = transforms.ToTensor()(resized_image)

    #     return tensor_image

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        # cameras = agent_input.cameras[-1]
        cameras = agent_input.cameras[3]

        # Crop to ensure 4:1 aspect ratio
        l0 = cameras.cam_l0.image[28:-28, 416:-416]
        f0 = cameras.cam_f0.image[28:-28]
        r0 = cameras.cam_r0.image[28:-28, 416:-416]

        # stitch l0, f0, r0 images
        stitched_image = np.concatenate([l0, f0, r0], axis=1)
        # resized_image = cv2.resize(f0, (480, 256))
        resized_image = cv2.resize(stitched_image, (1024, 256))
        tensor_image = transforms.ToTensor()(resized_image)

        return tensor_image
    
    def _get_camera_feature_prev(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        # 内联函数：裁剪、拼接、缩放并转为Tensor
        def proc(cam):
            l0 = cam.cam_l0.image[28:-28, 416:-416]
            f0 = cam.cam_f0.image[28:-28]
            r0 = cam.cam_r0.image[28:-28, 416:-416]
            stitched = np.concatenate([l0, f0, r0], axis=1)
            resized = cv2.resize(stitched, (1024, 256))
            return transforms.ToTensor()(resized)

        # 依次处理 3,2,1 帧
        return tuple(proc(agent_input.cameras[i]) for i in (2, 1, 0))

    def _get_camera_feature_next(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        # 内联函数：裁剪、拼接、缩放并转为Tensor
        def proc(cam):
            l0 = cam.cam_l0.image[28:-28, 416:-416]
            f0 = cam.cam_f0.image[28:-28]
            r0 = cam.cam_r0.image[28:-28, 416:-416]
            stitched = np.concatenate([l0, f0, r0], axis=1)
            resized = cv2.resize(stitched, (1024, 256))
            return transforms.ToTensor()(resized)

        # 依次处理 +1, +2, +3, +4, +5, +6, +7, +8
        return tuple(proc(agent_input.cameras[i]) for i in (4, 5, 6, 7, 8, 9, 10, 11))

    def crop_and_resize(self,image, K, crop_pixels, new_width, new_height):
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
        # 原图尺寸
        h, w = image.shape[:2]

        # 1. 对图像上下各裁剪crop_pixels个像素
        cropped_image = image[crop_pixels:h-crop_pixels, :]  # 水平方向不裁剪
        # 更新主点的y坐标（c_y）: 原来的 c_y 需要减去裁剪掉的上部像素数
        K_crop = K.copy()
        K_crop[1, 2] -= crop_pixels

        # 裁剪后图像高度
        h_crop = h - 2 * crop_pixels

        # 2. Resize: 计算x和y方向的缩放因子
        s_x = new_width / w
        s_y = new_height / h_crop

        # resize图像
        resized_image = cv2.resize(cropped_image, (new_width, new_height))

        # 3. 更新相机内参矩阵
        K_new = K_crop.copy()
        K_new[0, 0] *= s_x  # fx
        K_new[1, 1] *= s_y  # fy
        K_new[0, 2] *= s_x  # cx
        K_new[1, 2] *= s_y  # cy

        return resized_image, K_new

    def _get_camera_intrinsics_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras[-1]

        # Crop to ensure 4:1 aspect ratio
        # l0 = cameras.cam_l0.image[28:-28, 416:-416]
        f0 = cameras.cam_f0.image
        crop_pixels=28
        new_width=480
        new_height=256
        resized_image,K_new=self.crop_and_resize(f0,cameras.cam_f0.intrinsics,crop_pixels,new_width,new_height)
        # r0 = cameras.cam_r0.image[28:-28, 416:-416]

        # stitch l0, f0, r0 images
        # stitched_image = np.concatenate([l0, f0, r0], axis=1)
        # resized_image = cv2.resize(f0, (480, 256))
        # tensor_image = transforms.ToTensor()(resized_image)

        return resized_image,K_new

    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors
        """

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                (self._config.lidar_max_x - self._config.lidar_min_x) * int(self._config.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                (self._config.lidar_max_y - self._config.lidar_min_y) * int(self._config.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return torch.tensor(features)


class TransfuserTargetBuilder(AbstractTargetBuilder):
    """Output target builder for TransFuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes target builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        #计算前几帧对应的历史轨迹
        scene.scene_metadata.num_history_frames = 3
        trajectory_prev_1 = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._config.trajectory_sampling.num_poses).poses
        )
        scene.scene_metadata.num_history_frames = 2
        trajectory_prev_2 = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._config.trajectory_sampling.num_poses).poses
        )
        scene.scene_metadata.num_history_frames = 1
        trajectory_prev_3 = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._config.trajectory_sampling.num_poses).poses
        )

        scene.scene_metadata.num_history_frames = 4  # 恢复默认值
        trajectory = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._config.trajectory_sampling.num_poses).poses
        )
        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        agent_states, agent_labels = self._compute_agent_targets(annotations)
        bev_semantic_map = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)

        # 计算前几帧的信息
        frame_idx_prev_1 = scene.scene_metadata.num_history_frames - 2
        frame_idx_prev_2 = scene.scene_metadata.num_history_frames - 3
        frame_idx_prev_3 = scene.scene_metadata.num_history_frames - 4
        annotations_prev_1 = scene.frames[frame_idx_prev_1].annotations
        annotations_prev_2 = scene.frames[frame_idx_prev_2].annotations
        annotations_prev_3 = scene.frames[frame_idx_prev_3].annotations
        ego_pose_prev_1 = StateSE2(*scene.frames[frame_idx_prev_1].ego_status.ego_pose)
        ego_pose_prev_2 = StateSE2(*scene.frames[frame_idx_prev_2].ego_status.ego_pose)
        ego_pose_prev_3 = StateSE2(*scene.frames[frame_idx_prev_3].ego_status.ego_pose)
        agent_states_prev_1, agent_labels_prev_1 = self._compute_agent_targets(annotations_prev_1)
        agent_states_prev_2, agent_labels_prev_2 = self._compute_agent_targets(annotations_prev_2)
        agent_states_prev_3, agent_labels_prev_3 = self._compute_agent_targets(annotations_prev_3)
        bev_semantic_map_prev_1 = self._compute_bev_semantic_map(annotations_prev_1, scene.map_api, ego_pose_prev_1)
        bev_semantic_map_prev_2 = self._compute_bev_semantic_map(annotations_prev_2, scene.map_api, ego_pose_prev_2)
        bev_semantic_map_prev_3 = self._compute_bev_semantic_map(annotations_prev_3, scene.map_api, ego_pose_prev_3)

        return {
            "trajectory": trajectory,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
            "bev_semantic_map": bev_semantic_map,
            "trajectory_prev_1": trajectory_prev_1,
            "agent_states_prev_1": agent_states_prev_1,
            "agent_labels_prev_1": agent_labels_prev_1,
            "bev_semantic_map_prev_1": bev_semantic_map_prev_1,
            "trajectory_prev_2": trajectory_prev_2,
            "agent_states_prev_2": agent_states_prev_2,
            "agent_labels_prev_2": agent_labels_prev_2,
            "bev_semantic_map_prev_2": bev_semantic_map_prev_2,
            "trajectory_prev_3": trajectory_prev_3,
            "agent_states_prev_3": agent_states_prev_3,
            "agent_labels_prev_3": agent_labels_prev_3,
            "bev_semantic_map_prev_3": bev_semantic_map_prev_3,
        }
        # return {
        #     "trajectory": trajectory,
        #     "agent_states": agent_states,
        #     "agent_labels": agent_labels,
        #     "bev_semantic_map": bev_semantic_map,
        # }

    def _compute_agent_targets(self, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x: float, y: float, config: TransfuserConfig) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (config.lidar_min_y <= y <= config.lidar_max_y)

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )

            if name == "vehicle" and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32))

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _compute_bev_semantic_map(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _compute_map_polygon_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(map_object.baseline_path.linestring, ego_pose)
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(map_linestring_mask, [points], isClosed=False, color=255, thickness=2)
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(self, annotations: Annotations, layers: TrackedObjectType) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    @staticmethod
    def _query_map_objects(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> List[MapObject]:
        """
        Queries map objects
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: list of map objects
        """

        # query map api with interesting layers
        map_object_dict = map_api.get_proximal_map_objects(point=ego_pose.point, radius=self, layers=layers)
        map_objects: List[MapObject] = []
        for layer in layers:
            map_objects += map_object_dict[layer]
        return map_objects

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)


class BoundingBox2DIndex(IntEnum):
    """Intenum for bounding boxes in TransFuser."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
