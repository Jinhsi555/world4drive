from dataclasses import dataclass
from typing import Tuple

import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class TransfuserConfig:
    """Global TransFuser config."""
    use_depth: bool=False

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True
    cache_mode: bool = False

    # Transformer
    tf_d_model: int = 1024
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 10.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 10.0

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    ## model name
    model_name: str = "W4D"  # W4D or LAW
    model_version: int = 5  # 2 or 3 or 4

    ## Soft responsibility (multi-modal supervision) settings
    soft_resp: bool = False
    soft_resp_tau: float = 1.0
    soft_resp_topk: int = 0
    cls_entropy_weight: float = 0.0
    use_fde_for_weights: bool = True

    ## world4drive config
    num_mode: int = 18
    num_view: int = 3
    use_refine: bool = False
    traj_loss_weight: float = 1.0
    traj_cls_loss_weight: float = 0.2
    diversity_loss_weight: float = 0.0
    diversity_margin: float = 1.0  # 1m
    traj_cmd_loss_weight: float = 0.5
    use_cmd_embed: bool = True
    use_wm: bool = True
    use_wm_training: bool = True
    wm_loss_weight: float = 0.2
    use_ar: bool = False
    use_ar_wm: bool = False
    action_chunk_size: int = 4
    ar_wm_replace_prob: float = 0.1
    _use_mlp_ensemble: bool = False
    
    #epona config
    # closed_traj_evaluation: bool = False

    ## grpo config
    training_mode: str = 'sl'  # sl or ft
    ce_loss_weight: float = 0.5
    ade_best_traj_loss_weight: float = 0.1
    fde_best_traj_loss_weight: float = 0.1
    only_cls_head: bool = False  # if true, only train the cls head when in ft mode
    kl_loss_weight: float = 0.0001
    entropy_loss_weight: float = 0.001
    adv_temp: float = 1.0
    diversity_end_loss_weight: float = 1.0

    ## multi-modal trajectory vocab
    use_vocab_trajs: bool = False
    vocab_trajs_dict_path: str = ''
    vocab_trajs_path: str = ''

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
