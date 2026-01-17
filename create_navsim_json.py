import os
import json
import tqdm
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
# from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
import os
import pickle

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # 从 matplotlib.colors 导入 LogNorm
from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from pathlib import Path
from pyquaternion import Quaternion
import numpy.typing as npt
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from collections import defaultdict
from sklearn.cluster import KMeans
import gzip
import torch
from typing import Dict

ROOT = "/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/Epona/data" 
OPENSCENE_DATA_ROOT = os.getenv('OPENSCENE_DATA_ROOT', ROOT)
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', f'{ROOT}/maps')
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
# TRAIN_SPLIT_NAME = 'trainval'  
# TEST_SPLIT_NAME = 'test'
# valid_trainval_logs = os.listdir(f'{ROOT}/sensor_blobs/{TRAIN_SPLIT_NAME}') # list of log name
# valid_trainval_logs.sort()

# valid_test_logs = os.listdir(f'{ROOT}/sensor_blobs/{TEST_SPLIT_NAME}') # list of log name
# valid_test_logs.sort()

# valid_trainvaltest_logs = valid_trainval_logs + valid_test_logs
# valid_trainvaltest_logs.sort()

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--split_id', default=0, type=int, help='minibatch size')
    parser.add_argument('--split_num', default=2, type=int, help='minibatch size')
    args = parser.parse_args()
    return args

def load_single_log_db_data(scene_info_list=None, scene=None, metric_cache_loader=None):

    # lidar_pcs = log_db.lidar_pc
    # images_data = scene_info_list.image
    log_name = scene_info_list[0]['log_name']
    next_img_token = None
    img_root_path = f'{ROOT}/sensor_blobs/{SPLIT_NAME}/{log_name}'
    cameras={
        'CAM_L2':[],
        'CAM_F0':[],
        'CAM_R2':[],
        'CAM_L0':[],
        'CAM_L1':[],
        'CAM_R0':[],
        'CAM_R1':[],
        'CAM_B0':[],
        'data_root': img_root_path,
        }
    seq_tmp = []

    ego_pose_from_cam = {
        'CAM_L2':{},
        'CAM_F0':{},
        'CAM_R2':{},
        'CAM_L0':{},
        'CAM_L1':{},
        'CAM_R0':{},
        'CAM_R1':{},
        'CAM_B0':{},
    }

    scene_token = scene_info_list[0]['scene_token']
    
    for idx, img_item in enumerate(scene_info_list):
        token = img_item['token']
        
        img_info = img_item['cams']
        # for camera_channel in img_info:
        # 只取前视图
        camera_channel = 'CAM_F0'
            
        camera_intrinsic = img_info[camera_channel]['cam_intrinsic']
        img_name = os.path.basename(img_info[camera_channel]['data_path'])
        # if idx + 1 < len(scene_info_list): # 保证存在未来帧
            # print(this is the last frame)

        # next_img_token = scene_info_list[idx + 1]
        # curr_img_token = img_item.token
        # scene_token = img_item.lidar_pc.scene_token
        
        img_path = img_info[camera_channel]['data_path']
        if not os.path.exists(f'{OPENSCENE_DATA_ROOT}/sensor_blobs/{SPLIT_NAME}/{img_path}'):
            print(f'!!!!WARNING: {img_path} do not exist.')
            continue
        # curr_ego_pose = img_item['ego2global_translation']    # 这行是错的
        # curr_ego_pose_from_cam = scene.frames[idx].ego_status   # 这里的ego_pose是 x，y，yaw
        curr_ego2global_pose = img_item['ego2global_translation']  # 这里是绝对坐标x,y,z
        curr_ego2global_rotation = img_item['ego2global_rotation']  # 这里是四元数
        ego_dymamics = img_item['ego_dynamic_state']

        
        # cur_metric_cache = metric_cache_loader.get_from_token(token)

        ego_pose_from_cam[camera_channel][f'{camera_channel}/{img_name}'] = {
            'x':  curr_ego2global_pose[0],
            'y':  curr_ego2global_pose[1],
            'z':  curr_ego2global_pose[2],
            'qw': curr_ego2global_rotation[0],
            'qx': curr_ego2global_rotation[1],
            'qy': curr_ego2global_rotation[2],
            'qz': curr_ego2global_rotation[3],
            'vx': ego_dymamics[0],
            'vy': ego_dymamics[1],
            'ax': ego_dymamics[2],
            'ay': ego_dymamics[3],
            'timestamp': img_item['timestamp'],
        }
        
        seq_tmp.append(img_name)
        # pre_scene_token = scene_token
    cameras[camera_channel].append({'seq': seq_tmp, 'scene': scene_token})
    return cameras, ego_pose_from_cam

def loop_over_log_files(
    rank=0,
    workers=1,
    split_log_list=None,
    ego_save_dir='',
    seq_save_dir='', scene_loader=None):
    total_sequence = []
    # sensor_data_db_list = os.listdir(f'{ROOT}/nuplan-v1.1/sensor_blobs')
    # sensor_data_db_list.sort()
    print('In Loop over logs.')
    scene_frames_dicts = scene_loader.scene_frames_dicts
    for token, scene_info_list in tqdm.tqdm(scene_frames_dicts.items()):

        # scene = scene_loader.get_scene_from_token(token)  

        print(f">>>>Start {token}.")

        cameras, ego_pose = load_single_log_db_data(scene_info_list)
        scene_num = len(cameras['CAM_F0'])
        for i in range(scene_num):
            new_seq_meta = {
                    'CAM_F0': cameras['CAM_F0'][i]['seq'],
                    'scene': cameras['CAM_F0'][i]['scene'],
                    'data_root': cameras['data_root'],
                    'pose': f'test_ego_meta/{token}.json',
                }
            total_sequence.append(new_seq_meta)
        ego_meta_path = os.path.join(ego_save_dir, token+'.json')
        os.makedirs(os.path.dirname(ego_meta_path), exist_ok=True)
        with open(ego_meta_path, 'w') as f:
            json.dump(ego_pose, f)
        # seq_meta_path = os.path.join(seq_save_dir, token+'.json') # 旧用法，似乎有错误
    seq_meta_path = os.path.join(seq_save_dir, 'test_meta.json')
    os.makedirs(os.path.dirname(seq_meta_path), exist_ok=True)
    with open(seq_meta_path, 'w') as f:
        json.dump(total_sequence, f)
    return total_sequence


def accumulate_results(all_results):
    accumulated_results = []
    for result in all_results:
        accumulated_results.extend(result)
    return accumulated_results

if __name__ == '__main__':
    args = add_arguments()
    split_num = args.split_num
    split_id = args.split_id
    num_processes = args.workers
    print('##############', split_id, split_num)
    SPLIT_NAME = 'test'  # trainval
    # split_name = 'trainval'
    ego_save_dir = '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/Epona/data/test_ego_meta' # your path here
    seq_save_dir = '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/Epona/data/' # your path here
    # split_name = 'trainval'
    # if split_name == 'trainval':
    #     logs_list = valid_trainval_logs
    # elif split_name == 'test':
    #     logs_list = valid_test_logs

    # split_logs_list = logs_list[split_id::split_num]

    SYNTHETIC_SENSOR_PATH='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/navhard_two_stage/sensor_blobs'
    SYNTHETIC_SCENES_PATH='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/navhard_two_stage/synthetic_scene_pickles'
    if SPLIT_NAME == 'trainval':
        NAVSIM_LOG_PATH='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/meta_datas/trainval'
    elif SPLIT_NAME == 'test':
        NAVSIM_LOG_PATH='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/meta_datas/test'
    ORIGINAL_SENSOR_PATH='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/sensor_blobs'

    NAVSIM_LOG_PATH_NAVHARD='/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/dataset/meta_datas/test'

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    FILTER = "navtest" # navtrain
    metric_cache_path = '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/metric_cache_training'
    # metric_cache_path = '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/world4drive/exp/metric_cache_navhard_two_stage'
    metric_cache_loader = MetricCacheLoader(Path(metric_cache_path))

    hydra.initialize(config_path="./navsim/planning/script/config/common/train_test_split")
    scene_filter_cfg = hydra.compose(config_name=FILTER).scene_filter
    scene_filter: SceneFilter = instantiate(scene_filter_cfg)

    # scene_filter_val: SceneFilter = instantiate(hydra.compose(config_name='navtest'))
    # openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))


    ## navhard
    # scene_loader = SceneLoader(
    #         synthetic_sensor_path=Path(SYNTHETIC_SENSOR_PATH),
    #         original_sensor_path=Path(ORIGINAL_SENSOR_PATH),
    #         data_path=Path(NAVSIM_LOG_PATH),
    #         synthetic_scenes_path=Path(SYNTHETIC_SCENES_PATH),
    #         scene_filter=scene_filter,
    #     )

    # navtrain and test
    scene_loader = SceneLoader(
            synthetic_sensor_path=Path(SYNTHETIC_SENSOR_PATH),
            original_sensor_path=Path(ORIGINAL_SENSOR_PATH),
            data_path=Path(NAVSIM_LOG_PATH),
            synthetic_scenes_path=Path(SYNTHETIC_SCENES_PATH),
            scene_filter=scene_filter,
        )

    scene_loader_tokens = scene_loader.tokens    # list


    # nuplandb_wrapper = NuPlanDBWrapper(
    #     data_root=NUPLAN_DATA_ROOT,
    #     map_root=NUPLAN_MAPS_ROOT,
    #     db_files=db_path_lists,
    #     map_version=NUPLAN_MAP_VERSION,
    # )
    print('Sart loop.')
    accumulated_results = loop_over_log_files(0, 1, ego_save_dir=ego_save_dir, seq_save_dir=seq_save_dir, scene_loader=scene_loader)