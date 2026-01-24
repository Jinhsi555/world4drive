#!/usr/bin/env python3
"""
使用navsim项目中的函数读取gz文件
基于navsim/planning/training/dataset.py中的load_feature_target_from_pickle函数
"""

import gzip
import pickle
import sys
from pathlib import Path
import torch
from tqdm import tqdm

def load_feature_target_from_pickle(path: Path) -> dict:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict

def dump_feature_target_to_pickle(path: Path, data_dict) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)

def read_gz_file_simple(file_path):
    """读取并显示gz文件内容（使用navsim项目的方法）"""
    try:
        file_path = Path(file_path)
        data_dict = load_feature_target_from_pickle(file_path)
        
        new_data_dict = {
            'dino_feature': data_dict['dino_feature'][None, None, 1, ...],
            'geometry_feature': data_dict['geometry_feature'][:, None, 1, ...]
        }

        single_view_path = Path(str(file_path.parent) + '_single_view') / (file_path.stem + ".gz")
        dump_feature_target_to_pickle(single_view_path, new_data_dict)
        
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")

def find_and_read_gz_files_simple(directory_path):
    """查找并读取目录中的所有gz文件（使用navsim项目的方法）"""
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"目录不存在: {directory_path}")
        return
    
    # 查找所有.gz文件
    gz_files = list(directory.rglob("*.gz"))
    
    if not gz_files:
        print(f"在目录中未找到.gz文件: {directory_path}")
        # 检查子目录
        subdirs = [d for d in directory.iterdir() if d.is_dir()]
        print(f"目录中的子目录: {[str(d.name) for d in subdirs]}")
        return
    
    print(f"找到 {len(gz_files)} 个.gz文件:")
    for gz_file in tqdm(gz_files):
        read_gz_file_simple(gz_file)

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("使用方法: python read_gz_files_simple.py <目录路径>")
    #     print("示例: python read_gz_files_simple.py /path/to/directory")
    #     sys.exit(1)
    
    target_directory = "/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp/dino_geometry_cache_test/feature_cache"
    find_and_read_gz_files_simple(target_directory)