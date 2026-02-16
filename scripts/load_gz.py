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

def load_feature_target_from_pickle(path: Path) -> dict:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict

def read_gz_file_simple(file_path):
    """读取并显示gz文件内容（使用navsim项目的方法）"""
    try:
        file_path = Path(file_path)
        data_dict = load_feature_target_from_pickle(file_path)
        
        print(f"文件: {file_path}")
        print(f"大小: {file_path.stat().st_size} 字节")
        print("内容类型:", type(data_dict))
        print("键名:", list(data_dict.keys()) if isinstance(data_dict, dict) else "数据不是字典类型")
        
        # 显示每个键的数据信息
        if isinstance(data_dict, dict):
            for key, value in data_dict.items():
                print(f"  键 '{key}': 类型={type(value)}, 形状={getattr(value, 'shape', 'N/A') if torch.is_tensor(value) else 'N/A'}")
                
                # 如果是张量，显示基本统计信息
                if torch.is_tensor(value):
                    print(f"    最小值: {value.min().item() if value.numel() > 0 else 'N/A'}")
                    print(f"    最大值: {value.max().item() if value.numel() > 0 else 'N/A'}")
                    print(f"    平均值: {value.mean().item() if value.numel() > 0 else 'N/A'}")
                elif hasattr(value, '__len__'):
                    print(f"    长度: {len(value)}")
                
                print()
        else:
            print("数据内容:", data_dict)
            
        print("-" * 50)
        
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
    for gz_file in gz_files:
        read_gz_file_simple(gz_file)

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("使用方法: python read_gz_files_simple.py <目录路径>")
    #     print("示例: python read_gz_files_simple.py /path/to/directory")
    #     sys.exit(1)
    
    target_directory = "/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp/dino_geometry_cache_test/feature_cache"
    find_and_read_gz_files_simple(target_directory)