import gzip
import pickle
import json
import os
from tqdm import tqdm

def verify_pickle_file(filepath):
    try:
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
        return True
    except Exception as e:
        print(f"Corrupted file: {filepath}, error: {e}")
        return False

# 设置检查的目录和输出文件路径
base_dir = '/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp/dino_geometry_cache_trainval/feature_cache'
output_file = 'dino_geometry_cache_trainval_corrupted_files.json'

# # 存储损坏文件的相对路径
# corrupted_files = []

# # 检查所有文件
# for root, dirs, files in os.walk(base_dir):
#     for file in tqdm(files, desc='checking gz'):
#         if file.endswith('.gz'):
#             filepath = os.path.join(root, file)
#             if not verify_pickle_file(filepath):
#                 # 计算相对路径
#                 rel_path = os.path.relpath(filepath, base_dir)
#                 print(f"发现损坏文件: {filepath}")
#                 corrupted_files.append(rel_path[:-3])

# # 保存为JSON文件
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(corrupted_files, f, ensure_ascii=False, indent=2)

# print(f"检查完成，共发现 {len(corrupted_files)} 个损坏文件")
# print(f"结果已保存到: {output_file}")

# 检查json中的文件
broken_files_list = json.load(open(output_file, 'r'))

for root, dirs, files in os.walk(base_dir):
    for file in tqdm(files, desc='checking gz'):
        if file.endswith('.gz'):
            filepath = os.path.join(root, file)
            if file[:-3] in broken_files_list:
                if not verify_pickle_file(filepath):
                    print(f"发现损坏文件: {filepath}")
