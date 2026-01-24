import json
from pathlib import Path

def delete_files_from_json(json_path):
    """根据JSON文件中的路径列表删除文件"""
    
    # 读取JSON文件
    with open(json_path, 'r') as f:
        file_list = json.load(f)
    
    deleted_count = 0
    error_count = 0
    
    for file_path in file_list:
        path = Path('/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp/dino_geometry_cache_trainval_single_view/feature_cache') / Path(file_path + '.gz')
        try:
            if path.exists():
                path.unlink()
                print(f"已删除: {file_path}")
                deleted_count += 1
            else:
                print(f"文件不存在: {file_path}")
                error_count += 1
        except PermissionError:
            print(f"权限不足: {file_path}")
            error_count += 1
        except Exception as e:
            print(f"删除失败 {file_path}: {e}")
            error_count += 1
    
    print(f"\n统计: 成功删除 {deleted_count} 个文件, 失败 {error_count} 个")

delete_files_from_json('/vepfs-mlp2/c20250502/haoce/wlb/world4drive/dino_geometry_cache_trainval_single_view_corrupted_files.json')