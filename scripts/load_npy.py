import numpy as np

# load npy file
file_path = "/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp/all_info_training_cache_debug_test/feature_cache/813756e7275c5a9d/geometry_feature.npy"
data = np.load(file_path)
print(data.shape)
