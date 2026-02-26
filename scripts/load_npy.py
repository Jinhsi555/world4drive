import numpy as np

# load npy file
file_path = "/vepfs-mlp2/c20250502/haoce/wlb/world4drive/exp/feature_cache_navtest/feature_cache/0a0d0bb11ad45a75/geometry_feature.npy"
data = np.load(file_path)
print(data.shape)
