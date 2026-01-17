# 需要把一个[8192,40,3]的轨迹词典处理成[8192,8,3]的轨迹词典
import pickle
import numpy as np
# path
traj_vocab_path = '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/GTRS-eval/traj_final/8192.npy'  # (8192,40,3)
save_path = '/mnt/parallel_ssd/home/zdhs0121/Driving/navsim_workspace/Epona/exp'
# 处理轨迹
trajs_vocab = np.load(traj_vocab_path, allow_pickle=True)   # (8192,40,3)
# (40,3)是0.1s一个点，总共4s，需要处理成0.5s一个点，总共8个点,不取0点，取5,10,15,20,25,30,35,40点
indices = [4, 9, 14, 19, 24, 29, 34, 39]
trajs_vocab_reduced = trajs_vocab[:, indices, :]

# 保存处理后的轨迹词典
save_traj_vocab_path = f'{save_path}/traj_vocab_8192_8_3.npy'
np.save(save_traj_vocab_path, trajs_vocab_reduced)
print(f'Saved reduced trajectory vocabulary to: {save_traj_vocab_path}, shape: {trajs_vocab_reduced.shape}')
# 需要验证下采样是否正确
idx_to_check = 0
original_traj = trajs_vocab[idx_to_check]
reduced_traj = trajs_vocab_reduced[idx_to_check]
print(f'Original trajectory (first 5 points):\n{original_traj[:5]}')
print(f'Reduced trajectory:\n{reduced_traj}')