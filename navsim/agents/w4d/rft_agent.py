from typing import Any, List, Dict, Optional, Union
import pickle
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
import pytorch_lightning as pl
import copy
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import warnings
from pathlib import Path
writer = SummaryWriter(os.environ.get("WRITER_PATH", f'{os.getcwd()}/log/test'))
import lzma
import hydra
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
warnings.filterwarnings(
    "ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning
)
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
# from navsim.agents.transfuser.transfuser_model import TransfuserModel
# from navsim.agents.transfuser.law_model_release import LawModel
# from navsim.agents.transfuser.rl4drive_model import LawModel
# from navsim.agents.transfuser.rl4drive_model_sum import LawModel
# from navsim.agents.transfuser.rl4drive_model_sum_wm import LawModel
# from navsim.agents.transfuser.rl4drive_model_vggt import LawModel
# from navsim.agents.transfuser.rl4drive_model_vggt_wo_pretrained import LawModel
from navsim.agents.transfuser.rl4drive_model_vggt_v2 import LawModel
from navsim.agents.transfuser.rl4drive_model_vggt_v2 import TrajectoryGaussianPolicy
# from navsim.agents.transfuser.rl4drive_model_vggt_v2_wo_pool import LawModel
from navsim.agents.transfuser.transfuser_callback import TransfuserCallback
from navsim.agents.transfuser.transfuser_loss import transfuser_loss
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig,AgentInput, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.evaluate.pdm_score import pdm_score
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.distributions import MultivariateNormal
from navsim.visualization.plots import plot_bev_with_agent_ourtraj
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig

class TransfuserAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self._transfuser_model = LawModel(config)
        self.requires_token = False
        # self.build_encoder()
        # self.build_decoder()
        self.iter = 0
        self.val_iter = 0

        # 加载模型
        if self._checkpoint_path is not None:
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
            self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)

            if self._config.training_mode == "ft" or self._config.training_mode == 'bcsl':
                self._ref_model = copy.deepcopy(self._transfuser_model)
                self._ref_model.requires_grad_(False)
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
                hydra.initialize(config_path="../../planning/script/config/pdm_scoring")
                FILTER = "default_run_pdm_score"
                metric_cache_path = "/lpai/volumes/ad-vla-bd-ga/ypx/navsim_exp/metric_cache_train"
                overrides = [
                    f"metric_cache_path={metric_cache_path}",
                ]
                pdm_score_cfg = hydra.compose(config_name=FILTER, overrides=overrides)
                self.metric_cache_loader = MetricCacheLoader(Path(pdm_score_cfg.metric_cache_path))
                self.simulator: PDMSimulator = instantiate(pdm_score_cfg.simulator)
                self.scorer: PDMScorer = instantiate(pdm_score_cfg.scorer)

        

    # def build_encoder(self):
    #     self.encoder = DiffsimEncoder(self._config)
    #     for param in self.encoder.parameters():
    #         param.requires_grad = False
    #     self.encoder.eval()

    # def build_decoder(self):
    #     if self._config.delta_arch:
    #         from navsim.agents.diffsim.diffsim_decoder_delta import DiffSimDecoder
    #     else:
    #         from navsim.agents.diffsim.diffsim_decoder import DiffSimDecoder

    #     self.traj_decoder = DiffSimDecoder(self._config)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if self._checkpoint_path is not None:
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
            self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)

            if self._config.training_mode == "ft" and self._config.use_bc_loss:
                self._ref_model = copy.deepcopy(self._transfuser_model)
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
                hydra.initialize(config_path="../../planning/script/config/pdm_scoring")
                FILTER = "default_run_pdm_score"
                metric_cache_path = "/lpai/volumes/ad-vla-bd-ga/ypx/navsim_exp/metric_cache_train_all_2"
                overrides = [
                    f"metric_cache_path={metric_cache_path}",
                ]
                pdm_score_cfg = hydra.compose(config_name=FILTER, overrides=overrides)
                self.metric_cache_loader = MetricCacheLoader(Path(pdm_score_cfg.metric_cache_path))
                self.simulator: PDMSimulator = instantiate(pdm_score_cfg.simulator)
                self.scorer: PDMScorer = instantiate(pdm_score_cfg.scorer)



    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    # def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     """Inherited, see superclass."""
    #     return self._transfuser_model(features)
    def forward_train(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._transfuser_model.forward_train(features)
    
    def forward_test(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._transfuser_model.forward_test(features)
    

    # def compute_loss(
    #     self,
    #     features: Dict[str, torch.Tensor],
    #     targets: Dict[str, torch.Tensor],
    #     predictions: Dict[str, torch.Tensor],
    # ) -> torch.Tensor:
    #     """Inherited, see superclass."""
    #     # return transfuser_loss(targets, predictions, self._config)
    #     return self._transfuser_model.compute_loss(features,targets,predictions)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        logging_prefix,
    ) -> torch.Tensor:

        trajectory_label = targets["trajectory"].to(dtype=torch.float32)    # [64,8,3]

        if self._config.training_mode == "sl":
            
            return self._transfuser_model.compute_loss(features,targets,predictions)

        elif self._config.training_mode == "ft":

            B = trajectory_label.shape[0]
            K = self._config.gr_k_sample     # 采样轨迹数，包含ref轨迹，另外采样N个

            ##### 得到actor的分布轨迹采样
            actor_traj = predictions['trajectory'].to(dtype=torch.float32)  # 需要梯度
            actor_traj_var = predictions['trajectory_var'].to(dtype=torch.float32)

            with torch.no_grad():
                ref_traj = self._ref_model.forward_train(features)['trajectory'].to(dtype=torch.float32)    #[bz, 8, 3]
                gaussian_sampler = TrajectoryGaussianPolicy(n_samples=K) 
                # 采样多条轨迹，理论上不需要梯度  
                sampled_trajs = gaussian_sampler.sample_trajectories(actor_traj[..., :2], actor_traj_var)  # [n_samples, bs, 8, 3]
            # 计算概率分布  TODO: 开根号
            logprobs = gaussian_sampler.log_prob_batch(actor_traj[..., :2], actor_traj_var, sampled_trajs)   # [[n_samples, bs]]

            # 拼上ref角度
            sampled_trajs = torch.cat([sampled_trajs, actor_traj[None, ..., 2:3].repeat(K, 1, 1, 1)], dim=-1)  # [n_samples, bs , 8, 2->3]

            # # 使用head输出角度，而不是使用ref的角度
            # yaw = self._transfuser_model._yaw_head(sampled_trajs, predictions['keyval_final']) # [k,b,8]

            # sampled_trajs = torch.cat([sampled_trajs, yaw.unsqueeze(-1)], dim=-1)  # [n_samples, bs , 8, 2->3]


            ##============================================可视化轨迹================================================
            # import matplotlib.pyplot as plt
            # # data_path = '/lpai/dataset/openscene-lrk/0-1-0/dataset/openscene-v1.1/meta_datas/trainval'

            # SPLIT = "trainval"  # ["mini", "test", "trainval"]
            # FILTER = "navtrain" # all_scenes
            # if GlobalHydra.instance().is_initialized():
            #     GlobalHydra.instance().clear()
            # hydra.initialize(config_path="../../planning/script/config/common/train_test_split/scene_filter")
            # cfg = hydra.compose(config_name=FILTER)
            # scene_filter: SceneFilter = instantiate(cfg)
            # openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

            # scene_loader = SceneLoader(
            #     openscene_data_root / f"openscene-v1.1/meta_datas/{SPLIT}",
            #     openscene_data_root / f"openscene-v1.1/sensor_blobs/{SPLIT}",
            #     scene_filter,
            #     sensor_config=SensorConfig.build_all_sensors(),
            # )
            # all_trajs_for_pdm = torch.cat([
            #     actor_traj.unsqueeze(0),      # actor 轨迹
            #     ref_traj.unsqueeze(0),        # reference 轨迹  
            #     sampled_trajs            # 采样轨迹
            # ], dim=0)  # [K+2, bs, 8, 3]
            # all_trajs_for_pdm = all_trajs_for_pdm.detach().cpu().numpy()
            # trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

            # tokens = features['token']
            # for batch_idx, token in enumerate(tokens):
            #     # cur_batch_traj = all_trajs_for_pdm[:, batch_idx]    # [10, 8, 3]
            #     # for i in range(cur_batch_traj.shape[0]):
            #     #     cur_traj = Trajectory(cur_batch_traj[i], trajectory_sampling)
            #     #     cur_batch_traj[i] = cur_traj
            #     # token = np.random.choice(scene_loader.tokens)
            #     scene = scene_loader.get_scene_from_token(token)
            #     fig, ax = plot_bev_with_agent_ourtraj(scene, all_trajs_for_pdm[:,batch_idx])
            #     # 创建保存目录
            #     save_dir = "/lpai/law-navsim/navsim_exp/rl_traj_plots"  # 或者您想要的任何路径
            #     os.makedirs(save_dir, exist_ok=True)

            #     # 生成文件名（使用时间戳和token避免覆盖）
            #     filename = f"{save_dir}/traj_{token}.png"

            #     # 保存图片
            #     fig.savefig(filename, dpi=300, bbox_inches='tight')
            #     print(f"图片已保存到: {filename}")
            #     plt.close(fig)

            ##============================================可视化轨迹================================================


            ##### 计算pdms reward
            # 合并所有轨迹用于评估 PDM
            all_trajs_for_pdm = torch.cat([
                actor_traj.unsqueeze(0),      # actor 轨迹
                ref_traj.unsqueeze(0),        # reference 轨迹  
                sampled_trajs            # 采样轨迹
            ], dim=0)  # [K+2, bs, 8, 3]
            tokens = features['token']
            pdm_scores = torch.zeros(all_trajs_for_pdm.shape[0], B, device=logprobs.device) # [n_samples +2 , bs]
            # pdms需要numpy
            all_trajs_for_pdm = all_trajs_for_pdm.detach().cpu().numpy()  
            trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
            for batch_idx, token in enumerate(tokens):
                try:
                    metric_cache_path = self.metric_cache_loader.metric_cache_paths[token]
                    with lzma.open(metric_cache_path, "rb") as f:
                        metric_cache: MetricCache = pickle.load(f)
                    # 27c28e08bde55a23 文件损坏
                    cur_batch_pred_traj = all_trajs_for_pdm[:, batch_idx] # [n_sample, 8 ,3]

                    for traj_idx in range(all_trajs_for_pdm.shape[0]):
                        cur_pred_traj = Trajectory(cur_batch_pred_traj[traj_idx, ...], trajectory_sampling) # [8,3]
                        
                        # 得到pdms
                        pdm_result = pdm_score(
                            metric_cache=metric_cache,
                            model_trajectory=cur_pred_traj,
                            future_sampling=self.simulator.proposal_sampling,
                            simulator=self.simulator,
                            scorer=self.scorer,
                        )
                        pdm_scores[traj_idx, batch_idx] = pdm_result.score

                except (EOFError, Exception) as e:
                    print(f"Skipping corrupted file for token {token}: {e}")
                    # 跳过这个样本，pdm_scores[batch_idx] 保持为 均值
                    pdm_scores[:, batch_idx] = pdm_scores.mean()


                ##============================================可视化轨迹================================================
                # scene = scene_loader.get_scene_from_token(token)
                # cur_pdm_scores = pdm_scores[:, batch_idx].unsqueeze(-1)
                # fig, ax = plot_bev_with_agent_ourtraj(scene, all_trajs_for_pdm[:, batch_idx], cur_pdm_scores)
                # # 创建保存目录
                # save_dir = "/lpai/law-navsim/navsim_exp/rl_traj_plots"  # 或者您想要的任何路径
                # os.makedirs(save_dir, exist_ok=True)

                # # 生成文件名（使用时间戳和token避免覆盖）
                # filename = f"{save_dir}/traj_{token}.png"

                # # 保存图片
                # fig.savefig(filename, dpi=300, bbox_inches='tight')
                # print(f"图片已保存到: {filename}")
                # plt.close(fig)

                ##============================================可视化轨迹================================================
            # grpo
            reward = pdm_scores.permute(1, 0)   # [B, N+2]

            # 记录实际的ref和actor的reward表现
            actor_reward_mean = reward[:, 0] # 第一维是actor [B, 1]
            ref_reward = reward[:, 1]   # 第二维是ref

            # 实际计算时需要的reward
            actor_reward = reward[:, 2:]  # [B, N]

            # pdm_scores = pdm_scores.permute(1, 0)   
            ## 计算优势
            baseline = actor_reward.mean(dim=1, keepdim=True)
            adv = (actor_reward - baseline) / (actor_reward.std(axis=-1, keepdim=True) + 1e-8)  # [B, N]
            ## 计算grpo loss
            # cur_probs = torch.exp(logprobs.permute(1, 0))   # [B, N]
            # ### 计算old和cur的分布
            # old_probs = cur_probs.clone().detach()
            # 这样做不会出错
            cur_logprobs = logprobs.permute(1, 0).contiguous()   # [K, B] -> [B, K]
            old_logprobs = cur_logprobs.clone().detach()
            log_ratio = cur_logprobs - old_logprobs
            ratio = torch.exp(log_ratio)

            # ratio = cur_probs / (old_probs + 1e-20)

            actor_loss1 = - adv * ratio   # [B， K]

            actor_loss2 = - adv * torch.clamp(ratio, 1 - self._config.grpo_cliprange, 1 + self._config.grpo_cliprange)

            actor_loss = torch.mean(torch.max(actor_loss1, actor_loss2))

            # 计算熵正则化，鼓励探索
            actor_dist = gaussian_sampler.get_distribution(actor_traj[..., :2], actor_traj_var)
            entropy = actor_dist.entropy()  # [B*8]
            entropy_loss = entropy.mean() * self._config.gr_entropy_loss_weight
            
            # 考虑计算bc loss
            ## 计算kl loss
            kl_loss = self.joint_gaussian_nll(
                actor_dist,
                ref_traj[..., :2],
            ) * self._config.gr_kl_loss_weight


            ## 计算和gt的距离
            cost_ADE = F.smooth_l1_loss(actor_traj, trajectory_label, reduction="none").mean()  
            cost_FDE = F.smooth_l1_loss(actor_traj[:,-1], trajectory_label[:,-1], reduction="none").mean()
            cost_yaw = F.smooth_l1_loss(actor_traj[:,:, -1], trajectory_label[:,:, -1], reduction="none").mean()  
            w_ade = self._config.w_ade
            w_fde = self._config.w_fde
            w_yaw = self._config.w_yaw
            bc_loss = w_ade * cost_ADE + w_fde * cost_FDE
            # bc_loss = w_ade * cost_ADE + w_fde * cost_FDE + cost_yaw * w_yaw
            # cost_ADE_ref

            # 计算总loss
            loss = actor_loss - entropy_loss + bc_loss + kl_loss

            if logging_prefix=="val":
                self.val_iter += 1
                # writer.add_scalar(f'{logging_prefix} Reward/ade', cost_ADE.mean().item(), self.val_iter)
                # writer.add_scalar(f'{logging_prefix} Reward/fde', cost_FDE.mean().item(), self.val_iter)
                writer.add_scalar(f'{logging_prefix} Reward/reward', reward.mean().item(), self.val_iter)
                writer.add_scalar(f'{logging_prefix} Loss/actor loss', actor_loss.item(), self.val_iter)
            elif logging_prefix=="train":
                self.iter += 1
                # writer.add_scalar(f'{logging_prefix} Reward/ade', cost_ADE.mean().item(), self.iter)
                # writer.add_scalar(f'{logging_prefix} Reward/fde', cost_FDE.mean().item(), self.iter)
                writer.add_scalar(f'{logging_prefix} Reward/reward', reward.mean().item(), self.iter)
                writer.add_scalar(f'{logging_prefix} Loss/actor loss', actor_loss.item(), self.iter)

            # 记录实际的ref和actor的reward表现
            actor_reward_mean = actor_reward_mean.mean()
            ref_reward_mean = ref_reward.mean()
            reward_diff = actor_reward_mean - ref_reward_mean
            

            return loss, actor_reward.mean(), actor_reward_mean, ref_reward_mean, reward_diff, bc_loss, actor_loss, kl_loss, entropy_loss
        
        elif self._config.training_mode == "bcsl":
            B = trajectory_label.shape[0]
            K = self._config.gr_k_sample     # 采样轨迹数，包含ref轨迹，另外采样N个

            ##### 得到actor的分布轨迹采样
            with torch.no_grad():
                ref_traj = self._ref_model.forward_train(features)['trajectory'].to(dtype=torch.float32)    #[bz, 8, 3]
            actor_traj = predictions['trajectory'].to(dtype=torch.float32)
            actor_traj_var = predictions['trajectory_var'].to(dtype=torch.float32)
            gaussian_sampler = TrajectoryGaussianPolicy(n_samples=K) 
            # 计算熵正则化，鼓励探索
            actor_dist = gaussian_sampler.get_distribution(actor_traj, actor_traj_var)
            entropy = actor_dist.entropy()  # [B*8]
            entropy_loss = entropy.mean() * self._config.gr_entropy_loss_weight
            
            # 考虑计算bc loss
            ## 计算kl loss
            kl_loss = self.joint_gaussian_nll(
                actor_dist,
                ref_traj[..., :2],
            ) * self._config.gr_kl_loss_weight


            ## 计算和gt的距离
            cost_ADE = F.smooth_l1_loss(actor_traj, trajectory_label, reduction="none").mean()  
            cost_FDE = F.smooth_l1_loss(actor_traj[:,-1], trajectory_label[:,-1], reduction="none").mean()
            w_ade = self._config.w_ade
            w_fde = self._config.w_fde
            bc_loss = w_ade * cost_ADE + w_fde * cost_FDE


            # 计算总loss
            loss = - entropy_loss + bc_loss + kl_loss

            return loss, entropy_loss, bc_loss, kl_loss


    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr)
    
    def get_optimizers_ft(self, num_training_steps) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        # params = [
        #     # {"params": self.encoder.parameters(), "lr": 0.01 * self._lr},
        #     {"params": self.traj_decoder.parameters(), "lr": self._lr},
        # ]
        optimizer = torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr)
        final_div_factor = self._lr / 1e-7
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self._lr,
            pct_start=0.25,
            total_steps=num_training_steps,
            final_div_factor=final_div_factor,
        )

        return [optimizer], [scheduler]

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [TransfuserCallback(self._config),pl.callbacks.ModelCheckpoint(every_n_epochs=1,save_top_k=-1)]
    
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        # forward pass
        with torch.no_grad():
            predictions = self.forward_test(features)
            poses = predictions["trajectory"].squeeze(0).numpy()

        # extract trajectory
        return Trajectory(poses)
    
    def joint_gaussian_nll(self, dist: MultivariateNormal, 
                                target: torch.Tensor) -> torch.Tensor:
        """
        简化版本，不使用 mask
        
        Args:
            dist: 分布对象
            target: [batch, seq_len, 3] 目标轨迹
        
        Returns:
            标量损失
        """
        # batch_size, seq_len, dim = target.shape
        # target_flat = target.view(-1, dim)
        
        # # 计算负对数似然
        # nll = -dist.log_prob(target_flat)  # [batch*seq_len]
        # nll = nll.view(batch_size, seq_len).mean()
        
        # return nll

        # 二元
        batch_size, seq_len, dim = target.shape
        target_flat = target.view(-1, dim)
        
        # 计算负对数似然
        nll = -dist.log_prob(target_flat)  # [batch*seq_len]
        nll = nll.view(batch_size, seq_len).mean()
        
        return nll
    
    ##=======================================独立高斯=========================================
        # 计算log_prob
        # log_prob = dist.log_prob(target)  # [batch, seq_len]
        
        # # 计算负对数似然
        # nll = -log_prob.mean()  # 对所有维度取平均
        
        # return nll
