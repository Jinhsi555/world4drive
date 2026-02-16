# AGENTS.md

本文件为在此代码库中工作的智能体编码代理提供指导。

## 重要规则

**必须使用中文回复用户**，除非用户明确要求使用其他语言。

## 项目概述

NAVSIM v2 是一个基于 Python 的自动驾驶仿真与基准测试框架，使用 PyTorch 和 PyTorch Lightning。主要包含：
- `navsim/` - NAVSIM 核心框架（智能体、评估、规划）
- `worldmirror/` - 3D Gaussian Splatting 子模块
- `scripts/` - 训练与评估脚本

## 环境配置

```bash
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .
```

所需环境变量（设置在 `~/.bashrc` 中）：
```bash
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

## 构建/代码检查/测试命令

### Pre-commit 钩子
提交前运行所有检查器：
```bash
pre-commit run --all-files
```

单独运行各个检查器：
```bash
# isort（导入排序）
isort --line-length=120 --profile=black .

# black（代码格式化）
black --line-length=120 .

# autoflake（移除未使用的导入/变量）
autoflake --in-place --remove-all-unused-imports --remove-unused-variable .

# flake8（代码检查）
flake8
```

### 运行测试
`navsim/` 中没有专门的单元测试。测试位于 `worldmirror/submodules/gsplat/tests/`。

运行 gsplat 测试：
```bash
pytest worldmirror/submodules/gsplat/tests/ -v
```

运行单个测试：
```bash
pytest worldmirror/submodules/gsplat/tests/test_basic.py::test_constructor -v
```

### 训练命令
```bash
# 训练智能体
python navsim/planning/script/run_training.py \
    agent=transfuser_agent \
    dataloader.params.batch_size=64 \
    experiment_name=my_experiment \
    train_test_split=navtrain \
    use_cache_without_dataset=True \
    cache_path=$NAVSIM_EXP_ROOT/cache_for_training
```

### 评估命令
```bash
# 评估智能体
python navsim/planning/script/run_pdm_score.py \
    train_test_split=navhard_two_stage \
    agent=transfuser_agent \
    agent.checkpoint_path=/path/to/checkpoint.ckpt \
    experiment_name=my_experiment

# 指标缓存
python navsim/planning/script/run_dataset_caching.py
```

## 代码风格指南

### 格式化
- **最大行长度**：120 个字符
- **格式化工具**：Black，配置 `--line-length=120`
- **导入排序**：isort，配置 profile=black, multi_line_output=3, include_trailing_comma=true
- **配置文件**：参见 `.isort.cfg`

### 导入规范
- 标准库导入放在最前面
- 第三方导入放在中间
- 本地导入放在最后
- 使用绝对导入：`from navsim.agents.abstract_agent import AbstractAgent`
- 使用 isort 排序导入（通过 pre-commit 自动完成）

### 类型注解
- 为函数签名使用 Python 类型注解
- 常用类型：`Dict`、`List`、`Optional`、`Tuple`、`Union`
- PyTorch 张量使用 `torch.Tensor`
- 示例：
```python
def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
    ...
```

### 命名约定
- **类名**：PascalCase（例如 `AbstractAgent`、`ConstantVelocityAgent`）
- **函数/方法**：snake_case（例如 `compute_trajectory`、`get_sensor_config`）
- **常量**：SCREAMING_SNAKE_CASE（例如 `NAVSIM_INTERVAL_LENGTH`）
- **私有方法**：以下划线开头（例如 `_internal_method`）

### 数据类
- 使用 `@dataclass` 作为简单数据容器
- 位于 `navsim/common/dataclasses.py`
- 示例：
```python
@dataclass
class Camera:
    image: Optional[npt.NDArray[np.float32]] = None
    sensor2lidar_rotation: Optional[npt.NDArray[np.float32]] = None
    ...
```

### 错误处理
- 抽象方法使用 `raise NotImplementedError("message")`
- 无效参数使用 `raise ValueError("message")`
- 避免使用裸 `except:`，应捕获具体异常
- 在文档字符串中适当记录异常

### 文档字符串
- 公共方法使用 Google 风格的文档字符串
- 包含：描述、参数、返回值
- 示例：
```python
def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
    """计算自车轨迹。
    
    Args:
        agent_input: 包含智能体输入的数据类。
        
    Returns:
        表示自车未来位置预测的轨迹。
    """
```

### 智能体开发
- 继承 `navsim/agents/abstract_agent.py` 中的 `AbstractAgent`
- 实现必需方法：`name()`、`get_sensor_config()`、`initialize()`、`compute_trajectory()`
- 学习型智能体还需实现：`forward()`、`compute_loss()`、`get_optimizers()`、`get_feature_builders()`、`get_target_builders()`
- 在 `navsim/planning/script/config/agent/` 中添加配置

### 配置系统
- 使用 Hydra 进行配置管理
- 配置文件位于 `navsim/planning/script/config/`
- 通过命令行覆盖配置：`python script.py param=value`

### 关键目录
- `navsim/agents/` - 智能体实现
- `navsim/common/dataclasses.py` - 数据结构
- `navsim/evaluate/pdm_score.py` - 评估逻辑
- `navsim/planning/training/` - 训练流水线
- `navsim/planning/simulation/` - 仿真引擎

### 重要说明
- **禁止**使用 `test`/`navtest`/`navhard_two_stage`/`warmup_two_stage`/`private_test_two_stage` 数据集划分进行挑战赛提交的训练
- 轨迹输出必须为局部坐标（x, y, heading）
- 评估时长为 4 秒，频率 10Hz（40 个姿态）
- 无传感器输入的智能体使用 `SensorConfig.build_no_sensors()`

## W4D 模型训练稳定性讨论

### 关键文件
- `navsim/agents/transfuser/w4d_model_dino_lora_geometry_no_refine.py` - W4D 模型主实现
- `navsim/agents/transfuser/temporal_world_model.py` - 带 3D/4D RoPE 的时序世界模型

### 架构概述
- **Online DINO Encoder**: LoRA 微调的 DINOv2 编码器，用于视觉特征提取
- **Target DINO Encoder**: Online 编码器的 EMA 更新副本，用于生成 GT 特征
- **DINO Projector**: 将 DINO 特征 (384d) 投影到 Transformer 空间 (256d)
- **Temporal World Model**: 使用带 RoPE 的 Transformer decoder 预测未来帧 latent
- **Geometry Decoder**: 从 DINO 特征预测几何特征

### 训练稳定性问题及解决方案

#### 问题 1: WM Loss 不稳定（梯度爆炸）
**现象**:
- WM loss 突然飙升，然后急速下降，之后缓慢上升
- 梯度在飙升时变大到原来的 3 倍
- Geometry loss 在权重较大时也会变得不稳定

**根本原因**:
1. MSE loss 对尺度敏感: `loss = ||pred - gt||²`
2. 模型学会降低 latent 的 norm 而不是对齐特征
3. Norm 突然增大 → MSE loss 爆炸 → 梯度爆炸 → 模型学会输出更小的值

**解决方案**:
```python
# 修改前（不稳定）
wm_loss_a = F.mse_loss(predictions["wm_next_latent"], predictions["gt_next_latent"])

# 修改后（稳定）- 方案 1: 归一化后 MSE
wm_pred = F.normalize(predictions["wm_next_latent"], dim=-1)
wm_gt = F.normalize(predictions["gt_next_latent"], dim=-1)
wm_loss_a = F.mse_loss(wm_pred, wm_gt)

# 修改后（稳定）- 方案 2: Cosine Loss（推荐）
wm_loss_a = 2 - 2 * F.cosine_similarity(
    predictions["wm_next_latent"], 
    predictions["gt_next_latent"]
).mean()
```

#### 问题 2: Geometry Loss 与 WM Loss 平衡
**发现**: Geometry loss 使用 cosine similarity（天然归一化），而 WM loss 使用 MSE（对尺度敏感）。这导致两者的稳定性特征不同。

**解决方案**: 对两个 loss 都应用归一化，保持一致的行为。

#### 问题 3: DINO Projector 缺少归一化
**问题**: Projector 输出的幅度可能不稳定。

**DINO 输出特点**: DINO encoder 输出已经过 `model.layernorm`，输入 projector 时已归一化。

**推荐方案**:
```python
self.dino_projector = nn.Sequential(
    nn.Linear(config.dino_d_model, config.tf_d_ffn),
    nn.LayerNorm(config.tf_d_ffn),       # 中间层归一化，稳定激活
    nn.GELU(),
    nn.Dropout(config.tf_dropout),
    nn.Linear(config.tf_d_ffn, config.tf_d_model),
    nn.LayerNorm(config.tf_d_model),     # 输出归一化，确保特征稳定
)
```

**各层作用**:

| Layer | 作用 |
|-------|------|
| 中间 LayerNorm | 稳定 FFN 中间激活，防止梯度爆炸/消失 |
| 输出 LayerNorm | 确保投影后特征分布稳定，便于后续模块使用 |
| GELU vs ReLU | GELU 更平滑，训练更稳定 |

**注意事项**:
1. 不需要输入前 LayerNorm：DINO 输出已经归一化
2. 两层 LayerNorm 不冗余：中间层稳定激活值，输出层稳定特征分布
3. 与后续模块配合：输出 LayerNorm 确保 `keyval + embedding` 操作稳定

**极简方案**:
```python
self.dino_projector = nn.Sequential(
    nn.Linear(config.dino_d_model, config.tf_d_ffn),
    nn.GELU(),
    nn.Linear(config.tf_d_ffn, config.tf_d_model),
    nn.LayerNorm(config.tf_d_model),  # 最少保留输出 LayerNorm
)
```

### Loss 权重建议
1. 初始使用较低权重 (0.2) 作为 geometry_loss_weight 和 wm_loss_weight
2. 归一化后，先观察梯度大小再调整权重
3. 目标: 让不同 loss 的梯度量级相近
4. 考虑自动权重学习（Uncertainty Weighting）:
```python
log_wm_var = nn.Parameter(torch.zeros(1))
wm_loss = torch.exp(-log_wm_var) * wm_loss_a + log_wm_var
```

### 特征预测任务：MSE vs Cosine Loss
对于特征空间一致的预测未来帧特征任务：

| 方面 | MSE Loss | Cosine Loss |
|------|----------|-------------|
| 优化目标 | 方向 + 幅度 | 仅方向 |
| 对尺度敏感 | 是 | 否 |
| 训练稳定性 | 差 | 好 |
| 保留信息 | 幅度信息 | 丢失幅度 |

**建议**: 使用 Cosine Loss 获得稳定性。对于此任务，特征方向（语义内容）比幅度更重要。

#### Cosine Loss 公式选择：`2 - 2*cos_sim` vs `1 - cos_sim`

**数学推导**：
归一化向量的欧氏距离平方：
```
||a - b||² = ||a||² + ||b||² - 2·(a·b)
```
如果 `||a|| = ||b|| = 1`（归一化后）：
```
||a - b||² = 1 + 1 - 2·cos(a,b) = 2 - 2·cos_sim
```

**对比**：

| 公式 | 范围 | 含义 |
|------|------|------|
| `1 - cos_sim` | [0, 2] | 简单的余弦距离 |
| `2 - 2*cos_sim` | [0, 4] | 归一化后的欧氏距离平方 |

**`2 - 2*cos_sim` 的好处**：
1. **等价于归一化 MSE**：`2 - 2*cos_sim = ||norm(a) - norm(b)||²`
2. **梯度是两倍**：当需要更大梯度时，不需要调整 loss weight
3. **几何意义明确**：直接表示特征空间中的距离

**结论**：两者都可以，`1 - cos_sim` 也完全没问题。选择 `2 - 2*cos_sim` 主要是为了和"归一化后 MSE"保持一致的数值尺度。

### 梯度裁剪
添加梯度裁剪防止梯度爆炸:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Target Encoder 的 EMA Momentum
确保 target encoder 的 EMA momentum 足够高 (0.996-0.999)，防止变化过快导致训练不稳定。
