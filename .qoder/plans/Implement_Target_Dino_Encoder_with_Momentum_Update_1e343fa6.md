# Implement Target Dino Encoder with Momentum Update

## Step 1: 修改 W4DModel (navsim/agents/transfuser/w4d_model_dino_lora.py)
- 在 `__init__` 方法中，创建 `self.target_dino_encoder` 作为 `self.lora_dino_encoder` 的副本。
- 确保 `target_dino_encoder` 的所有参数 `requires_grad=False`。
- 实现 `update_target_encoder` 方法，使用公式 `target = momentum * target + (1 - momentum) * online` 更新参数。

## Step 2: 修改 AbstractAgent (navsim/agents/abstract_agent.py)
- 添加 `update_target_encoder` 作为一个空的成员函数，以便在 Lightning 模块中安全调用。

## Step 3: 修改 TransfuserAgent (navsim/agents/transfuser/transfuser_agent.py)
- 重写 `update_target_encoder` 方法，将其转发给 `self._transfuser_model.update_target_encoder()`。

## Step 4: 修改 AgentLightningModule (navsim/planning/training/agent_lightning_module.py)
- 添加 `on_train_batch_end` 生命周期钩子。
- 在钩子中调用 `self.agent.update_target_encoder()`。
