# Flash Attention 集成到 TemporalWorldModel

## 目标
将 `temporal_world_model.py` 中的手写注意力计算替换为 PyTorch 2.0+ 的 `F.scaled_dot_product_attention`，实现 Flash Attention 优化。

## 实施步骤

### 1. 修改 RoPEMultiheadAttention.forward 方法

**文件**: `/vepfs-mlp2/c20250502/haoce/wlb/world4drive/navsim/agents/transfuser/temporal_world_model.py`

**位置**: 第 53-119 行的 `forward` 方法

**修改内容**:

保留前半部分（第 68-86 行）：
- Q/K/V 线性投影
- 多头维度变换
- RoPE 位置编码应用

删除手写注意力计算部分（第 88-113 行）：
```python
# 删除以下代码：
attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
# ... (所有手动 mask 处理和 softmax 计算)
attn_output = torch.matmul(attn_weights, v)
```

替换为 Flash Attention 实现（在第 87 行之后插入）：
```python
# Prepare attention mask for SDPA
final_mask = None
if attn_mask is not None:
    # attn_mask: [tgt_len, src_len] or [batch*heads, tgt_len, src_len]
    # Convert to [batch, num_heads, tgt_len, src_len] or broadcastable format
    if attn_mask.dim() == 2:
        final_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, src_len]
    elif attn_mask.dim() == 3:
        # Assume [batch*heads, tgt_len, src_len], reshape to [batch, heads, tgt_len, src_len]
        final_mask = attn_mask.view(batch_size, self.num_heads, tgt_len, src_len)

# Merge key_padding_mask if provided
if key_padding_mask is not None:
    # key_padding_mask: [batch_size, src_len], True means IGNORE
    # Convert to additive mask: -inf for ignored positions
    padding_mask = torch.zeros(batch_size, 1, 1, src_len, dtype=q.dtype, device=q.device)
    padding_mask.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
    
    if final_mask is None:
        final_mask = padding_mask
    else:
        final_mask = final_mask + padding_mask

# Use Flash Attention via SDPA
attn_output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=final_mask,
    dropout_p=self.dropout.p if self.training else 0.0,
    is_causal=False
)
```

保留后半部分（第 115-119 行）：
- 输出维度重排和投影

### 2. 移除不再使用的 self.scale 属性（可选优化）

**位置**: 第 51 行

由于 `scaled_dot_product_attention` 内部已经处理缩放，可以移除：
```python
# 删除: self.scale = self.head_dim ** -0.5
```

### 3. 更新文档注释（可选）

在 `RoPEMultiheadAttention` 类的 docstring 中添加：
```python
"""Multi-head attention with 3D/4D RoPE support and Flash Attention.

This module implements multi-head attention that applies 3D or 4D Rotary Position
Embeddings to query and key tensors, then uses PyTorch's scaled_dot_product_attention
for optimized computation (Flash Attention on supported hardware).
"""
```

## 预期效果

1. **显存优化**: 对于 4D RoPE 场景（11 frames × 3 views × 16 × 16 = 8,448 tokens），注意力矩阵显存从 O(N²) 降为 O(N)
2. **速度提升**: 在 Ampere/Ada/Hopper GPU 上可获得 2-4x 加速
3. **兼容性**: 与现有 RoPE 实现完全兼容，无需修改其他代码
4. **自动优化**: PyTorch 根据硬件自动选择 Flash Attention、Memory Efficient 或标准实现

## 验证

修改完成后运行测试：
```bash
python /vepfs-mlp2/c20250502/haoce/wlb/world4drive/navsim/agents/transfuser/temporal_world_model.py
```

确认 3D 和 4D RoPE 测试通过，输出形状和梯度正确。

## 注意事项

- PyTorch 版本需 >= 2.0（项目使用 pytorch-lightning 2.2.1，通常对应 PyTorch 2.0+）
- 在 Turing 及更新架构 GPU 上可获得 Flash Attention 优化
- attn_mask 需为加性掩码（0 或 -inf），与现有实现一致
