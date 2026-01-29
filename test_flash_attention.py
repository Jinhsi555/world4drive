"""测试 Flash Attention 集成的简化脚本"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 简化版本的 RoPE 模块（用于测试）
class DummyRoPE(nn.Module):
    def forward(self, x, pos):
        return x  # 简化版本，不做实际旋转

# 从 temporal_world_model.py 复制的 Flash Attention 实现
class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with 3D/4D RoPE support and Flash Attention."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 rope_module = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.rope_module = rope_module
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                query_pos: Optional[torch.Tensor] = None,
                key_pos: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]
        
        # Linear projections and reshape to multi-head format
        q = self.q_proj(query).view(batch_size, tgt_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, src_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, src_len, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE if available and positions are provided
        if self.rope_module is not None:
            if query_pos is not None:
                q = self.rope_module(q, query_pos)
            if key_pos is not None:
                k = self.rope_module(k, key_pos)
        
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
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output


def test_flash_attention():
    print("="*60)
    print("测试 Flash Attention 集成")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    print(f"PyTorch 版本: {torch.__version__}\n")
    
    # 测试参数
    batch_size = 2
    embed_dim = 384
    num_heads = 8
    seq_len = 256
    
    # 创建模型
    model = RoPEMultiheadAttention(embed_dim, num_heads, dropout=0.1, rope_module=DummyRoPE()).to(device)
    
    # 测试 1: 基本前向传播
    print("\n" + "#"*60)
    print("# 测试 1: 基本前向传播（无 mask）")
    print("#"*60)
    
    query = torch.randn(batch_size, seq_len, embed_dim).to(device)
    key = torch.randn(batch_size, seq_len, embed_dim).to(device)
    value = torch.randn(batch_size, seq_len, embed_dim).to(device)
    
    try:
        output = model(query, key, value)
        print(f"✓ 前向传播成功!")
        print(f"  输入形状: {query.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  输出统计 - mean: {output.mean():.4f}, std: {output.std():.4f}")
        assert output.shape == query.shape, "输出形状不匹配"
        print("✓ 形状检查通过")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 2: 带 attention mask
    print("\n" + "#"*60)
    print("# 测试 2: 带 attention mask")
    print("#"*60)
    
    attn_mask = torch.zeros(seq_len, seq_len).to(device)
    attn_mask[:, seq_len//2:] = float('-inf')  # 屏蔽后半部分
    
    try:
        output = model(query, key, value, attn_mask=attn_mask)
        print(f"✓ 带 mask 的前向传播成功!")
        print(f"  Mask 形状: {attn_mask.shape}")
        print(f"  输出统计 - mean: {output.mean():.4f}, std: {output.std():.4f}")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 3: 梯度反向传播
    print("\n" + "#"*60)
    print("# 测试 3: 梯度反向传播")
    print("#"*60)
    
    model.train()
    query.requires_grad = True
    
    try:
        output = model(query, key, value)
        loss = output.sum()
        loss.backward()
        
        print(f"✓ 梯度计算成功!")
        print(f"  Query 梯度形状: {query.grad.shape}")
        print(f"  Query 梯度范数: {query.grad.norm():.4f}")
        
        has_grads = any(p.grad is not None for p in model.parameters())
        print(f"  模型参数梯度: {'已计算' if has_grads else '未计算'}")
        assert has_grads, "模型参数梯度未计算"
        print("✓ 梯度检查通过")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 4: 大序列长度（模拟 4D RoPE 场景）
    print("\n" + "#"*60)
    print("# 测试 4: 大序列长度 (模拟 4D RoPE: 11×3×16×16=8448 tokens)")
    print("#"*60)
    
    long_seq_len = 11 * 3 * 16 * 16  # 8448 tokens
    query_long = torch.randn(1, long_seq_len, embed_dim).to(device)
    key_long = torch.randn(1, long_seq_len, embed_dim).to(device)
    value_long = torch.randn(1, long_seq_len, embed_dim).to(device)
    
    try:
        model.eval()
        with torch.no_grad():
            output_long = model(query_long, key_long, value_long)
        print(f"✓ 大序列前向传播成功!")
        print(f"  序列长度: {long_seq_len}")
        print(f"  输出形状: {output_long.shape}")
        print(f"  输出统计 - mean: {output_long.mean():.4f}, std: {output_long.std():.4f}")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("所有测试通过! Flash Attention 集成成功!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_flash_attention()
    exit(0 if success else 1)
