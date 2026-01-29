# Implementation of 3D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 3D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 3D spatiotemporal positions (time, height, width).

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PositionGetter:
    """Generates and caches 3D spatiotemporal positions for patches in a grid.

    This class efficiently manages the generation of spatiotemporal coordinates for patches
    in a 3D grid (time, height, width), caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, time: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatiotemporal positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            time: Number of temporal frames.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, time*height*width, 3) containing t,y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (time, height, width) not in self.position_cache:
            t_coords = torch.arange(time, device=device)
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(t_coords, y_coords, x_coords)
            self.position_cache[time, height, width] = positions

        cached_positions = self.position_cache[time, height, width]
        return cached_positions.view(1, time * height * width, 3).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding3D(nn.Module):
    """3D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    3D spatiotemporal positions (time, height, width). It handles the position-dependent 
    rotation of features separately for temporal, vertical and horizontal dimensions.

    Args:
        temporal_frequency: Base frequency for temporal dimension. Default: 10.0 (for ~12 frames)
        spatial_frequency: Base frequency for spatial dimensions. Default: 100.0 (for ~16x32 grid)
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        temporal_frequency: Base frequency for computing temporal embeddings.
        spatial_frequency: Base frequency for computing spatial embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, temporal_frequency: float = 10.0, spatial_frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 3D RoPE module."""
        super().__init__()
        self.temporal_frequency = temporal_frequency
        self.spatial_frequency = spatial_frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, base_freq: float, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            base_freq: Base frequency for this dimension.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, base_freq, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (base_freq**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 3D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 6 for 3D positioning.
            positions: Position tensor of shape (batch_size, n_tokens, 3) containing
                      the t, y and x coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 3D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        # assert tokens.size(-1) % 3 == 0, "Feature dimension must be divisible by 3 for 3D RoPE"
        assert positions.ndim == 3 and positions.shape[-1] == 3, "Positions must have shape (batch_size, n_tokens, 3)"

        # Compute feature dimension for each spatiotemporal direction
        # feature_dim = tokens.size(-1) // 3
        temp_dim = 512
        vert_dim = 256
        hori_dim = 256

        # Get frequency components for each dimension with different base frequencies
        max_temporal = int(positions[..., 0].max()) + 1
        max_vertical = int(positions[..., 1].max()) + 1
        max_horizontal = int(positions[..., 2].max()) + 1
        
        # Temporal dimension uses temporal_frequency
        cos_temp, sin_temp = self._compute_frequency_components(
            temp_dim, max_temporal, self.temporal_frequency, tokens.device, tokens.dtype
        )
        # Spatial dimensions use spatial_frequency
        cos_vert, sin_vert = self._compute_frequency_components(
            vert_dim, max_vertical, self.spatial_frequency, tokens.device, tokens.dtype
        )
        cos_hori, sin_hori = self._compute_frequency_components(
            hori_dim, max_horizontal, self.spatial_frequency, tokens.device, tokens.dtype
        )

        # Split features for temporal, vertical and horizontal processing
        temporal_features, vertical_features, horizontal_features = tokens.split([temp_dim, vert_dim, hori_dim], dim=-1)

        # Apply RoPE separately for each dimension with dimension-specific frequencies
        temporal_features = self._apply_1d_rope(temporal_features, positions[..., 0], cos_temp, sin_temp)
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 1], cos_vert, sin_vert)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 2], cos_hori, sin_hori)

        # Combine processed features
        return torch.cat((temporal_features, vertical_features, horizontal_features), dim=-1)


class PositionGetter4D:
    """Generates and caches 4D spatiotemporal positions for patches in a grid.

    This class efficiently manages the generation of spatiotemporal coordinates for patches
    in a 4D grid (time, view, height, width), caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int, int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, time: int, views: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatiotemporal positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            time: Number of temporal frames.
            views: Number of camera views.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, time*views*height*width, 4) containing t,v,h,w coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (time, views, height, width) not in self.position_cache:
            t_coords = torch.arange(time, device=device)
            v_coords = torch.arange(views, device=device)
            h_coords = torch.arange(height, device=device)
            w_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(t_coords, v_coords, h_coords, w_coords)
            self.position_cache[time, views, height, width] = positions

        cached_positions = self.position_cache[time, views, height, width]
        return cached_positions.view(1, time * views * height * width, 4).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding4D(nn.Module):
    """4D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    4D spatiotemporal positions (time, view, height, width). It handles the position-dependent 
    rotation of features separately for temporal, view, vertical and horizontal dimensions.

    Args:
        temporal_frequency: Base frequency for temporal dimension. Default: 10.0 (for ~12 frames)
        view_frequency: Base frequency for view dimension. Default: 10.0 (for ~3 views)
        spatial_frequency: Base frequency for spatial dimensions. Default: 100.0 (for ~16x16 grid)
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        temporal_frequency: Base frequency for computing temporal embeddings.
        view_frequency: Base frequency for computing view embeddings.
        spatial_frequency: Base frequency for computing spatial embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, temporal_frequency: float = 10.0, view_frequency: float = 10.0, 
                 spatial_frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 4D RoPE module."""
        super().__init__()
        self.temporal_frequency = temporal_frequency
        self.view_frequency = view_frequency
        self.spatial_frequency = spatial_frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, base_freq: float, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            base_freq: Base frequency for this dimension.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, base_freq, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (base_freq**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 4D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4 for 4D positioning.
            positions: Position tensor of shape (batch_size, n_tokens, 4) containing
                      the t, v, h and w coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 4D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        assert tokens.size(-1) % 4 == 0, "Feature dimension must be divisible by 4 for 4D RoPE"
        assert positions.ndim == 3 and positions.shape[-1] == 4, "Positions must have shape (batch_size, n_tokens, 4)"

        # Compute feature dimension for each dimension
        feature_dim = tokens.size(-1) // 4

        # Get frequency components for each dimension with different base frequencies
        max_temporal = int(positions[..., 0].max()) + 1
        max_view = int(positions[..., 1].max()) + 1
        max_vertical = int(positions[..., 2].max()) + 1
        max_horizontal = int(positions[..., 3].max()) + 1
        
        # Temporal dimension uses temporal_frequency
        cos_temp, sin_temp = self._compute_frequency_components(
            feature_dim, max_temporal, self.temporal_frequency, tokens.device, tokens.dtype
        )
        # View dimension uses view_frequency
        cos_view, sin_view = self._compute_frequency_components(
            feature_dim, max_view, self.view_frequency, tokens.device, tokens.dtype
        )
        # Spatial dimensions use spatial_frequency
        cos_vert, sin_vert = self._compute_frequency_components(
            feature_dim, max_vertical, self.spatial_frequency, tokens.device, tokens.dtype
        )
        cos_hori, sin_hori = self._compute_frequency_components(
            feature_dim, max_horizontal, self.spatial_frequency, tokens.device, tokens.dtype
        )

        # Split features for temporal, view, vertical and horizontal processing
        temporal_features, view_features, vertical_features, horizontal_features = tokens.chunk(4, dim=-1)

        # Apply RoPE separately for each dimension with dimension-specific frequencies
        temporal_features = self._apply_1d_rope(temporal_features, positions[..., 0], cos_temp, sin_temp)
        view_features = self._apply_1d_rope(view_features, positions[..., 1], cos_view, sin_view)
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 2], cos_vert, sin_vert)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 3], cos_hori, sin_hori)

        # Combine processed features
        return torch.cat((temporal_features, view_features, vertical_features, horizontal_features), dim=-1)


if __name__ == "__main__":
    print("Testing 3D Rotary Position Embedding...\n")
    
    # Test parameters matching your use case
    batch_size = 2
    n_heads = 8
    time_frames = 12
    height = 16
    width = 32
    feature_dim = 1024  # Must be divisible by 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Test 1: PositionGetter
    print("Test 1: PositionGetter")
    print(f"  Grid: {time_frames}x{height}x{width}")
    position_getter = PositionGetter()
    positions = position_getter(batch_size, time_frames, height, width, device)
    print(f"  Generated positions shape: {positions.shape}")
    print(f"  Expected shape: ({batch_size}, {time_frames*height*width}, 3)")
    print(f"  Position range - t: [{positions[..., 0].min()}, {positions[..., 0].max()}]")
    print(f"  Position range - y: [{positions[..., 1].min()}, {positions[..., 1].max()}]")
    print(f"  Position range - x: [{positions[..., 2].min()}, {positions[..., 2].max()}]")
    assert positions.shape == (batch_size, time_frames*height*width, 3), "Position shape mismatch!"
    print("  ✓ PositionGetter test passed!\n")
    
    # Test 2: RotaryPositionEmbedding3D with default frequencies
    print("Test 2: RotaryPositionEmbedding3D (default frequencies)")
    rope_3d = RotaryPositionEmbedding3D(
        temporal_frequency=10.0,
        spatial_frequency=100.0
    ).to(device)
    
    # Create sample tokens
    n_tokens = time_frames * height * width
    tokens = torch.randn(batch_size, n_heads, n_tokens, feature_dim, device=device)
    print(f"  Input tokens shape: {tokens.shape}")
    
    # Apply 3D RoPE
    output = rope_3d(tokens, positions)
    print(f"  Output shape: {output.shape}")
    assert output.shape == tokens.shape, "Output shape mismatch!"
    print(f"  Input  stats - mean: {tokens.mean():.4f}, std: {tokens.std():.4f}")
    print(f"  Output stats - mean: {output.mean():.4f}, std: {output.std():.4f}")
    print("  ✓ RotaryPositionEmbedding3D test passed!\n")
    
    # Test 3: Different frequency settings
    print("Test 3: Testing different frequency configurations")
    freq_configs = [
        (10.0, 50.0, "Low temporal, medium spatial"),
        (20.0, 100.0, "Medium temporal, high spatial"),
        (10.0, 10.0, "Uniform low frequency"),
    ]
    
    for temp_freq, spat_freq, desc in freq_configs:
        rope = RotaryPositionEmbedding3D(
            temporal_frequency=temp_freq,
            spatial_frequency=spat_freq
        ).to(device)
        out = rope(tokens, positions)
        print(f"  {desc} (t={temp_freq}, s={spat_freq})")
        print(f"    Output mean: {out.mean():.4f}, std: {out.std():.4f}")
    print("  ✓ Frequency configuration tests passed!\n")
    
    # Test 4: Cache mechanism
    print("Test 4: Testing cache mechanism")
    rope_cached = RotaryPositionEmbedding3D().to(device)
    
    # First call - populates cache
    _ = rope_cached(tokens, positions)
    cache_size_1 = len(rope_cached.frequency_cache)
    print(f"  Cache size after first call: {cache_size_1}")
    
    # Second call - should reuse cache
    _ = rope_cached(tokens, positions)
    cache_size_2 = len(rope_cached.frequency_cache)
    print(f"  Cache size after second call: {cache_size_2}")
    assert cache_size_1 == cache_size_2, "Cache not working properly!"
    print("  ✓ Cache mechanism test passed!\n")
    
    # Test 5: Gradient flow
    print("Test 5: Testing gradient flow")
    rope_grad = RotaryPositionEmbedding3D().to(device)
    tokens_grad = torch.randn(batch_size, n_heads, n_tokens, feature_dim, 
                              device=device, requires_grad=True)
    output_grad = rope_grad(tokens_grad, positions)
    loss = output_grad.sum()
    loss.backward()
    print(f"  Gradient exists: {tokens_grad.grad is not None}")
    print(f"  Gradient shape: {tokens_grad.grad.shape}")
    print(f"  Gradient mean: {tokens_grad.grad.mean():.6f}")
    assert tokens_grad.grad is not None, "Gradient not flowing!"
    print("  ✓ Gradient flow test passed!\n")
    
    # Test 6: Edge cases
    print("Test 6: Testing edge cases")
    # Single frame
    positions_single = position_getter(1, 1, 8, 8, device)
    tokens_single = torch.randn(1, 4, 64, feature_dim, device=device)
    out_single = rope_3d(tokens_single, positions_single)
    print(f"  Single frame test - output shape: {out_single.shape}")
    assert out_single.shape == tokens_single.shape
    print("  ✓ Edge case tests passed!\n")
    
    print("="*60)
    print("All tests passed successfully! ✓")
    print("="*60)
    
    # Test 4D RoPE
    print("\n" + "="*60)
    print("Testing 4D Rotary Position Embedding...")
    print("="*60)
    
    # Test parameters matching multi-view setup: 3 views, 12 frames, 16x16 grid
    batch_size_4d = 2
    n_heads_4d = 8
    num_views = 3
    time_frames_4d = 11  # 12帧时序
    height_4d = 16
    width_4d = 16
    feature_dim_4d = 256  # Must be divisible by 4
    
    print(f"\n4D RoPE Configuration:")
    print(f"  Views: {num_views}")
    print(f"  Time frames: {time_frames_4d}")
    print(f"  Spatial grid: {height_4d}x{width_4d}")
    print(f"  Feature dim: {feature_dim_4d} (per head)\n")
    
    # Test 1: PositionGetter4D
    print("Test 1: PositionGetter4D")
    position_getter_4d = PositionGetter4D()
    positions_4d = position_getter_4d(batch_size_4d, time_frames_4d, num_views, height_4d, width_4d, device)
    n_tokens_4d = time_frames_4d * num_views * height_4d * width_4d
    print(f"  Generated positions shape: {positions_4d.shape}")
    print(f"  Expected shape: ({batch_size_4d}, {n_tokens_4d}, 4)")
    print(f"  Position range - t: [{positions_4d[..., 0].min()}, {positions_4d[..., 0].max()}]")
    print(f"  Position range - v: [{positions_4d[..., 1].min()}, {positions_4d[..., 1].max()}]")
    print(f"  Position range - h: [{positions_4d[..., 2].min()}, {positions_4d[..., 2].max()}]")
    print(f"  Position range - w: [{positions_4d[..., 3].min()}, {positions_4d[..., 3].max()}]")
    assert positions_4d.shape == (batch_size_4d, n_tokens_4d, 4), "Position shape mismatch!"
    print("  ✓ PositionGetter4D test passed!\n")
    
    # Test 2: RotaryPositionEmbedding4D
    print("Test 2: RotaryPositionEmbedding4D (with multi-view support)")
    rope_4d = RotaryPositionEmbedding4D(
        temporal_frequency=10.0,   # 12 frames (时间优先)
        view_frequency=10.0,       # 3 views
        spatial_frequency=100.0    # 16x16 grid
    ).to(device)
    
    # Create sample tokens
    tokens_4d = torch.randn(batch_size_4d, n_heads_4d, n_tokens_4d, feature_dim_4d, device=device)
    print(f"  Input tokens shape: {tokens_4d.shape}")
    
    # Apply 4D RoPE
    output_4d = rope_4d(tokens_4d, positions_4d)
    print(f"  Output shape: {output_4d.shape}")
    assert output_4d.shape == tokens_4d.shape, "Output shape mismatch!"
    print(f"  Input  stats - mean: {tokens_4d.mean():.4f}, std: {tokens_4d.std():.4f}")
    print(f"  Output stats - mean: {output_4d.mean():.4f}, std: {output_4d.std():.4f}")
    print("  ✓ RotaryPositionEmbedding4D test passed!\n")
    
    # Test 3: Gradient flow for 4D RoPE
    print("Test 3: Testing 4D RoPE gradient flow...")
    rope_4d_grad = RotaryPositionEmbedding4D().to(device)
    tokens_4d_grad = torch.randn(batch_size_4d, n_heads_4d, n_tokens_4d, feature_dim_4d, 
                                 device=device, requires_grad=True)
    output_4d_grad = rope_4d_grad(tokens_4d_grad, positions_4d)
    loss_4d = output_4d_grad.sum()
    loss_4d.backward()
    print(f"  Gradient exists: {tokens_4d_grad.grad is not None}")
    print(f"  Gradient mean: {tokens_4d_grad.grad.mean():.6f}")
    assert tokens_4d_grad.grad is not None, "Gradient not flowing!"
    print("  ✓ 4D RoPE gradient flow test passed!\n")
    
    print("="*60)
    print("All 4D RoPE tests passed successfully! ✓")
    print("4D RoPE supports (time, view, height, width) dimensions! 🎉")
    print("Dimension order: [t, v, h, w] - time is prioritized!")
    print("="*60)