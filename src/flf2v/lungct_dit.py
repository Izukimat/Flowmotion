"""
Lung CT DiT Module - Scaled-down DiT for FLF2V
Based on Wan2.1 architecture but adapted for medical imaging
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.compiler import cudagraph_mark_step_begin 

import numpy as np
import math
from typing import Optional, Tuple, List, Dict
from einops import rearrange, repeat


class WindowedSelfAttention(nn.Module):
    """
    Local (T,H,W) window attention for 3-D tokens.
    """
    def __init__(self, dim, num_heads=8, window=(8, 8, 8), head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.window    = window              # (T, H, W)
        self.scale     = head_dim ** -0.5

        self.qkv  = nn.Linear(dim, 3 * num_heads * head_dim, bias=False)
        self.proj = nn.Linear(num_heads * head_dim, dim)

    def forward(self, x, shape):
        """
        x : [B, N, C]  – flattened tokens
        shape : (T, H, W) before flattening
        """
        B, N, C = x.shape
        T, H, W = shape
        wT, wH, wW = self.window

        # 🔧 CRITICAL FIX: Validate tensor dimensions match spatial_shape
        expected_tokens = T * H * W
        if N != expected_tokens:
            
            # Calculate actual dimensions from tensor
            # Assume H and W are correct, solve for T
            actual_T = N // (H * W)
            T = actual_T
            shape = (T, H, W)

        # ------------- reshape & pad -----------------
        x = x.reshape(B, T, H, W, C)            # -> grid
        pad_t = (wT - T % wT) % wT
        pad_h = (wH - H % wH) % wH
        pad_w = (wW - W % wW) % wW
        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, 0,                    # channels
                          0, pad_w,                # width
                          0, pad_h,                # height
                          0, pad_t))               # depth/time
        Tp, Hp, Wp = x.shape[1:4]

        # partition windows -> [B * nW, win_size, C]
        x = x.reshape(B,
                   Tp // wT, wT,
                   Hp // wH, wH,
                   Wp // wW, wW, C)
        x = x.permute(0,1,3,5,2,4,6,7).contiguous()
        x = x.reshape(-1, wT*wH*wW, C)

        # ------------- attention ---------------------
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(t.shape[0], t.shape[1], self.num_heads, self.head_dim
                          ).transpose(1, 2) for t in qkv]  # [B*nW, H, S, Dh]

        attn  = (q @ k.transpose(-2, -1)) * self.scale
        attn  = attn.softmax(dim=-1)
        x_out = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1],
                                                   self.num_heads * self.head_dim)
        x_out = self.proj(x_out)

        # ------------- reverse windows ---------------
        x_out = x_out.reshape(B,
                           Tp // wT, Hp // wH, Wp // wW,
                           wT, wH, wW, C)
        x_out = x_out.permute(0,1,4,2,5,3,6,7).contiguous()
        x_out = x_out.reshape(B, Tp, Hp, Wp, C)[:, :T, :H, :W]  # drop pad
        return x_out.reshape(B, N, C)        # flatten back


# ORIGINAL ARCHITECTURE - Keep exact same structure as trained model
class PatchMerging3D(nn.Module):
    """
    Down-samples (T,H,W) by (dT,dH,dW) with linear projection.
    ORIGINAL ARCHITECTURE: Uses 'reduction' and 'norm' (matches checkpoint)
    """
    def __init__(self, hidden_dim: int, down: Tuple[int, int, int] = (1, 2, 2)):
        super().__init__()
        self.down = down
        dT, dH, dW = down
        
        # ORIGINAL NAMES - Match checkpoint exactly
        self.reduction = nn.Linear(hidden_dim * dT * dH * dW, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, spatial_shape: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Args:
            x: [B, N, C] where N = T*H*W
            spatial_shape: (T, H, W)
        Returns:
            x_merged: [B, N', C] where N' = T'*H'*W'
            new_shape: (T', H', W')
        """
        B, N, C = x.shape
        T, H, W = spatial_shape
        dT, dH, dW = self.down
        
        # 🔧 FIXED: Adaptive dimension validation
        expected_tokens = T * H * W
        if N != expected_tokens:
            # Adapt T to match actual tokens
            actual_T = N // (H * W)
            T = actual_T
        
        # Reshape to grid
        x = x.reshape(B, T, H, W, C)
        
        # Pad if needed
        pad_t = (dT - T % dT) % dT
        pad_h = (dH - H % dH) % dH
        pad_w = (dW - W % dW) % dW
        
        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
        
        T_pad, H_pad, W_pad = x.shape[1:4]
        
        # Merge patches
        x = x.reshape(B,
                   T_pad // dT, dT,
                   H_pad // dH, dH,
                   W_pad // dW, dW, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x = x.reshape(B, (T_pad // dT) * (H_pad // dH) * (W_pad // dW), dT * dH * dW * C)
        
        # Project - Use ORIGINAL layer names
        x = self.reduction(x)
        x = self.norm(x)
        
        new_shape = (T_pad // dT, H_pad // dH, W_pad // dW)
        return x, new_shape


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with zero initialization
    ORIGINAL ARCHITECTURE: Uses 'adaLN_modulation' and 'norm' (matches checkpoint)
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        
        # ORIGINAL NAMES - Match checkpoint exactly
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
        # Zero init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        # Normalize
        x_norm = self.norm(x)
        
        # Condition - Use ORIGINAL layer name
        cond = self.adaLN_modulation(c)  # [B, 6*dim]
        shift_mha, scale_mha, gate_mha, shift_ffn, scale_ffn, gate_ffn = cond.chunk(6, dim=-1)
        
        # Apply modulation
        x_mha = x_norm * (1 + scale_mha.unsqueeze(1)) + shift_mha.unsqueeze(1)
        
        return x_mha, (shift_ffn, scale_ffn, gate_mha, gate_ffn)


class DiTBlock(nn.Module):
    """
    DiT block with self-attention and FFN
    ORIGINAL ARCHITECTURE: Matches trained checkpoint exactly
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Self-attention
        self.attn = WindowedSelfAttention(
            dim, num_heads=num_heads,
            window=(8, 8, 8), head_dim=self.head_dim
        )
        
        # FFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim, bias=False),
            nn.Dropout(dropout)
        )
        
        # Normalization and modulation
        self.norm1 = AdaLNZero(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
    
    def _forward_block(
        self, 
        x: torch.Tensor, 
        c: torch.Tensor,
        spatial_shape: Tuple[int, int, int]
        ) -> torch.Tensor:
        """
        Args:
            x: Input features [B, N, C]
            c: Conditioning (timestep + any other) [B, C]
            mask: Attention mask [B, N, N]
        """
        if len(x.shape) == 3:
            B, N, C = x.shape
        elif len(x.shape) == 4:
            B, S, N, C = x.shape
            x = x.reshape(B, S * N, C)  # Flatten to [B, N, C]
            N = S * N
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}. Expected [B, N, C] or [B, S, N, C]")
        
            
        # Pre-norm and modulation
        x_norm, gates = self.norm1(x, c)
        shift_ffn, scale_ffn, gate_attn, gate_ffn = gates
        
        # local window attention
        x_attn = self.attn(x_norm, spatial_shape)   # [B, N, C]
        
        # Gated residual for attention
        x = x + gate_attn.unsqueeze(1) * x_attn
        
        # FFN with gating
        x_norm2 = self.norm2(x)
        x_ffn = self.mlp(x_norm2 * (1 + scale_ffn.unsqueeze(1)) + shift_ffn.unsqueeze(1))
        x = x + gate_ffn.unsqueeze(1) * x_ffn
        
        return x
    
    def forward(self, x, c, spatial_shape):
        """
        Run the block with gradient-checkpointing.
        We clone **after** the checkpoint so the cloned tensor
        is outside the CUDA graph replay buffer used by torch.compile.
        """
        
        if self.training:
            x = checkpoint(self._forward_block, x, c, spatial_shape)
        else:
            x = self._forward_block(x, c, spatial_shape)

        return x

class LungCTDiT(nn.Module):
    """
    Lung CT Diffusion Transformer for FLF2V
    Scaled down to ~0.6-1B parameters
    """
    
    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: Tuple[int, int, int] = (21, 32, 32),  # After 8x compression of 40x128x128
        hidden_dim: int = 384,  # Reduced from typical 1024-1280
        depth: int = 12,  # 24 layers -> ~0.6B params
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
        use_rope: bool = True,
        gradient_checkpointing: bool = True
    ):
        super().__init__()

        # Validate latent_size matches compression reality
        T, H, W = latent_size
        min_size, max_size = 16, 256
        assert min_size <= H <= max_size, f"Expected H in [{min_size},{max_size}] for 4x VAE, got {H}"
                
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.hidden_dim = hidden_dim
        self.num_tokens = np.prod(latent_size)
        self.patch_merge = PatchMerging3D(hidden_dim, down=(1,2,2))
        _, self.H, self.W = latent_size 

        # Patchify projection (1x1x1 conv)
        self.patchify = nn.Conv3d(
            latent_channels, hidden_dim,
            kernel_size=1, stride=1, padding=0
        )
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        
        # Output projection
        self.norm_out = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.proj_out = nn.Linear(hidden_dim, latent_channels)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with scaled init"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.apply(_basic_init)
        
        # Zero-init output projection
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def patchify_and_merge(self, x_latent):
        """
        latent [B,C,1,H,W] → tokens [B, H'·W', C] after 1× merge
        """
        x = self.patchify(x_latent)                       # [B,C,1,H',W']
        B, _, T, H, W = x.shape                          # infer H', W'
        x = rearrange(x, 'b c t h w -> b (t h w) c')      # flatten
        x, _ = self.patch_merge(x, (T, H, W))             # merge with correct shape
        return x
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        frozen_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for flow matching
        
        Args:
            x_t: Noisy latent at timestep t [B, C, D, H, W]
            t: Timestep [B]
            first_frame: First frame latent [B, C, D, H, W]
            last_frame: Last frame latent [B, C, D, H, W]
            frozen_mask: Binary mask for frozen frames [B, T]
        
        Returns:
            Predicted velocity [B, C, D, H, W]
        """
        cudagraph_mark_step_begin()
        B, C, D, H, W = x_t.shape
    
        
        # Patchify
        x = self.patchify(x_t)  # [B, hidden_dim, D, H, W]
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        
        # 🔧 CRITICAL FIX: Calculate spatial_shape from ACTUAL tensor dimensions
        actual_tokens = x.shape[1]  # N = D * H * W after patchify
        actual_H, actual_W = H, W  # Patchify doesn't change spatial dims
        actual_D = actual_tokens // (actual_H * actual_W)  # Solve for D
        
        initial_spatial_shape = (actual_D, actual_H, actual_W)
        
        # ---- patch merge ---- (ORIGINAL IMPLEMENTATION)
        x, spatial_shape = self.patch_merge(x, initial_spatial_shape)
        
        # Time embedding
        t_emb = self.time_embed(t)
        
        # FLF2V conditioning
        if frozen_mask is None:
            frozen_mask = torch.zeros(B, actual_D, device=x.device)
            frozen_mask[:, 0] = 1
            frozen_mask[:, -1] = 1
        
        first_tok  = self.patchify_and_merge(first_frame)
        last_tok   = self.patchify_and_merge(last_frame)
        flf_cond   = torch.cat([first_tok, last_tok], dim=1)   # [B, 2·H′·W′, C]       
        
        # Combine conditioning
        x = torch.cat([flf_cond, x], dim=1)
        
        # 🔧 FIXED: Update spatial_shape for conditioning tokens
        T_merged, H_merged, W_merged = spatial_shape
        conditioning_spatial_shape = (T_merged + 2, H_merged, W_merged)
                
        # Apply transformer blocks - ORIGINAL ARCHITECTURE
        for i, block in enumerate(self.blocks):
            x = block(x, t_emb, spatial_shape=conditioning_spatial_shape)
        
        # Remove conditioning tokens
        x = x[:, flf_cond.shape[1]:]
        
        # Output projection
        x = self.norm_out(x)
        x = self.proj_out(x)
        
        # Reshape back
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=T_merged, h=H_merged, w=W_merged)
        
        # 🔧 CRITICAL: Upsample back to original spatial dimensions if needed
        if x.shape[-2:] != (H, W):
            B_x, C_x, D_x = x.shape[:3]
            x = x.reshape(B_x * D_x, C_x, H_merged, W_merged)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            x = x.reshape(B_x, C_x, D_x, H, W)
        
        return x


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


def create_dit_model(config: Dict) -> LungCTDiT:
    """Factory function to create DiT model from config"""
    default_config = {
        'latent_channels': 4,           # ← REDUCED from 8 to 4
        'latent_size': [21, 32, 32],    # ← CORRECTED from [21, 16, 16] to [21, 32, 32]
        'hidden_dim': 384,              # ← REDUCED from 512+ to 384 for memory
        'depth': 12,                    # ← REDUCED from 24 to 12 for memory
        'num_heads': 6,                 # ← REDUCED accordingly
        'mlp_ratio': 4.0,
        'dropout': 0.05,
        'use_rope': True,
        'gradient_checkpointing': True  # ← KEEP enabled for memory savings
    }
    
    # Merge with user config
    final_config = {**default_config, **config}
    
    return LungCTDiT(**final_config)