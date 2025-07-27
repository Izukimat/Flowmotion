"""
Lung CT DiT Module - Scaled-down DiT for FLF2V
Based on Wan2.1 architecture but adapted for medical imaging
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # ------------- reshape & pad -----------------
        x = x.view(B, T, H, W, C)            # -> grid
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
        x = x.view(B,
                   Tp // wT, wT,
                   Hp // wH, wH,
                   Wp // wW, wW, C)
        x = x.permute(0,1,3,5,2,4,6,7).contiguous()
        x = x.view(-1, wT*wH*wW, C)

        # ------------- attention ---------------------
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(t.shape[0], t.shape[1], self.num_heads, self.head_dim
                          ).transpose(1, 2) for t in qkv]  # [B*nW, H, S, Dh]

        attn  = (q @ k.transpose(-2, -1)) * self.scale
        attn  = attn.softmax(dim=-1)
        x_out = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1],
                                                   self.num_heads * self.head_dim)
        x_out = self.proj(x_out)

        # ------------- reverse windows ---------------
        x_out = x_out.view(B,
                           Tp // wT, Hp // wH, Wp // wW,
                           wT, wH, wW, C)
        x_out = x_out.permute(0,1,4,2,5,3,6,7).contiguous()
        x_out = x_out.view(B, Tp, Hp, Wp, C)[:, :T, :H, :W]  # drop pad
        return x_out.view(B, N, C)        # flatten back

# -------------------------------------------------------------------

class PatchMerging3D(nn.Module):
    """
    Down-samples (T,H,W) by (dT,dH,dW) with linear projection.
    """
    def __init__(self, dim, down=(1,2,2)):
        super().__init__()
        self.down = down
        self.reduction = nn.Linear(dim * np.prod(down), dim)
        self.norm = nn.LayerNorm(dim * np.prod(down))

    def forward(self, x, shape):
        B, N, C = x.shape
        T, H, W = shape
        dT, dH, dW = self.down
        Tn, Hn, Wn = T//dT, H//dH, W//dW

        x = x.view(B, T, H, W, C)[:, :Tn*dT, :Hn*dH, :Wn*dW]    # crop
        x = x.view(B, Tn, dT, Hn, dH, Wn, dW, C)
        x = x.permute(0,1,3,5,2,4,6,7).contiguous()
        x = x.view(B, Tn*Hn*Wn, -1)                 # concat neighbours
        x = self.reduction(self.norm(x))
        return x, (Tn, Hn, Wn)


class RoPE3D(nn.Module):
    """
    3D Rotary Position Embeddings for spatiotemporal data
    Adapted for medical volume sequences
    """
    
    def __init__(self, dim: int, max_seq_len: int = 10000):
        super().__init__()
        self.dim = dim
        
        # Compute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for efficiency
        self.max_seq_len = max_seq_len
        self._build_cache()
    
    def _build_cache(self):
        with torch.no_grad():                 # avoid autograd tracking
            positions = torch.arange(
                self.max_seq_len, dtype=torch.float, device=self.inv_freq.device
            )
            freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
            cos = freqs.cos().repeat_interleave(2, dim=-1)
            sin = freqs.sin().repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)
    
    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """Apply RoPE to input tensor"""
        seq_len = x.shape[seq_dim]
        
        # Rebuild cache if the current one is too small
        if seq_len > self.max_seq_len:
            self.max_seq_len = int(seq_len)          # <- update the limit
            self._build_cache()                      #    rebuild cos/sin
        
        cos = self.cos_cache[:seq_len]
        sin = self.sin_cache[:seq_len]
        
        return self.apply_rotary_emb(x, cos, sin, seq_dim)
    
    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """Apply rotation to input embeddings"""
        # Reshape for rotation
        x_rot = x.reshape(*x.shape[:-1], -1, 2)
        x_rot = torch.stack([-x_rot[..., 1], x_rot[..., 0]], dim=-1)
        x_rot = x_rot.reshape(x.shape)
        
        # Apply rotation
        if seq_dim == 1:
            cos = cos[:, None, :]
            sin = sin[:, None, :]
        
        return x * cos + x_rot * sin


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Norm with Zero init (from DiT)
    Modulates with 6 parameters: 2 for norm, 4 for attention/FFN gates
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        # LayerNorm parameters
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
        # Modulation network - outputs 6 params per block
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
        # Initialize to zero for residual behavior
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [B, N, C]
            c: Conditioning features [B, C]
        Returns:
            norm_x: Normalized input
            gates: Modulation parameters (shift, scale, gate_attn, gate_ffn)
        """
        # Get modulation parameters
        mod_params = self.adaLN_modulation(c)
        shift_mha, scale_mha, gate_mha, shift_ffn, scale_ffn, gate_ffn = mod_params.chunk(6, dim=-1)
        
        # Normalize
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm * self.gamma + self.beta
        
        # Apply modulation for attention
        x_mod = x_norm * (1 + scale_mha.unsqueeze(1)) + shift_mha.unsqueeze(1)
        
        return x_mod, (shift_ffn, scale_ffn, gate_mha, gate_ffn)


class DiTBlock(nn.Module):
    """
    DiT block with self-attention and FFN
    Specialized for FLF2V with frozen frame conditioning
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
    
    def forward(
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
        B, N, C = x.shape
        
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


class FLF2VConditioning(nn.Module):
    """
    First-Last-Frame conditioning for video generation
    Handles frozen frame injection and binary masks
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        
        # Project concatenated first+last frames
        self.frame_proj = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Positional embeddings for first/last distinction
        self.pos_emb = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        
        # Binary mask embedding
        self.mask_emb = nn.Embedding(2, hidden_dim)
    
    def forward(
        self, 
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        frozen_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            first_frame: First frame latent [B, C, 1, H, W]
            last_frame: Last frame latent [B, C, 1, H, W]   
            frozen_mask: Binary mask [B, T] indicating frozen frames
        Returns:
            Conditioning features [B, N, C]
        """
        B = first_frame.shape[0]
        
        # Flatten spatial dimensions
        first_flat = rearrange(first_frame, 'b c t h w -> b (t h w) c')  # [B, 1*H*W, C]
        last_flat = rearrange(last_frame, 'b c t h w -> b (t h w) c')    # [B, 1*H*W, C]
        
        # Concatenate and project
        frames_concat = torch.cat([first_flat, last_flat], dim=-1) # [B, H*W, 2*C]
        frames_feat = self.frame_proj(frames_concat)
        
        # Add positional embeddings
        pos_emb = repeat(
            self.pos_emb[:, 0],      # [1, C]
            '1 c -> b n c',          # broadcast to every token
            b=B,
            n=frames_feat.shape[1]
        )

        frames_feat = frames_feat + pos_emb          # shape  [B, N, C]
                
        # Add mask embeddings
        mask_feat = self.mask_emb(frozen_mask.long())
        
        return frames_feat, mask_feat


class LungCTDiT(nn.Module):
    """
    Lung CT Diffusion Transformer for FLF2V
    Scaled down to ~0.6-1B parameters
    """
    
    def __init__(
        self,
        latent_channels: int = 8,
        latent_size: Tuple[int, int, int] = (5, 16, 16),  # After 8x compression of 40x128x128
        hidden_dim: int = 768,  # Reduced from typical 1024-1280
        depth: int = 24,  # 24 layers -> ~0.6B params
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.hidden_dim = hidden_dim
        self.num_tokens = np.prod(latent_size)
        self.patch_merge = PatchMerging3D(hidden_dim, down=(1,2,2))
        
        # Patchify projection (1x1x1 conv)
        self.patchify = nn.Conv3d(
            latent_channels, hidden_dim,
            kernel_size=1, stride=1, padding=0
        )
        
        # FLF2V conditioning
        self.flf_conditioning = FLF2VConditioning(
            latent_channels,
            hidden_dim
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
        B, C, D, H, W = x_t.shape
        
        # Patchify
        x = self.patchify(x_t)  # [B, hidden_dim, D, H, W]
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        spatial_shape = (D, H//2, W//2)              # after merge

        # ---- patch merge ----
        x, spatial_shape = self.patch_merge(x, (D, H, W))

        # Time embedding
        t_emb = self.time_embed(t)
        
        # FLF2V conditioning
        if frozen_mask is None:
            # Default mask: first and last frames are frozen
            frozen_mask = torch.zeros(B, D, device=x.device)
            frozen_mask[:, 0] = 1
            frozen_mask[:, -1] = 1
        
        flf_cond, mask_cond = self.flf_conditioning(first_frame, last_frame, frozen_mask)
        
        # Combine conditioning
        # Option 1: Add FLF conditioning as extra tokens
        x = torch.cat([flf_cond, x], dim=1)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, spatial_shape=spatial_shape)
        
        # Remove conditioning tokens
        x = x[:, flf_cond.shape[1]:]
        
        # Output projection
        x = self.norm_out(x)
        x = self.proj_out(x)
        
        # Reshape back
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        
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
    return LungCTDiT(
        latent_channels=config.get('latent_channels', 8),
        latent_size=config.get('latent_size', (5, 16, 16)),
        hidden_dim=config.get('hidden_dim', 768),
        depth=config.get('depth', 24),
        num_heads=config.get('num_heads', 12),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        dropout=config.get('dropout', 0.1),
    )