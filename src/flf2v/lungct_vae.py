"""
Lung CT VAE Module - Adapted from MedVAE for FLF2V
Extends MedVAE 3D with increased channels and temporal consistency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from medvae.utils.factory import create_model


class LungCTVAE(nn.Module):
    """
    VAE for lung CT specialized for FLF2V task
    Based on MedVAE 3D but with modifications:
    - Increased latent channels (8-16 instead of 1)
    - Temporal consistency regularization
    - Tanh output scaling for zero-mean unit-var latents
    
    IMPORTANT: Channel expansion is applied only to the sampled z, not mu/logvar.
    The KL divergence is computed on the original 1-channel distribution from MedVAE.
    This preserves the probabilistic interpretation while allowing multi-channel latents
    for the flow matching model.
    """
    
    def __init__(
        self,
        base_model_name: str = "medvae_8x1_2d",   # 8× down per axis (x64 total)
        latent_channels:  int  = 8,
        temporal_weight:  float= 0.1,
        use_tanh_scaling: bool  = True,
        freeze_pretrained: bool = False,
    ):
        super().__init__()
        self.latent_channels   = latent_channels
        self.temporal_weight   = temporal_weight
        self.use_tanh_scaling  = use_tanh_scaling
        self.compression_factor= 4                # 8× in H and W (no temporal downsampling)
        self._vae_frozen = False
        # ───────────────────────────────── MedVAE backbone (2-D) ────────────
        self.base_vae = create_model(base_model_name, training=True, state_dict=True)

        # optional 1→N channel expansion
        if latent_channels != 1:
            self.channel_exp = nn.Conv2d(1, latent_channels, 1)
            self.channel_red = nn.Conv2d(latent_channels, 1, 1)
            nn.init.xavier_uniform_(self.channel_exp.weight);   nn.init.zeros_(self.channel_exp.bias)
            nn.init.xavier_uniform_(self.channel_red.weight);   nn.init.zeros_(self.channel_red.bias)

        if use_tanh_scaling:
            self.latent_scale = nn.Parameter(torch.ones(1))
            self.latent_shift = nn.Parameter(torch.zeros(1))

        if freeze_pretrained:
            for p in self.base_vae.parameters(): p.requires_grad = False

    # ───────────────────────────── helper: encode one 2-D frame ─────────────
    def _encode_2d(self, x2d: torch.Tensor):
        post = self.base_vae.encode(x2d)
        mu, logvar = post.mean, post.logvar
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, mu, logvar


    def _encode_2d_chunked(self, x2d: torch.Tensor, chunk_size: int = 8):
        """Process frames in chunks to reduce memory usage"""
        B_T = x2d.shape[0]
        chunks = []
        
        # Process in chunks with no_grad to save memory during encoding
        with torch.no_grad():
            for i in range(0, B_T, chunk_size):
                chunk = x2d[i:i+chunk_size]
                post = self.base_vae.encode(chunk)
                chunks.append((post.mean.detach(), post.logvar.detach()))
        
        # Concatenate results
        mu = torch.cat([c[0] for c in chunks], dim=0)
        logvar = torch.cat([c[1] for c in chunks], dim=0)
        
        # Only compute gradients for sampling if needed
        if self.training and not self._vae_frozen:
            mu = mu.requires_grad_(True)
            logvar = logvar.requires_grad_(True)
        
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, mu, logvar
    # ───────────────────────────────────────── public API ───────────────────
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode with chunked processing"""
        B, _, T, H, W = x.shape
        x2d = x.permute(0,2,1,3,4).reshape(B*T, 1, H, W)
        
        # Use chunked encoding for large inputs
        if H * W > 256 * 256:  # Threshold for chunking
            z, mu, logvar = self._encode_2d_chunked(x2d, chunk_size=8)
        else:
            z, mu, logvar = self._encode_2d(x2d)

        # Keep original 1-channel z for VAE losses (ADDITION for memory fix)
        z1_original = z.clone() 
        # Rest of the processing remains the same...
        if self.latent_channels != 1:
            z = self.channel_exp(z)
        
        if self.use_tanh_scaling:
            z = torch.tanh(z * self.latent_scale + self.latent_shift)
        
        h, w = z.shape[-2:]
        z = z.view(B, T, self.latent_channels, h, w).permute(0,2,1,3,4)
        mu = mu.view(B, T, 1, h, w).permute(0,2,1,3,4)
        logvar = logvar.view(B, T, 1, h, w).permute(0,2,1,3,4)
        
        z1_original = z1_original.view(B, T, 1, h, w).permute(0,2,1,3,4)

        return {"latent": z, "mu": mu, "logvar": logvar, "z1": z1_original}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B, C, T, h, w = z.shape
        if self.use_tanh_scaling:
            z = torch.atanh(torch.clamp(z, -0.999, 0.999))
            z = (z - self.latent_shift) / self.latent_scale

        z2d = z.permute(0,2,1,3,4).reshape(B*T, C, h, w)
        if self.latent_channels != 1:
            z2d = self.channel_red(z2d)

        recon2d = self.base_vae.decode(z2d)                      # (B·T,1,H,W)
        H, W = recon2d.shape[-2:]
        recon = recon2d.view(B, T, 1, H, W).permute(0,2,1,3,4)   # (B,1,T,H,W)
        return recon

    def decode_z1(self, z1: torch.Tensor) -> torch.Tensor:
        """Decode from 1-channel latent (for VAE loss computation)"""
        B, C, T, h, w = z1.shape  # C should be 1
        assert C == 1, f"Expected 1-channel input, got {C}"
        
        if self.use_tanh_scaling:
            z1 = torch.atanh(torch.clamp(z1, -0.999, 0.999))
            z1 = (z1 - self.latent_shift) / self.latent_scale
        
        # Reshape to 4D for base VAE
        z2d = z1.permute(0,2,1,3,4).reshape(B*T, 1, h, w)  # [B*T, 1, h, w]
        
        # Direct decode (no channel reduction needed)
        recon2d = self.base_vae.decode(z2d)                      # (B·T,1,H,W)
        H, W = recon2d.shape[-2:]
        recon = recon2d.view(B, T, 1, H, W).permute(0,2,1,3,4)   # (B,1,T,H,W)
        return recon

    def forward(
        self, 
        x: torch.Tensor, 
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with optional temporal loss
        """
        # Encode
        encoded = self.encode(x)
        z = encoded['latent']
        
        # Decode
        x_recon = self.decode(z)
        
        # Compute losses
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence - use original 1-channel mu/logvar
        mu, logvar = encoded['mu'], encoded['logvar']
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Temporal consistency loss (for video sequences)
        temporal_loss = torch.tensor(0.0, device=x.device)
        if x.size(2) > 1:  # If depth > 1, compute temporal consistency
            # Use the multi-channel latent for temporal consistency
            z_diff = z[:, :, 1:] - z[:, :, :-1]
            temporal_loss = torch.mean(z_diff.pow(2))
        
        if return_dict:
            return {
                'reconstruction': x_recon,
                'latent': z,
                'mu': mu,  # Original 1-channel
                'logvar': logvar,  # Original 1-channel
                'loss_recon': recon_loss,
                'loss_kl': kl_loss,
                'loss_temporal': temporal_loss,
                'loss_total': recon_loss + 0.01 * kl_loss + self.temporal_weight * temporal_loss
            }
        else:
            return x_recon
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent features for downstream DiT
        """
        with torch.no_grad():
            encoded = self.encode(x)
            return encoded['latent']
    
    @property
    def latent_shape_factor(self) -> int:
        """Compression factor per dimension"""
        return self.compression_factor


class VAELoss(nn.Module):
    """
    Loss function for VAE training
    Includes reconstruction, KL, temporal, and perceptual losses
    """
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 0.01,
        temporal_weight: float = 0.1,
        perceptual_weight: float = 0.1,
        edge_weight: float = 0.05
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.temporal_weight = temporal_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        
        # Sobel filters for edge detection
        self.register_buffer('sobel_x', self._get_sobel_kernel_3d('x'))
        self.register_buffer('sobel_y', self._get_sobel_kernel_3d('y'))
        self.register_buffer('sobel_z', self._get_sobel_kernel_3d('z'))
    
    def _get_sobel_kernel_3d(self, direction: str) -> torch.Tensor:
        """Get 3D Sobel kernel for edge detection"""
        if direction == 'x':
            kernel = torch.tensor([
                [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                 [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                 [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
            ], dtype=torch.float32)
        elif direction == 'y':
            kernel = torch.tensor([
                [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                 [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                 [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]
            ], dtype=torch.float32)
        else:  # z
            kernel = torch.tensor([
                [[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                 [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                 [[1, 2, 1], [2, 4, 2], [1, 2, 1]]]
            ], dtype=torch.float32)
        
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def edge_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Compute edge preservation loss using 3D Sobel filters"""
        # Compute gradients
        grad_x = F.conv3d(x, self.sobel_x, padding=1)
        grad_y = F.conv3d(x, self.sobel_y, padding=1)
        grad_z = F.conv3d(x, self.sobel_z, padding=1)
        
        grad_x_recon = F.conv3d(x_recon, self.sobel_x, padding=1)
        grad_y_recon = F.conv3d(x_recon, self.sobel_y, padding=1)
        grad_z_recon = F.conv3d(x_recon, self.sobel_z, padding=1)
        
        # Edge magnitude
        edge_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-6)
        edge_mag_recon = torch.sqrt(grad_x_recon**2 + grad_y_recon**2 + grad_z_recon**2 + 1e-6)
        
        return F.mse_loss(edge_mag_recon, edge_mag)
    
    def forward(self, model_output: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute total loss
        """
        losses = {}
        
        # Get individual losses from model output
        losses['recon'] = model_output['loss_recon'] * self.recon_weight
        losses['kl'] = model_output['loss_kl'] * self.kl_weight
        losses['temporal'] = model_output['loss_temporal'] * self.temporal_weight
        
        # Edge preservation loss
        if self.edge_weight > 0:
            losses['edge'] = self.edge_loss(target, model_output['reconstruction']) * self.edge_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
def validate_dit_config_for_vae(vae_compression_factor: int, input_size: int, dit_config: Dict):
    """Validate DiT config matches VAE reality"""
    expected_latent_spatial = input_size // vae_compression_factor
    
    dit_spatial = dit_config['latent_size'][1]  # Assuming [T, H, W]
    
    if dit_spatial != expected_latent_spatial:
        print(f"⚠️  DiT config mismatch!")
        print(f"   VAE {vae_compression_factor}x compression on {input_size}² → {expected_latent_spatial}²")
        print(f"   DiT config expects {dit_spatial}²")
        print(f"   Updating DiT config...")
        
        dit_config['latent_size'][1] = expected_latent_spatial
        dit_config['latent_size'][2] = expected_latent_spatial
    
    return dit_config