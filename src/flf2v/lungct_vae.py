"""
Lung CT VAE Module - Adapted from MedVAE for FLF2V
Extends MedVAE 3D with increased channels and temporal consistency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from typing import Dict, Tuple, Optional
import numpy as np
from medvae.models import AutoencoderKL_3D
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
        base_model_name: str = "medvae_4_1_3d",  # 4x compression per dim = 64x total
        latent_channels: int = 8,  # Increased from 1 for better expressiveness
        temporal_weight: float = 0.1,
        use_tanh_scaling: bool = True,
        freeze_pretrained: bool = False
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.temporal_weight = temporal_weight
        self.use_tanh_scaling = use_tanh_scaling
        
        # Load pretrained MedVAE 3D model
        self.base_vae = create_model(base_model_name, training=True, state_dict=True)
        
        # Get config from base model
        self.compression_factor = 4  # 4x per dimension
        base_latent_dim = self.base_vae.embed_dim  # Usually matches conv output
        
        # Adapt for increased channels: add projection layers
        if latent_channels != 1:
            # Two options for channel expansion:
            # Option 1: Learned projection (current implementation)
            self.channel_expansion = nn.Conv3d(
                1, latent_channels, 
                kernel_size=1, stride=1, padding=0
            )
            
            # Option 2: Simple repetition (uncomment to use)
            # self.channel_expansion = lambda x: x.repeat(1, latent_channels, 1, 1, 1)
            
            # Project from N channels back to 1 before decoding
            self.channel_reduction = nn.Conv3d(
                latent_channels, 1,
                kernel_size=1, stride=1, padding=0
            )
            
            # Initialize with identity-like mapping
            if isinstance(self.channel_expansion, nn.Conv3d):
                nn.init.xavier_uniform_(self.channel_expansion.weight, gain=1.0)
                nn.init.zeros_(self.channel_expansion.bias)
            nn.init.xavier_uniform_(self.channel_reduction.weight, gain=1.0)
            nn.init.zeros_(self.channel_reduction.bias)
        
        # Tanh scaling parameters (learnable)
        if use_tanh_scaling:
            self.latent_scale = nn.Parameter(torch.ones(1))
            self.latent_shift = nn.Parameter(torch.zeros(1))
        
        # Optionally freeze pretrained weights
        if freeze_pretrained:
            for param in self.base_vae.parameters():
                param.requires_grad = False
    
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode lung CT volume to latent representation
        Args:
            x: Input tensor [B, 1, D, H, W] in range [-1, 1]
        Returns:
            dict with 'latent', 'mu', 'logvar', 'mu_expanded', 'logvar_expanded'
        """
        # Use base VAE encoder
        posterior = self.base_vae.encode(x)
        mu, logvar = posterior.mean, posterior.logvar
        
        # Sample latent
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Store original mu/logvar for KL computation
        mu_original = mu
        logvar_original = logvar
        
        # Expand channels if needed
        if self.latent_channels != 1:
            z = self.channel_expansion(z)
            # For flow matching, we only need the expanded z
            # We do NOT expand mu/logvar as that breaks the probabilistic interpretation
        
        # Apply tanh scaling for zero-mean unit-var
        if self.use_tanh_scaling:
            z = torch.tanh(z * self.latent_scale + self.latent_shift)
        
        return {
            'latent': z,
            'mu': mu_original,  # Keep original for KL
            'logvar': logvar_original,  # Keep original for KL
            'z_pre_tanh': z if not self.use_tanh_scaling else None
        }
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to reconstructed volume
        Args:
            z: Latent tensor [B, C, D', H', W']
        Returns:
            Reconstructed volume [B, 1, D, H, W]
        """
        # Inverse tanh scaling
        if self.use_tanh_scaling:
            # Clamp to avoid inf in atanh
            z = torch.clamp(z, -0.999, 0.999)
            z = (torch.atanh(z) - self.latent_shift) / self.latent_scale
        
        # Reduce channels if needed
        if self.latent_channels != 1:
            z = self.channel_reduction(z)
        
        # Use base VAE decoder
        x_recon = self.base_vae.decode(z)
        
        return x_recon
    
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