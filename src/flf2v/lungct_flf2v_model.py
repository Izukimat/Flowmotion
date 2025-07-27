"""
LungCT FLF2V Model Implementation - Fixed version
Addresses loss dict pollution, VAE freezing, and loss key naming
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import from package namespace (Fix: proper imports)
from .lungct_vae import LungCTVAE
from .lungct_dit import LungCTDiT
from .lungct_flow_matching import FlowMatching


class LungCTFLF2V(nn.Module):
    """
    First-Last Frame to Video (FLF2V) model for lung CT sequences
    Combines VAE, DiT, and Flow Matching for temporal interpolation
    Fixed: proper loss handling and VAE freezing
    """
    
    def __init__(
        self,
        vae: LungCTVAE,
        dit: LungCTDiT,
        flow_matching: FlowMatching,
        freeze_vae_after: int = 5000
    ):
        super().__init__()
        self.vae = vae
        self.dit = dit
        self.flow_matching = flow_matching
        self.freeze_vae_after = freeze_vae_after
        self.training_step = 0
        self._vae_frozen = False  # Track if VAE is actually frozen
        
    def _freeze_vae(self):
        """Actually freeze VAE parameters"""
        if not self._vae_frozen:
            for param in self.vae.parameters():
                param.requires_grad = False
            self._vae_frozen = True
            self.vae._vae_frozen = True
            logging.info(f"VAE parameters frozen at step {self.training_step}")
    
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to latent space"""
        vae_output = self.vae.encode(frames)
        return vae_output["latent"]
    
    def decode_frames(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to frame space"""
        return self.vae.decode(latents)
    
    def forward(
        self,
        video: torch.Tensor,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for training
        
        Args:
            video: Full breathing sequence [B, C, T, H, W]
            return_dict: Whether to return loss dictionary
        
        Returns:
            Loss tensor or dictionary with separate loss and latent keys
        """
        batch_size, channels, num_frames, height, width = video.shape
        
        # Extract first and last frames for conditioning
        first_frame = video[:, :, 0:1]  # [B, C, 1, H, W]
        last_frame = video[:, :, -1:]   # [B, C, 1, H, W]
        
        # Encode to latent space
        video_latent = self.encode_frames(video)
        first_latent = self.encode_frames(first_frame)
        last_latent = self.encode_frames(last_frame)
        
        # Flow matching loss - Fix: ensure consistent loss key names
        flow_output = self.flow_matching(
            video_latent,
            first_latent,
            last_latent,
            dit_model=self.dit,
            return_dict=True
        )
        
        # Extract flow losses with consistent naming
        flow_losses = {}
        if 'velocity_loss' in flow_output:
            flow_losses['loss_velocity'] = flow_output['velocity_loss']
        if 'flf_loss' in flow_output:
            flow_losses['loss_flf'] = flow_output['flf_loss']
        # Add any other flow losses with loss_ prefix
        for key, value in flow_output.items():
            if key.endswith('_loss') and key not in ['velocity_loss', 'flf_loss']:
                flow_losses[f'loss_{key[:-5]}'] = value
        
        # VAE losses - Fix: only compute if VAE not frozen
        vae_losses = {}
        if self.training_step < self.freeze_vae_after and not self._vae_frozen:
            vae_output = self.vae(video, return_dict=True)
            if 'loss_recon' in vae_output:
                vae_losses['loss_vae_recon'] = vae_output['loss_recon']
            if 'loss_kl' in vae_output:
                vae_losses['loss_vae_kl'] = vae_output['loss_kl']
            if 'loss_temporal' in vae_output:
                vae_losses['loss_vae_temporal'] = vae_output['loss_temporal']
        elif self.training_step >= self.freeze_vae_after and not self._vae_frozen:
            # Freeze VAE parameters
            self._freeze_vae()
        
        self.training_step += 1
        
        # Combine all losses
        all_losses = {**flow_losses, **vae_losses}
        
        if return_dict:
            # Fix: separate losses from latents to prevent pollution
            result = {
                # Loss keys (safe for summation)
                **all_losses,
                # Latent keys (separate from losses)
                'latents': {
                    'video_latent': video_latent,
                    'first_latent': first_latent,
                    'last_latent': last_latent
                }
            }
            return result
        else:
            # Return sum of only loss values
            total_loss = sum(loss for loss in all_losses.values() if isinstance(loss, torch.Tensor))
            return total_loss
    
    @torch.no_grad()
    def generate(
        self,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        num_frames: int = 82,
        guidance_scale: float = 1.0,
        decode: bool = True
    ) -> torch.Tensor:
        """
        Generate video between first and last frames
        
        Args:
            first_frame: First frame [B, C, H, W]
            last_frame: Last frame [B, C, H, W]
            num_frames: Total number of frames to generate
            guidance_scale: CFG scale
            decode: Whether to decode to pixel space
        
        Returns:
            Generated video [B, C, T, H, W] or latents
        """
        # Add temporal dimension
        first_frame = first_frame.unsqueeze(2)  # [B, C, 1, H, W]
        last_frame = last_frame.unsqueeze(2)
        
        # Encode to latent
        first_latent = self.encode_frames(first_frame)
        last_latent = self.encode_frames(last_frame)
        
        # Generate latent video
        latent_video = self.flow_matching.sample(
            first_latent,
            last_latent,
            dit_model=self.dit,
            num_frames=num_frames,
            guidance_scale=guidance_scale
        )
        
        if decode:
            # Decode to pixel space
            video = self.decode_frames(latent_video)
            return video
        else:
            return latent_video

