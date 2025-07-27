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
        freeze_vae_after: int = 5000,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.vae = vae
        self.dit = dit
        self.flow_matching = flow_matching
        self.freeze_vae_after = freeze_vae_after
        self.training_step = 0
        self._vae_frozen = False  # Track if VAE is actually frozen
        
        # Configurable loss weights
        self.loss_weights = loss_weights or {
            'velocity_weight': 1.0,
            'flf_weight': 0.1,
            'vae_recon_weight': 1.0,
            'vae_kl_weight': 0.01,
            'vae_temporal_weight': 0.1
        }
        
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
        
        # Extract flow losses with consistent naming (Improvement 1)
        flow_losses = {
            f'loss_{k.replace("_loss", "")}': v 
            for k, v in flow_output.items() 
            if k.endswith('_loss') and isinstance(v, torch.Tensor)
        }
        
        # VAE losses - Fix: only compute if VAE not frozen
        vae_losses = {}
        if self.training_step < self.freeze_vae_after and not self._vae_frozen:
            vae_output = self.vae(video, return_dict=True)
            
            # Map VAE loss keys with fallbacks (Improvement 2)
            vae_loss_mapping = {
                'loss_recon': 'loss_vae_recon',
                'loss_kl': 'loss_vae_kl', 
                'loss_temporal': 'loss_vae_temporal',
                # Add fallback mappings for different VAE implementations
                'recon_loss': 'loss_vae_recon',
                'kl_loss': 'loss_vae_kl',
                'temporal_loss': 'loss_vae_temporal'
            }
            
            for vae_key, mapped_key in vae_loss_mapping.items():
                if vae_key in vae_output and isinstance(vae_output[vae_key], torch.Tensor):
                    vae_losses[mapped_key] = vae_output[vae_key]
                    
        elif self.training_step >= self.freeze_vae_after and not self._vae_frozen:
            # Freeze VAE parameters
            self._freeze_vae()
        
        # Only increment training step during training (Improvement 4)
        if self.training:
            self.training_step += 1
        
        # Combine all losses
        all_losses = {**flow_losses, **vae_losses}
        
        # Calculate weighted total loss
        total_loss = self._calculate_total_loss(all_losses)
        all_losses['loss_total'] = total_loss
        
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
            # Return total loss
            return total_loss

    def _calculate_total_loss(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate weighted total loss from individual loss components
        
        Args:
            loss_dict: Dictionary containing individual losses
            
        Returns:
            Weighted total loss tensor
        """
        if not loss_dict:
            # Return zero tensor on appropriate device
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        
        # Get device from first tensor in loss_dict
        device = next(iter(loss_dict.values())).device
        total_loss = torch.tensor(0.0, device=device)
        
        # Only sum keys that start with 'loss_'
        for key, value in loss_dict.items():
            if key.startswith('loss_'):
                # Improved: handle different value types (Improvement 6)
                if isinstance(value, torch.Tensor):
                    loss_tensor = value
                elif isinstance(value, (int, float)):
                    loss_tensor = torch.tensor(float(value), device=device)
                else:
                    continue  # Skip non-numeric values
                
                # Map loss key to weight key
                weight = self._get_loss_weight(key)
                total_loss = total_loss + weight * loss_tensor
        
        return total_loss
    
    def _get_loss_weight(self, loss_key: str) -> float:
        """Get the appropriate weight for a loss key"""
        weight_mapping = {
            'loss_velocity': 'velocity_weight',
            'loss_flf': 'flf_weight',
            'loss_vae_recon': 'vae_recon_weight',
            'loss_vae_kl': 'vae_kl_weight',
            'loss_vae_temporal': 'vae_temporal_weight'
        }
        
        weight_key = weight_mapping.get(loss_key)
        if weight_key and weight_key in self.loss_weights:
            return self.loss_weights[weight_key]
        else:
            return 1.0  # Default weight for unknown losses
    
    def state_dict(self, *args, **kwargs):
        """Include _vae_frozen state in checkpoint (Improvement 3)"""
        state = super().state_dict(*args, **kwargs)
        state['_vae_frozen'] = self._vae_frozen
        state['training_step'] = self.training_step
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Restore _vae_frozen state from checkpoint (Improvement 3)"""
        # Extract custom state before loading
        if '_vae_frozen' in state_dict:
            self._vae_frozen = state_dict.pop('_vae_frozen')
        if 'training_step' in state_dict:
            self.training_step = state_dict.pop('training_step')
            
        return super().load_state_dict(state_dict, strict)

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

