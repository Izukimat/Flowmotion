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
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            
            return {
                'allocated_gb': allocated,
                'cached_gb': cached, 
                'max_allocated_gb': max_allocated,
                'vae_frozen': self._vae_frozen,
                'training_step': self.training_step
            }
        else:
            return {'error': 'CUDA not available'}

    def validate_model_config(self):
        """Validate that DiT config matches VAE compression reality"""
        # Check VAE compression factor
        actual_compression = self.vae.compression_factor
        
        # Get DiT expected latent size
        dit_latent_size = self.dit.latent_size
        
        print(f"🔍 Model Configuration Validation:")
        print(f"   VAE compression factor: {actual_compression}x")
        print(f"   VAE latent channels: {self.vae.latent_channels}")
        print(f"   DiT expected latent size: {dit_latent_size}")
        print(f"   DiT hidden dim: {self.dit.hidden_dim}")
        print(f"   DiT depth: {self.dit.depth}")
        
        # Validate spatial dimensions match
        expected_spatial = 128 // actual_compression  # Assuming 128x128 input
        dit_spatial = dit_latent_size[1] if len(dit_latent_size) > 1 else None
        
        if dit_spatial and dit_spatial != expected_spatial:
            print(f"⚠️  MISMATCH: DiT expects {dit_spatial}², VAE outputs {expected_spatial}²")
            return False
        else:
            print(f"✅ Configuration looks correct")
            return True
        
    def training_step_completed(self):
        """Call this after each training step"""
        self.training_step += 1
        
        # Freeze VAE if needed
        if self.training_step >= self.freeze_vae_after and not self._vae_frozen:
            self._freeze_vae()
        
        # Log memory stats every 10 steps
        if self.training_step % 10 == 0:
            stats = self.get_memory_stats()
            if 'allocated_gb' in stats:
                print(f"Step {self.training_step}: {stats['allocated_gb']:.1f}GB allocated, "
                    f"max: {stats['max_allocated_gb']:.1f}GB")

    def forward(
        self,
        video: torch.Tensor,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        MEMORY-OPTIMIZED forward pass
        - Single encode per video (no double encoding)
        - Reuse encoded components for VAE losses
        - Detach returned latents
        """
        batch_size, channels, num_frames, height, width = video.shape
        
        # Extract first and last frames for conditioning
        first_frame = video[:, :, 0:1]  # [B, C, 1, H, W]
        last_frame = video[:, :, -1:]   # [B, C, 1, H, W]
        
        # ✅ SINGLE ENCODE - eliminate double encoding memory waste
        enc_full = self.vae.encode(video)
        video_latent = enc_full['latent']    # Multi-channel for DiT
        mu = enc_full['mu']                  # For KL loss
        logvar = enc_full['logvar']          # For KL loss  
        z1 = enc_full['z1']                  # 1-channel for VAE reconstruction
        
        # Encode first/last frames (these are small, minimal memory impact)
        first_latent = self.vae.encode(first_frame)['latent']
        last_latent = self.vae.encode(last_frame)['latent']
        
        # Flow matching loss
        flow_output = self.flow_matching(
            video_latent,
            first_latent,
            last_latent,
            dit_model=self.dit,
            return_dict=True
        )
        
        # Extract flow losses
        flow_losses = {
            f'loss_{k.replace("_loss", "")}': v 
            for k, v in flow_output.items() 
            if k.endswith('_loss') and isinstance(v, torch.Tensor)
        }
        
        # ✅ VAE LOSSES - reuse components, no second encode
        vae_losses = {}
        if self.training_step < self.freeze_vae_after and not self._vae_frozen:
            
            # Option 1: Decode from multi-channel latent (if VAE decode is fixed)
            try:
                x_recon = self.vae.decode(video_latent)
            except:
                # Option 2: Fallback to 1-channel decode if multi-channel fails
                x_recon = self.vae.decode_z1(z1)
            
            # Compute VAE losses without re-encoding
            loss_recon = torch.nn.functional.mse_loss(x_recon, video, reduction='mean')
            loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Temporal consistency loss
            loss_temporal = torch.tensor(0.0, device=video.device)
            if video_latent.size(2) > 1:
                z_diff = video_latent[:, :, 1:] - video_latent[:, :, :-1]
                loss_temporal = torch.mean(z_diff.pow(2))
            
            vae_losses = {
                'loss_vae_recon': loss_recon,
                'loss_vae_kl': loss_kl, 
                'loss_vae_temporal': loss_temporal
            }
        
        # Combine all losses
        all_losses = {**flow_losses, **vae_losses}
        
        # Weighted total loss
        total_loss = torch.tensor(0.0, device=video.device)
        for loss_key, loss_value in all_losses.items():
            if isinstance(loss_value, torch.Tensor):
                weight_key = loss_key.replace('loss_', '') + '_weight'
                weight = self.loss_weights.get(weight_key, 1.0)
                total_loss += weight * loss_value
        
        if return_dict:
            # ✅ DETACH latents to prevent memory retention
            result = {
                **all_losses,
                'loss_total': total_loss,
                'latents': {
                    'video_latent': video_latent.detach(),      # ← DETACHED
                    'first_latent': first_latent.detach(),      # ← DETACHED
                    'last_latent': last_latent.detach(),        # ← DETACHED
                }
            }
            return result
        else:
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

