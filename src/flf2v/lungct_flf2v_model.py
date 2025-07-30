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
import torch.nn.functional as F

# Import from package namespace (Fix: proper imports)
from .lungct_vae import LungCTVAE
from .lungct_dit import LungCTDiT
from .lungct_flow_matching import FlowMatching


class LungCTFLF2V(nn.Module):
    """
    First-Last Frame to Video (FLF2V) model for lung CT sequences
    Combines VAE, DiT, and Flow Matching for temporal interpolation
    FIXED: Configurable sequence lengths for runtime cropping
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
    
    def crop_sequence_runtime(
        self, 
        video: torch.Tensor, 
        target_length: int,
        crop_strategy: str = "center"
    ) -> torch.Tensor:
        """
        Runtime sequence cropping for flexible training/inference
        
        Args:
            video: Input video [B, C, T, H, W]
            target_length: Desired sequence length
            crop_strategy: "center", "start", "end", or "phase_0_to_5"
        
        Returns:
            Cropped video [B, C, target_length, H, W]
        """
        B, C, T, H, W = video.shape
        
        if target_length >= T:
            return video  # No cropping needed
        
        if crop_strategy == "center":
            start_idx = (T - target_length) // 2
            end_idx = start_idx + target_length
        elif crop_strategy == "start":
            start_idx = 0
            end_idx = target_length
        elif crop_strategy == "end":
            start_idx = T - target_length
            end_idx = T
        elif crop_strategy == "phase_0_to_5":
            # For FLF2V: crop from phase 0 (0%) to phase 5 (50% breathing)
            # Assuming 82-frame sequence covers 90% of breathing cycle
            # Phase 5 (50%) is at frame index ~45 (50/90 * 82)
            phase_5_idx = min(int(0.5 / 0.9 * T), T - 1)
            end_idx = min(phase_5_idx + 1, target_length)
            start_idx = 0
            if end_idx - start_idx < target_length:
                end_idx = target_length
        else:
            raise ValueError(f"Unknown crop_strategy: {crop_strategy}")
        
        cropped = video[:, :, start_idx:end_idx]
        
        # Pad if necessary
        if cropped.shape[2] < target_length:
            pad_length = target_length - cropped.shape[2]
            pad_tensor = cropped[:, :, -1:].repeat(1, 1, pad_length, 1, 1)
            cropped = torch.cat([cropped, pad_tensor], dim=2)
        
        return cropped
    
    def forward(
        self,
        video: torch.Tensor,
        target_sequence_length: Optional[int] = None,
        crop_strategy: str = "center",
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        MEMORY-OPTIMIZED forward pass with configurable sequence length
        
        Args:
            video: Input video [B, C, T, H, W]
            target_sequence_length: Optional sequence length for runtime cropping
            crop_strategy: How to crop the sequence
            return_dict: Whether to return loss dictionary
        """
        # 🔧 NEW: Runtime sequence cropping
        if target_sequence_length is not None:
            video = self.crop_sequence_runtime(video, target_sequence_length, crop_strategy)
        num_frames = video.shape[2]
        # Extract first and last frames for conditioning
        first_frame = video[:, :, 0:1]  # [B, C, 1, H, W]
        last_frame = video[:, :, -1:]   # [B, C, 1, H, W]
        
        # ✅ SINGLE ENCODE - eliminate double encoding memory waste
        enc_full = self.vae.encode(video)
        video_latent = enc_full['latent']    # Multi-channel for DiT
        mu = enc_full['mu']                  # For KL loss
        logvar = enc_full['logvar']          # For KL loss  
        
        # Encode first/last frames (these are small, minimal memory impact)
        first_latent = self.vae.encode(first_frame)['latent']
        last_latent = self.vae.encode(last_frame)['latent']
        
        # Flow matching returns: velocity_loss, flf_loss
        flow_output = self.flow_matching(
            video_latent,
            first_latent,
            last_latent,
            dit_model=self.dit,
            return_dict=True
        )
        
        device = video.device  # current CUDA device

        losses = {
            # prefer whichever alias FlowMatching really returns
            "loss_velocity": flow_output.get("loss_velocity",
                                            flow_output.get("velocity_loss",
                                                            torch.tensor(0.0, device=device))),
            "loss_flf":      flow_output.get("loss_flf",
                                            flow_output.get("flf_loss",
                                                            torch.tensor(0.0, device=device))),
        }

            
        # VAE losses (only if not frozen)
        if self.training_step < self.freeze_vae_after and not self._vae_frozen:
            video_recon = self.vae.decode(video_latent)
            losses["loss_vae_recon"]    = F.mse_loss(video_recon, video)
            losses["loss_vae_kl"]       = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                                        / mu.numel())
            losses["loss_vae_temporal"] = (F.mse_loss(video_recon[:, :, 1:] - video_recon[:, :, :-1],
                                                    video[:,        :, 1:] - video[:,        :, :-1])
                                        if num_frames > 1 else torch.tensor(0.0))
        else:
            losses.update({
                "loss_vae_recon":    torch.tensor(0.0, device=video.device),
                "loss_vae_kl":       torch.tensor(0.0, device=video.device),
                "loss_vae_temporal": torch.tensor(0.0, device=video.device),
            })
                
        # Compute weighted total loss
        loss_total = (
            self.loss_weights["velocity_weight"] * losses["loss_velocity"]
            + self.loss_weights["flf_weight"]       * losses["loss_flf"]
            + self.loss_weights["vae_recon_weight"] * losses["loss_vae_recon"]
            + self.loss_weights["vae_kl_weight"]    * losses["loss_vae_kl"]
            + self.loss_weights["vae_temporal_weight"] * losses["loss_vae_temporal"]
        )
        losses["loss_total"] = loss_total
        
        # Update training step
        self.training_step += 1
        
        # Check if we should freeze VAE
        if self.training_step >= self.freeze_vae_after and not self._vae_frozen:
            self._freeze_vae()
        
        if return_dict:
            return losses
        else:
            return loss_total
    
    @torch.no_grad()
    def generate(
        self,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        num_frames: int = 41,
        target_sequence_length: Optional[int] = None,
        guidance_scale: float = 1.0,
        decode: bool = True,
        progress_bar: bool = True
    ) -> torch.Tensor:
        """
        Generate video sequence between first and last frames
        FIXED: Support for configurable sequence lengths
        
        Args:
            first_frame: First frame [B, C, H, W] or [B, C, 1, H, W]
            last_frame: Last frame [B, C, H, W] or [B, C, 1, H, W]
            num_frames: Number of frames to generate (default from training)
            target_sequence_length: Override for sequence length (for FLF2V)
            guidance_scale: Classifier-free guidance scale
            decode: Whether to decode to pixel space
            progress_bar: Show progress bar
        
        Returns:
            Generated sequence [B, C, T, H, W]
        """
        # 🔧 NEW: Override num_frames if target_sequence_length specified
        if target_sequence_length is not None:
            num_frames = target_sequence_length
        
        # Ensure correct input dimensions
        if first_frame.dim() == 4:
            first_frame = first_frame.unsqueeze(2)  # [B, C, H, W] → [B, C, 1, H, W]
        if last_frame.dim() == 4:
            last_frame = last_frame.unsqueeze(2)
        
        print(f"🎬 Generating {num_frames} frames")
        print(f"   Input shapes: first={first_frame.shape}, last={last_frame.shape}")
        
        # Encode frames to latent space
        first_latent = self.vae.encode(first_frame)['latent']
        last_latent = self.vae.encode(last_frame)['latent']
        
        print(f"   Latent shapes: first={first_latent.shape}, last={last_latent.shape}")
        
        # Generate latent sequence
        latent_video = self.flow_matching.sample(
            first_frame=first_latent,
            last_frame=last_latent,
            dit_model=self.dit,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            progress_bar=progress_bar
        )
        
        print(f"   Generated latent: {latent_video.shape}")
        
        # Decode to pixel space if requested
        if decode:
            generated_video = self.vae.decode(latent_video)
            print(f"   Decoded video: {generated_video.shape}")
            return generated_video
        else:
            return latent_video
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'free_gb': reserved - allocated
            }
        return {}
    
    def enable_flf2v_mode(self, sequence_length: int = 41):
        """
        Enable FLF2V mode with shorter sequences
        
        Args:
            sequence_length: Target sequence length (41 for half breathing cycle)
        """
        self.flf2v_mode = True
        self.flf2v_length = sequence_length
        print(f"✅ FLF2V mode enabled: {sequence_length} frames")
    
    def disable_flf2v_mode(self):
        """Disable FLF2V mode and return to full sequences"""
        self.flf2v_mode = False
        self.flf2v_length = None
        print("✅ FLF2V mode disabled: full sequences")
    
    def get_effective_sequence_length(self, default_length: int = 82) -> int:
        """Get the effective sequence length based on current mode"""
        if hasattr(self, 'flf2v_mode') and self.flf2v_mode:
            return self.flf2v_length
        return default_length