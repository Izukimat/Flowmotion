"""
Flow Matching Module for Lung CT FLF2V
Implements training and inference for flow matching
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np 

@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching"""
    num_sampling_steps: int = 50
    sigma_min: float = 1e-5
    use_ode_solver: str = "euler"  # "euler", "heun", "dpm"
    guidance_scale: float = 1.0
    
    # Training specific
    time_sampling: str = "uniform"  # "uniform", "logit_normal"
    loss_weighting: str = "uniform"  # "uniform", "velocity", "truncated"
    
    # FLF2V specific
    interpolation_method: str = "optimal_transport"  # "linear", "optimal_transport"
    
    # mid frame loss
    num_midpoints: int = 41
    mid_loss_weight: float = 1.0
    tv_loss_weight: float = 1e-3

class FlowMatching(nn.Module):
    """
    Flow Matching for FLF2V generation
    Learns to transform noise to data distribution while preserving first/last frames
    """
    
    def __init__(self, config: FlowMatchingConfig = None):
        super().__init__()
        self.config = config or FlowMatchingConfig()
    
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps for training"""
        if self.config.time_sampling == "uniform":
            t = torch.rand(batch_size, device=device)
        elif self.config.time_sampling == "logit_normal":
            u = torch.randn(batch_size, device=device)
            t = torch.sigmoid(u)
        else:
            raise ValueError(f"Unknown time sampling: {self.config.time_sampling}")
        
        return t
    
    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate between noise and data
        Returns interpolated sample and target velocity
        """
        # Reshape t for broadcasting
        t = t.view(-1, 1, 1, 1, 1)
        
        if self.config.interpolation_method == "linear":
            # Simple linear interpolation
            xt = t * x1 + (1 - t) * x0
            vt = x1 - x0
        elif self.config.interpolation_method == "optimal_transport":
            # Optimal transport path with sigma_min
            sigma_min = self.config.sigma_min
            xt = t * x1 + (1 - (1 - sigma_min) * t) * x0
            vt = x1 - (1 - sigma_min) * x0
        else:
            raise ValueError(f"Unknown interpolation: {self.config.interpolation_method}")
        
        return xt, vt
    
    def compute_loss(
        self,
        x1: torch.Tensor,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        dit_model: nn.Module,
        frozen_mask: Optional[torch.Tensor] = None,
        z_gt_mid: Optional[torch.Tensor] = None, 
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss
        
        Args:
            x1: Target latent video [B, C, D, H, W]
            first_frame: First frame to condition on
            last_frame: Last frame to condition on
            frozen_mask: Binary mask for frozen frames
        
        Returns:
            Dictionary of losses
        """
        B = x1.shape[0]
        device = x1.device
        
        # Sample noise
        noise_scale = 0.01
        x0 = torch.randn_like(x1) * noise_scale
        
        # Sample time
        t = self.sample_time(B, device)
        
        # Interpolate
        xt, vt_target = self.interpolate(x0, x1, t)
        
        # For FLF2V: replace first and last frames in xt with actual frames
        xt_flf = xt.clone()
        xt_flf[:, :, 0] = first_frame[:, :, 0]  # First depth slice
        xt_flf[:, :, -1] = last_frame[:, :, -1]  # Last depth slice
        
        # Predict velocity using passed DiT model
        vt_pred = dit_model(xt_flf, t, first_frame, last_frame, frozen_mask)
        # Compute loss with optional weighting
        if self.config.loss_weighting == "uniform":
            loss_weight = 1.0
        elif self.config.loss_weighting == "velocity":
            # Weight by velocity magnitude
            loss_weight = torch.norm(vt_target.reshape(B, -1), dim=1, keepdim=True)
            loss_weight = loss_weight / (loss_weight.mean() + 1e-8)  # Normalize
            loss_weight = loss_weight.view(-1, 1, 1, 1, 1)
        elif self.config.loss_weighting == "truncated":
            # Truncated weighting for stability
            loss_weight = torch.clamp(t.view(-1, 1, 1, 1, 1), 0.1, 0.9)
        else:
            loss_weight = 1.0
        
        # for 3D
        # vt_target = F.interpolate(
        # vt_target, size=vt_pred.shape[2:], mode="trilinear", align_corners=False
        # )
        
        if vt_pred.shape != vt_target.shape:
            B, C, D, H, W = vt_target.shape
            vt_target = vt_target.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            vt_target = F.interpolate(vt_target, size=vt_pred.shape[-2:], mode="bilinear", align_corners=False)
            vt_target = vt_target.reshape(B, D, C, *vt_pred.shape[-2:]).permute(0, 2, 1, 3, 4)
        # MSE loss on velocity
        loss_velocity = F.mse_loss(vt_pred * loss_weight, vt_target * loss_weight)
        
        # Additional loss to preserve first/last frames
        loss_flf = F.mse_loss(vt_pred[:, :, [0, -1]], vt_target[:, :, [0, -1]])

        # --------- mid-frame latent supervision -----------
        loss_mid = torch.tensor(0.0, device=x1.device)
        if z_gt_mid is not None and self.config.num_midpoints > 0:
            T = z_gt_mid.shape[2]
            mid_ids = torch.tensor(
                np.linspace(1, T - 2, num=min(self.config.num_midpoints, max(T-2,1)), dtype=int),
                device=x1.device
            )
            loss_sum = 0.0
            for idx in mid_ids:
                pred_mid = xt_flf[:, :, idx]
                gt_mid = z_gt_mid[:, :, idx]
                loss_sum = loss_sum + F.mse_loss(pred_mid, gt_mid)
            loss_mid = loss_sum / len(mid_ids)

        # --------- temporal smoothness loss ---------------
        loss_tv = torch.mean((xt_flf[:, :, 1:] - xt_flf[:, :, :-1]) ** 2)
        losses = {
            'loss_velocity': loss_velocity,
            'loss_flf': loss_flf,
            'loss_mid': loss_mid * self.config.mid_loss_weight,
            'loss_tv':  loss_tv  * self.config.tv_loss_weight,
        }

        return losses
    
    @torch.no_grad()
    def sample(
        self,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        dit_model: nn.Module,
        num_frames: int,
        guidance_scale: Optional[float] = None,
        progress_bar: bool = True
    ) -> torch.Tensor:
        """
        Generate video frames between first and last frame
        
        Args:
            first_frame: First frame latent [B, C, 1, H, W]
            last_frame: Last frame latent [B, C, 1, H, W]
            num_frames: Total number of frames to generate (including first/last)
            guidance_scale: Optional classifier-free guidance scale
            progress_bar: Show progress bar during sampling
        
        Returns:
            Generated video latent [B, C, D, H, W]
        """
        B, C, _, H, W = first_frame.shape
        device = first_frame.device
        D = num_frames
        
        # Initialize with noise
        x = torch.randn(B, C, D, H, W, device=device)
        
        # Set first and last frames
        x[:, :, 0] = first_frame.squeeze(2)
        x[:, :, -1] = last_frame.squeeze(2)
        
        # Create frozen mask
        frozen_mask = torch.zeros(B, D, device=device)
        frozen_mask[:, 0] = 1
        frozen_mask[:, -1] = 1
        
        # Expand first/last frames to full depth for conditioning
        first_frame_full = first_frame.repeat(1, 1, D, 1, 1)
        last_frame_full = last_frame.repeat(1, 1, D, 1, 1)
        
        # Setup timesteps
        num_steps = self.config.num_sampling_steps
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)
        dt = 1.0 / num_steps

        # Sampling loop with debugging
        iterator = tqdm(range(num_steps), desc="Sampling", disable=not progress_bar)
        
        for i in iterator:
            t = timesteps[i]
            t_batch = torch.full((B,), t.item(), device=device)
            
            # Predict velocity
            with torch.cuda.amp.autocast(enabled=True):
                v = dit_model(x, t_batch, first_frame_full, last_frame_full, frozen_mask)
            
            # Classifier-free guidance if requested
            if guidance_scale is not None and guidance_scale != 1.0:
                v_uncond = dit_model(
                    x, t_batch,
                    torch.randn_like(first_frame_full),
                    torch.randn_like(last_frame_full),
                    frozen_mask
                )
                v = v_uncond + guidance_scale * (v - v_uncond)
            
            # Update x (Euler step)
            if self.config.use_ode_solver == "euler":
                x = x + dt * v
            elif self.config.use_ode_solver == "heun":
                # Heun's method (2nd order)
                if i < num_steps - 1:
                    t_next = timesteps[i + 1]
                    t_next_batch = torch.full((B,), t_next.item(), device=device)
                    x_temp = x + dt * v
                    v_next = dit_model(
                        x_temp, t_next_batch,
                        first_frame_full, last_frame_full, frozen_mask
                    )
                    x = x + dt * 0.5 * (v + v_next)
                else:
                    x = x + dt * v
            
            # Enforce first/last frame constraints
            x[:, :, 0] = first_frame.squeeze(2)
            x[:, :, -1] = last_frame.squeeze(2)
        
        print(f"✅ Sampling completed. Final shape: {x.shape}")
        return x
    
    def forward(
        self,
        x1: torch.Tensor,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        dit_model: nn.Module, 
        return_dict: bool = True,
        frozen_mask: Optional[torch.Tensor] = None,
        z_gt_mid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computes loss during training"""
        return self.compute_loss(x1, first_frame, last_frame, dit_model, frozen_mask, z_gt_mid=z_gt_mid)

def create_flow_matching_model(config: Optional[Dict] = None) -> FlowMatching:
    """
    Factory function to create flow matching model
    """
    if config and 'config' in config:
        config_dict = config['config'].copy()
        # Robust type casting:
        for k in ['sigma_min', 'guidance_scale', 'mid_loss_weight', 'tv_loss_weight']:
            if k in config_dict:
                config_dict[k] = float(config_dict[k])
        if 'num_sampling_steps' in config_dict:
            config_dict['num_sampling_steps'] = int(config_dict['num_sampling_steps'])
        if 'num_midpoints' in config_dict:
            config_dict['num_midpoints'] = int(config_dict['num_midpoints'])
        fm_config = FlowMatchingConfig(**config_dict)
    else:
        fm_config = FlowMatchingConfig(**(config or {}))
    return FlowMatching(fm_config)