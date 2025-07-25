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
from .lungct_dit import DiTBlock
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
        dit: DiTBlock,
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
        """Actually freeze VAE parameters - Fix: proper freezing"""
        if not self._vae_frozen:
            for param in self.vae.parameters():
                param.requires_grad = False
            self._vae_frozen = True
            logging.info(f"VAE parameters frozen at step {self.training_step}")
    
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to latent space"""
        return self.vae.encode(frames)
    
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
        Fix: proper loss key naming and separation of losses from latents
        
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
            num_frames=num_frames,
            guidance_scale=guidance_scale
        )
        
        if decode:
            # Decode to pixel space
            video = self.decode_frames(latent_video)
            return video
        else:
            return latent_video


class FLF2VTrainer:
    """
    Training pipeline for FLF2V model
    Handles the complete training workflow with proper loss handling
    """
    
    def __init__(
        self,
        model: LungCTFLF2V,
        config: Dict,
        device: str = 'cuda',
        output_dir: str = './outputs'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_logging()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        
    def setup_optimizers(self):
        """Setup optimizers and schedulers for VAE and DiT components"""
        # VAE optimizer (used for first N steps)
        vae_params = list(self.model.vae.parameters())
        self.vae_optimizer = optim.AdamW(
            vae_params,
            lr=self.config.get('vae_lr', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # DiT + Flow Matching optimizer
        dit_params = (list(self.model.dit.parameters()) + 
                     list(self.model.flow_matching.parameters()))
        self.dit_optimizer = optim.AdamW(
            dit_params,
            lr=self.config.get('dit_lr', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Learning rate schedulers - Fix: add missing schedulers
        self.vae_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.vae_optimizer, 
            T_max=self.config.get('num_epochs', 100)
        )
        self.dit_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.dit_optimizer,
            T_max=self.config.get('num_epochs', 100)
        )
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss weights with defaults - Fix: provide all defaults"""
        return {
            'velocity_weight': self.config.get('velocity_weight', 1.0),
            'flf_weight': self.config.get('flf_weight', 0.1),
            'vae_recon_weight': self.config.get('vae_recon_weight', 1.0),
            'vae_kl_weight': self.config.get('vae_kl_weight', 0.01),
            'vae_temporal_weight': self.config.get('vae_temporal_weight', 0.1)
        }
    
    def calculate_weighted_loss(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate weighted total loss - Fix: only sum loss keys"""
        weights = self.get_loss_weights()
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Only sum keys that start with 'loss_'
        for key, value in loss_dict.items():
            if key.startswith('loss_') and isinstance(value, torch.Tensor):
                # Map loss key to weight key
                if key == 'loss_velocity':
                    weight = weights['velocity_weight']
                elif key == 'loss_flf':
                    weight = weights['flf_weight']
                elif key == 'loss_vae_recon':
                    weight = weights['vae_recon_weight']
                elif key == 'loss_vae_kl':
                    weight = weights['vae_kl_weight']
                elif key == 'loss_vae_temporal':
                    weight = weights['vae_temporal_weight']
                else:
                    weight = 1.0  # Default weight for unknown losses
                
                total_loss = total_loss + weight * value
        
        return total_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'vae_optimizer_state_dict': self.vae_optimizer.state_dict(),
            'dit_optimizer_state_dict': self.dit_optimizer.state_dict(),
            'vae_scheduler_state_dict': self.vae_scheduler.state_dict(),
            'dit_scheduler_state_dict': self.dit_scheduler.state_dict(),
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        self.dit_optimizer.load_state_dict(checkpoint['dit_optimizer_state_dict'])
        self.vae_scheduler.load_state_dict(checkpoint['vae_scheduler_state_dict'])
        self.dit_scheduler.load_state_dict(checkpoint['dit_scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train_epoch(self, train_loader, val_loader=None):
        """Train for one epoch - Fix: proper loss handling"""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            if 'video' in batch:
                video = batch['video'].to(self.device)
            elif 'target_frames' in batch:
                video = batch['target_frames'].to(self.device)
            else:
                raise ValueError("No video data found in batch")
            
            # Forward pass
            loss_dict = self.model(video, return_dict=True)
            
            # Calculate weighted total loss - Fix: only sum loss keys
            total_loss = self.calculate_weighted_loss(loss_dict)
            
            # Backward pass - Fix: proper optimizer selection
            if self.model.training_step < self.model.freeze_vae_after:
                # Train VAE
                self.vae_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.vae.parameters(), 
                    self.config.get('grad_clip', 1.0)
                )
                self.vae_optimizer.step()
            else:
                # Train DiT
                self.dit_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.dit.parameters()) + list(self.model.flow_matching.parameters()),
                    self.config.get('grad_clip', 1.0)
                )
                self.dit_optimizer.step()
            
            epoch_losses.append(total_loss.item())
            self.global_step += 1
            
            # Logging
            if batch_idx % 50 == 0:
                self.logger.info(
                    f"Epoch {self.epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {total_loss.item():.4f}, "
                    f"Training: {'VAE' if self.model.training_step < self.model.freeze_vae_after else 'DiT'}"
                )
        
        # Update learning rates - Fix: proper scheduler selection
        if self.model.training_step < self.model.freeze_vae_after:
            self.vae_scheduler.step()
        else:
            self.dit_scheduler.step()
        
        avg_loss = np.mean(epoch_losses)
        self.logger.info(f"Epoch {self.epoch} completed, Average Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model - Fix: proper loss handling"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                if 'video' in batch:
                    video = batch['video'].to(self.device)
                elif 'target_frames' in batch:
                    video = batch['target_frames'].to(self.device)
                else:
                    continue
                
                loss_dict = self.model(video, return_dict=True)
                total_loss = self.calculate_weighted_loss(loss_dict)  # Fix: use same loss calculation
                val_losses.append(total_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        self.logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        
        return avg_val_loss
    
    def train(self, train_loader, val_loader=None, num_epochs: int = None):
        """Complete training loop"""
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 100)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader, val_loader)
            
            # Validate
            val_loss = float('inf')
            if val_loader is not None and epoch % self.config.get('val_freq', 5) == 0:
                val_loss = self.validate(val_loader)
                
                # Check if this is the best model
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
            
            # Save checkpoint
            if epoch % self.config.get('save_freq', 10) == 0:
                self.save_checkpoint(epoch, is_best=(val_loss < best_val_loss))
        
        # Save final model
        self.save_checkpoint(num_epochs - 1, is_best=False)
        self.logger.info("Training completed!")