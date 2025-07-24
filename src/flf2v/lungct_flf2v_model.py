"""
Main FLF2V Model for Lung CT
Integrates VAE, DiT, and Flow Matching components
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path
import wandb
from tqdm import tqdm
import json


class LungCTFLF2V(nn.Module):
    """
    Complete FLF2V model for lung CT video generation
    """
    
    def __init__(
        self,
        vae: nn.Module,
        dit: nn.Module,
        flow_matching: nn.Module,
        freeze_vae_after: int = 5000
    ):
        super().__init__()
        self.vae = vae
        self.dit = dit
        self.flow_matching = flow_matching
        self.freeze_vae_after = freeze_vae_after
        self.training_step = 0
    
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to latent space"""
        return self.vae.encode(frames)['latent']
    
    def decode_frames(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to frames"""
        return self.vae.decode(latents)
    
    def forward(
        self,
        video: torch.Tensor,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass
        
        Args:
            video: Input video [B, 1, D, H, W]
        """
        B, C, D, H, W = video.shape
        
        # Extract first and last frames
        first_frame = video[:, :, 0:1]
        last_frame = video[:, :, -1:]
        
        # Encode video to latent
        if self.training and self.training_step < self.freeze_vae_after:
            # VAE is trainable
            video_latent = self.encode_frames(video)
            first_latent = video_latent[:, :, 0:1]
            last_latent = video_latent[:, :, -1:]
        else:
            # VAE is frozen
            with torch.no_grad():
                video_latent = self.encode_frames(video)
                first_latent = video_latent[:, :, 0:1]
                last_latent = video_latent[:, :, -1:]
        
        # Flow matching loss
        flow_losses = self.flow_matching(
            video_latent,
            first_latent.repeat(1, 1, video_latent.size(2), 1, 1),
            last_latent.repeat(1, 1, video_latent.size(2), 1, 1)
        )
        
        # VAE reconstruction loss (if VAE is trainable)
        vae_losses = {}
        if self.training and self.training_step < self.freeze_vae_after:
            vae_output = self.vae(video)
            vae_losses = {
                'vae_recon': vae_output['loss_recon'],
                'vae_kl': vae_output['loss_kl'],
                'vae_temporal': vae_output['loss_temporal']
            }
        
        self.training_step += 1
        
        if return_dict:
            return {
                **flow_losses,
                **vae_losses,
                'video_latent': video_latent,
                'first_latent': first_latent,
                'last_latent': last_latent
            }
        else:
            return flow_losses['loss_total']
    
    @torch.no_grad()
    def generate(
        self,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        num_frames: int = 40,
        guidance_scale: float = 1.0,
        decode: bool = True
    ) -> torch.Tensor:
        """
        Generate video between first and last frames
        
        Args:
            first_frame: First frame [B, 1, H, W]
            last_frame: Last frame [B, 1, H, W]
            num_frames: Total number of frames to generate
            guidance_scale: CFG scale
            decode: Whether to decode to pixel space
        
        Returns:
            Generated video [B, 1, D, H, W] or latents
        """
        # Add depth dimension
        first_frame = first_frame.unsqueeze(2)  # [B, 1, 1, H, W]
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


class LungCTDataset(Dataset):
    """
    Dataset for lung CT sequences
    Handles loading and preprocessing
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        num_frames: int = 40,
        slice_indices: List[int] = None,  # e.g., [10, 25, 40] for apex/middle/base
        transform: Optional[callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.slice_indices = slice_indices
        self.transform = transform
        
        # Load file list
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load sample metadata"""
        # This is a placeholder - implement based on your data format
        samples = []
        
        # Example structure:
        # samples.append({
        #     'patient_id': 'P001',
        #     'scan_path': 'path/to/scan.nii.gz',
        #     'slice_idx': 25,  # Middle slice
        #     'breathing_phase': 'inhale'
        # })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load CT volume (placeholder - use your actual loading code)
        # volume = load_ct_volume(sample['scan_path'])
        
        # Extract temporal sequence around the slice
        # sequence = extract_breathing_sequence(volume, sample['slice_idx'], self.num_frames)
        
        # For now, return dummy data
        sequence = torch.randn(1, self.num_frames, 128, 128)
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return {
            'video': sequence,
            'patient_id': sample.get('patient_id', f'patient_{idx}'),
            'metadata': sample
        }


class FLF2VTrainer:
    """
    Training pipeline for FLF2V model
    """
    
    def __init__(
        self,
        model: LungCTFLF2V,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizers
        self.setup_optimizers()
        
        # Setup logging
        self.setup_logging()
    
    def setup_optimizers(self):
        """Setup optimizers for VAE and DiT"""
        # VAE optimizer (only for first N steps)
        vae_params = list(self.model.vae.parameters())
        self.vae_optimizer = optim.AdamW(
            vae_params,
            lr=self.config['vae_lr'],
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # DiT optimizer
        dit_params = list(self.model.dit.parameters()) + list(self.model.flow_matching.parameters())
        self.dit_optimizer = optim.AdamW(
            dit_params,
            lr=self.config['dit_lr'],
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Learning rate schedulers
        self.vae_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.vae_optimizer,
            T_max=self.config.get('vae_freeze_step', 5000)
        )
        
        self.dit_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.dit_optimizer,
            T_0=self.config.get('scheduler_T0', 1000),
            T_mult=2
        )
    
    def setup_logging(self):
        """Setup logging and wandb"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if self.config.get('use_wandb', False):
            wandb.init(
                project="lung-ct-flf2v",
                config=self.config,
                name=self.config.get('run_name', 'flf2v_training')
            )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        video = batch['video'].to(self.device)
        
        # Forward pass
        outputs = self.model(video)
        
        # Compute total loss
        total_loss = 0
        losses_dict = {}
        
        # Flow matching losses
        for k, v in outputs.items():
            if k.startswith('loss_'):
                losses_dict[k] = v.item()
                if k == 'loss_velocity':
                    total_loss += v * self.config.get('velocity_weight', 1.0)
                elif k == 'loss_flf':
                    total_loss += v * self.config.get('flf_weight', 0.1)
                elif k == 'vae_recon':
                    total_loss += v * self.config.get('vae_recon_weight', 1.0)
                elif k == 'vae_kl':
                    total_loss += v * self.config.get('vae_kl_weight', 0.01)
                elif k == 'vae_temporal':
                    total_loss += v * self.config.get('vae_temporal_weight', 0.1)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if self.config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )
        
        # Optimizer steps
        if self.model.training_step < self.model.freeze_vae_after:
            self.vae_optimizer.step()
            self.vae_optimizer.zero_grad()
            self.vae_scheduler.step()
        
        self.dit_optimizer.step()
        self.dit_optimizer.zero_grad()
        self.dit_scheduler.step()
        
        losses_dict['total_loss'] = total_loss.item()
        
        return losses_dict
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100
    ):
        """Main training loop"""
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_losses = {}
            
            # Training
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                losses = self.train_step(batch)
                
                # Update running averages
                for k, v in losses.items():
                    if k not in epoch_losses:
                        epoch_losses[k] = []
                    epoch_losses[k].append(v)
                
                # Update progress bar
                pbar.set_postfix({
                    k: np.mean(v[-100:]) for k, v in epoch_losses.items()
                })
                
                # Log to wandb
                if self.config.get('use_wandb', False) and batch_idx % 100 == 0:
                    wandb.log({
                        f"train/{k}": v for k, v in losses.items()
                    })
            
            # Validation
            if val_loader is not None and epoch % self.config.get('val_freq', 5) == 0:
                self.validate(val_loader, epoch)
            
            # Save checkpoint
            if epoch % self.config.get('save_freq', 10) == 0:
                self.save_checkpoint(epoch)
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int):
        """Validation step with generation"""
        self.model.eval()
        
        val_losses = []
        
        for batch in tqdm(val_loader, desc="Validation"):
            video = batch['video'].to(self.device)
            outputs = self.model(video)
            
            # Collect losses
            batch_losses = {
                k: v.item() for k, v in outputs.items() 
                if k.startswith('loss_')
            }
            val_losses.append(batch_losses)
        
        # Average losses
        avg_losses = {}
        for k in val_losses[0].keys():
            avg_losses[k] = np.mean([l[k] for l in val_losses])
        
        # Generate samples
        sample_batch = next(iter(val_loader))
        sample_video = sample_batch['video'][:4].to(self.device)
        
        first_frame = sample_video[:, :, 0]
        last_frame = sample_video[:, :, -1]
        
        generated = self.model.generate(
            first_frame, last_frame,
            num_frames=sample_video.size(2)
        )
        
        # Log results
        if self.config.get('use_wandb', False):
            wandb.log({
                f"val/{k}": v for k, v in avg_losses.items()
            })
            
            # Log generated videos
            wandb.log({
                "generated_videos": wandb.Video(
                    generated.cpu().numpy(),
                    fps=4,
                    format="mp4"
                )
            })
        
        self.model.train()
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        save_path = Path(self.config['save_dir']) / f"checkpoint_epoch_{epoch}.pt"
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'vae_optimizer': self.vae_optimizer.state_dict(),
            'dit_optimizer': self.dit_optimizer.state_dict(),
            'config': self.config
        }, save_path)
        
        self.logger.info(f"Saved checkpoint to {save_path}")


def create_flf2v_model(config: Dict) -> LungCTFLF2V:
    """
    Factory function to create complete FLF2V model
    """
    from .vae import LungCTVAE
    from .dit import create_dit_model
    from .flow_matching import create_flow_matching_model
    
    # Create VAE
    vae = LungCTVAE(
        base_model_name=config.get('vae_base_model', 'medvae_4_1_3d'),
        latent_channels=config.get('latent_channels', 8),
        temporal_weight=config.get('vae_temporal_weight', 0.1),
        use_tanh_scaling=True,
        freeze_pretrained=False
    )
    
    # Create DiT
    dit = create_dit_model(config.get('dit_config', {}))
    
    # Create Flow Matching
    flow_matching = create_flow_matching_model(
        dit,
        config.get('flow_matching_config', {})
    )
    
    # Create full model
    model = LungCTFLF2V(
        vae=vae,
        dit=dit,
        flow_matching=flow_matching,
        freeze_vae_after=config.get('freeze_vae_after', 5000)
    )
    
    return model