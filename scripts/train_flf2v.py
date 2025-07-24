#!/usr/bin/env python3
"""
Training script for Lung CT FLF2V model
Handles the complete training pipeline including data loading, model training, and checkpointing
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import wandb
from einops import rearrange
import yaml

# Import our modules (adjust path as needed)
from lungct_vae import LungCTVAE, VAELoss
from lungct_dit import create_dit_model
from lungct_flow_matching import create_flow_matching_model, FlowMatchingConfig
from lungct_flf2v_model import LungCTFLF2V


# ============= Data Loading =============

class LungCTDataset(Dataset):
    """
    Dataset for lung CT breathing sequences
    Expects data organized as:
    - data_dir/
        - patient_001/
            - inhale_001.nii.gz
            - exhale_001.nii.gz
            - breathing_sequence_001.nii.gz  # 4D volume
        - patient_002/
            ...
    """
    
    def __init__(
        self,
        data_dir: str,
        csv_file: str,  # CSV with patient IDs and metadata
        split: str = 'train',
        num_frames: int = 40,
        target_size: Tuple[int, int, int] = (128, 128, 40),
        slice_mode: str = 'three_slices',  # 'three_slices' or 'full_volume'
        slice_indices: Optional[List[int]] = None,
        intensity_window: Tuple[float, float] = (-1000, 500),  # Lung window
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.target_size = target_size
        self.slice_mode = slice_mode
        self.slice_indices = slice_indices or [10, 25, 40]  # Apex, middle, base
        self.intensity_window = intensity_window
        self.augment = augment and (split == 'train')
        
        # Load metadata
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split]
        
        # Build sample list
        self.samples = self._build_samples()
        
        logging.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _build_samples(self) -> List[Dict]:
        """Build list of samples from metadata"""
        samples = []
        
        for _, row in self.df.iterrows():
            patient_dir = self.data_dir / row['patient_id']
            
            if self.slice_mode == 'three_slices':
                # Create samples for each slice position
                for slice_idx in self.slice_indices:
                    samples.append({
                        'patient_id': row['patient_id'],
                        'sequence_path': patient_dir / row['sequence_file'],
                        'slice_idx': slice_idx,
                        'slice_name': f"slice_{slice_idx}",
                        'metadata': row.to_dict()
                    })
            else:
                # Full volume mode
                samples.append({
                    'patient_id': row['patient_id'],
                    'sequence_path': patient_dir / row['sequence_file'],
                    'metadata': row.to_dict()
                })
        
        return samples
    
    def _load_sequence(self, path: Path) -> np.ndarray:
        """Load 4D NIfTI sequence"""
        nii = nib.load(str(path))
        data = nii.get_fdata()
        
        # Ensure 4D
        if data.ndim == 3:
            data = data[..., np.newaxis]
        
        return data
    
    def _preprocess(self, volume: np.ndarray) -> np.ndarray:
        """Apply intensity windowing and normalization"""
        # Apply lung window
        volume = np.clip(volume, self.intensity_window[0], self.intensity_window[1])
        
        # Normalize to [-1, 1]
        volume = 2 * (volume - self.intensity_window[0]) / (self.intensity_window[1] - self.intensity_window[0]) - 1
        
        return volume
    
    def _augment(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation"""
        if not self.augment:
            return volume
        
        # Random intensity shift
        if torch.rand(1) < 0.5:
            shift = torch.randn(1) * 0.1
            volume = volume + shift
        
        # Random intensity scale
        if torch.rand(1) < 0.5:
            scale = 1 + torch.randn(1) * 0.1
            volume = volume * scale
        
        # Random noise
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(volume) * 0.02
            volume = volume + noise
        
        # Clamp to valid range
        volume = torch.clamp(volume, -1, 1)
        
        return volume
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load sequence
        sequence = self._load_sequence(sample['sequence_path'])
        
        if self.slice_mode == 'three_slices':
            # Extract specific slice across time
            slice_idx = sample['slice_idx']
            slice_sequence = sequence[:, :, slice_idx, :]  # [H, W, T]
            
            # Transpose to [T, H, W]
            slice_sequence = np.transpose(slice_sequence, (2, 0, 1))
            
            # Sample temporal window
            T = slice_sequence.shape[0]
            if T > self.num_frames:
                start_idx = np.random.randint(0, T - self.num_frames)
                slice_sequence = slice_sequence[start_idx:start_idx + self.num_frames]
            elif T < self.num_frames:
                # Pad with repetition
                pad_frames = self.num_frames - T
                slice_sequence = np.pad(
                    slice_sequence,
                    ((0, pad_frames), (0, 0), (0, 0)),
                    mode='edge'
                )
            
            # Resize spatial dimensions
            if slice_sequence.shape[1:] != self.target_size[:2]:
                import cv2
                resized = []
                for t in range(self.num_frames):
                    frame = cv2.resize(
                        slice_sequence[t],
                        self.target_size[:2][::-1],  # cv2 uses (W, H)
                        interpolation=cv2.INTER_LINEAR
                    )
                    resized.append(frame)
                slice_sequence = np.stack(resized)
            
            # Preprocess
            slice_sequence = self._preprocess(slice_sequence)
            
            # Convert to tensor [1, T, H, W]
            video = torch.from_numpy(slice_sequence).float().unsqueeze(0)
            
        else:
            # Full volume mode - implement as needed
            raise NotImplementedError("Full volume mode not yet implemented")
        
        # Augment
        video = self._augment(video)
        
        # Rearrange to [1, T, H, W] -> [1, T, H, W] (already in correct format)
        
        return {
            'video': video,
            'patient_id': sample['patient_id'],
            'slice_name': sample.get('slice_name', 'full'),
            'metadata': sample['metadata']
        }


# ============= Training Functions =============

def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def create_model(config: Dict, device: str) -> LungCTFLF2V:
    """Create model from config"""
    # Create VAE
    vae = LungCTVAE(
        base_model_name=config['model']['vae_base_model'],
        latent_channels=config['model']['latent_channels'],
        temporal_weight=config['model']['vae_temporal_weight'],
        use_tanh_scaling=True,
        freeze_pretrained=False
    )
    
    # Create DiT
    dit = create_dit_model(config['model']['dit_config'])
    
    # Create Flow Matching
    fm_config = FlowMatchingConfig(**config['model']['flow_matching_config'])
    flow_matching = create_flow_matching_model(dit, {'config': fm_config})
    
    # Create full model
    model = LungCTFLF2V(
        vae=vae,
        dit=dit,
        flow_matching=flow_matching,
        freeze_vae_after=config['model']['freeze_vae_after']
    )
    
    return model.to(device)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    vae_optimizer: optim.Optimizer,
    dit_optimizer: optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    config: Dict,
    device: str,
    wandb_log: bool = True
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    epoch_losses = {
        'total': [],
        'velocity': [],
        'flf': [],
        'vae_recon': [],
        'vae_kl': [],
        'vae_temporal': []
    }
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        video = batch['video'].to(device)
        
        # Zero gradients
        vae_optimizer.zero_grad()
        dit_optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(video)
            
            # Compute total loss
            total_loss = 0
            loss_weights = {
                'loss_velocity': config['training']['velocity_weight'],
                'loss_flf': config['training']['flf_weight'],
                'vae_recon': config['training']['vae_recon_weight'],
                'vae_kl': config['training']['vae_kl_weight'],
                'vae_temporal': config['training']['vae_temporal_weight']
            }
            
            for loss_name, weight in loss_weights.items():
                if loss_name in outputs:
                    total_loss += weight * outputs[loss_name]
                    epoch_losses[loss_name.replace('loss_', '')].append(
                        outputs[loss_name].item()
                    )
        
        # Backward pass
        scaler.scale(total_loss).backward()
        
        # Gradient clipping
        if config['training']['grad_clip'] > 0:
            scaler.unscale_(vae_optimizer)
            scaler.unscale_(dit_optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip']
            )
        
        # Optimizer steps
        if model.training_step < model.freeze_vae_after:
            scaler.step(vae_optimizer)
        scaler.step(dit_optimizer)
        scaler.update()
        
        epoch_losses['total'].append(total_loss.item())
        
        # Update progress bar
        avg_losses = {k: np.mean(v[-100:]) if v else 0 for k, v in epoch_losses.items()}
        pbar.set_postfix(avg_losses)
        
        # Log to wandb
        if wandb_log and batch_idx % 100 == 0:
            wandb.log({
                f'train/{k}': v for k, v in avg_losses.items()
            })
    
    # Return epoch averages
    return {k: np.mean(v) if v else 0 for k, v in epoch_losses.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    epoch: int,
    device: str,
    num_samples: int = 4
) -> Dict[str, float]:
    """Validation with sample generation"""
    model.eval()
    
    val_losses = []
    generated_samples = []
    
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
        video = batch['video'].to(device)
        
        # Forward pass
        outputs = model(video)
        
        # Collect losses
        batch_losses = {
            k.replace('loss_', ''): v.item() 
            for k, v in outputs.items() 
            if k.startswith('loss_')
        }
        val_losses.append(batch_losses)
        
        # Generate samples for first batch
        if batch_idx == 0 and len(generated_samples) < num_samples:
            B = min(num_samples, video.shape[0])
            first_frame = video[:B, :, 0]
            last_frame = video[:B, :, -1]
            
            generated = model.generate(
                first_frame,
                last_frame,
                num_frames=video.shape[2],
                guidance_scale=1.0
            )
            
            generated_samples.append({
                'input': video[:B].cpu(),
                'generated': generated.cpu(),
                'first_frame': first_frame.cpu(),
                'last_frame': last_frame.cpu()
            })
    
    # Average losses
    avg_losses = {}
    if val_losses:
        for k in val_losses[0].keys():
            avg_losses[k] = np.mean([l[k] for l in val_losses])
    
    return avg_losses, generated_samples


def save_checkpoint(
    model: nn.Module,
    vae_optimizer: optim.Optimizer,
    dit_optimizer: optim.Optimizer,
    epoch: int,
    config: Dict,
    save_path: Path,
    is_best: bool = False
):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'vae_optimizer': vae_optimizer.state_dict(),
        'dit_optimizer': dit_optimizer.state_dict(),
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.parent / 'best_model.pt'
        torch.save(checkpoint, best_path)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Lung CT FLF2V model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--csv-file', type=str, required=True, help='CSV with metadata')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--wandb-project', type=str, default='lungct-flf2v', help='WandB project name')
    parser.add_argument('--wandb-run', type=str, default=None, help='WandB run name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='DataLoader prefetch')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = f'cuda:{local_rank}'
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup output directory
    output_dir = Path(args.output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
    
    # Setup wandb
    if not args.no_wandb and rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"flf2v_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config
        )
    
    # Create datasets
    train_dataset = LungCTDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        split='train',
        num_frames=config['data']['num_frames'],
        target_size=config['data']['target_size'],
        slice_mode=config['data']['slice_mode'],
        slice_indices=config['data'].get('slice_indices'),
        augment=True
    )
    
    val_dataset = LungCTDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        split='val',
        num_frames=config['data']['num_frames'],
        target_size=config['data']['target_size'],
        slice_mode=config['data']['slice_mode'],
        slice_indices=config['data'].get('slice_indices'),
        augment=False
    )
    
    # Create data loaders
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.num_workers > 0
    )
    
    # Create model
    logging.info("Creating model...")
    model = create_model(config, device)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        logging.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Setup distributed model
    if args.distributed:
        model = DDP(model, device_ids=[local_rank])
    
    # Create optimizers
    vae_params = list(model.vae.parameters()) if not args.distributed else list(model.module.vae.parameters())
    dit_params = list(model.dit.parameters()) + list(model.flow_matching.parameters())
    if args.distributed:
        dit_params = list(model.module.dit.parameters()) + list(model.module.flow_matching.parameters())
    
    vae_optimizer = optim.AdamW(
        vae_params,
        lr=config['training']['vae_lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    dit_optimizer = optim.AdamW(
        dit_params,
        lr=config['training']['dit_lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Load optimizer states if resuming
    if args.resume:
        vae_optimizer.load_state_dict(checkpoint['vae_optimizer'])
        dit_optimizer.load_state_dict(checkpoint['dit_optimizer'])
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_losses = train_epoch(
            model, train_loader,
            vae_optimizer, dit_optimizer,
            scaler, epoch, config, device,
            wandb_log=(not args.no_wandb and rank == 0)
        )
        
        # Validate
        if epoch % config['training']['val_freq'] == 0:
            val_losses, generated_samples = validate(
                model, val_loader, epoch, device
            )
            
            if rank == 0:
                # Log validation losses
                if not args.no_wandb:
                    wandb.log({
                        f'val/{k}': v for k, v in val_losses.items()
                    })
                    
                    # Log generated samples
                    if generated_samples:
                        sample = generated_samples[0]
                        wandb.log({
                            'val/generated_video': wandb.Video(
                                sample['generated'].numpy(),
                                fps=4,
                                format="mp4"
                            )
                        })
                
                # Save checkpoint
                is_best = val_losses.get('total', float('inf')) < best_val_loss
                if is_best:
                    best_val_loss = val_losses.get('total', float('inf'))
                
                save_checkpoint(
                    model.module if args.distributed else model,
                    vae_optimizer,
                    dit_optimizer,
                    epoch,
                    config,
                    output_dir / f'checkpoint_epoch_{epoch}.pt',
                    is_best=is_best
                )
    
    # Final save
    if rank == 0:
        save_checkpoint(
            model.module if args.distributed else model,
            vae_optimizer,
            dit_optimizer,
            config['training']['num_epochs'] - 1,
            config,
            output_dir / 'final_model.pt',
            is_best=False
        )
        
        logging.info(f"Training complete! Models saved to {output_dir}")


if __name__ == '__main__':
    main()