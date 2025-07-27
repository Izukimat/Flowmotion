#!/usr/bin/env python3
"""
Training script for Lung CT FLF2V model - Fixed version
Addresses all critical issues: proper imports, loss handling, schedulers, etc.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import wandb
import yaml

from src.flf2v import LungCTDataset, LungCTFLF2V, lungct_collate_fn
from src.flf2v.lungct_vae import LungCTVAE, VAELoss
from src.flf2v.lungct_dit import create_dit_model
from src.flf2v.lungct_flow_matching import create_flow_matching_model, FlowMatchingConfig


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
    
    # Create Flow Matching - Fix: proper config handling and no DiT duplication
    # fm_config = FlowMatchingConfig(**)
    flow_matching = create_flow_matching_model({'config': config['model']['flow_matching_config']})
    
    # Create full model
    model = LungCTFLF2V(
        vae=vae,
        dit=dit,
        flow_matching=flow_matching,
        freeze_vae_after=config['model']['freeze_vae_after']
    )
    
    return model.to(device)


def get_loss_weights(config: Dict) -> Dict[str, float]:
    """Get loss weights with defaults"""
    training_config = config.get('training', {})
    return {
        'velocity_weight': training_config.get('velocity_weight', 1.0),
        'flf_weight': training_config.get('flf_weight', 0.1),
        'vae_recon_weight': training_config.get('vae_recon_weight', 1.0),
        'vae_kl_weight': training_config.get('vae_kl_weight', 0.01),
        'vae_temporal_weight': training_config.get('vae_temporal_weight', 0.1)
    }


def calculate_weighted_loss(loss_dict: Dict, weights: Dict[str, float], device: str) -> torch.Tensor:
    """Calculate weighted total loss - only sum loss keys"""
    total_loss = torch.tensor(0.0, device=device)
    
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

def train_epoch(
    model: LungCTFLF2V,
    dataloader: DataLoader,
    vae_optimizer: optim.Optimizer,
    dit_optimizer: optim.Optimizer,
    vae_scheduler: Any,
    dit_scheduler: Any,
    scaler: GradScaler,
    epoch: int,
    config: Dict,
    device: str,
    use_wandb: bool = False
) -> Dict[str, float]:
    """Memory-optimized training epoch"""
    model.train()
    losses = defaultdict(list)
    
    # Check if VAE should be frozen
    model_ref = model.module if hasattr(model, 'module') else model
    if model_ref.training_step >= model_ref.freeze_vae_after:
        model_ref._freeze_vae()
        vae_optimizer = None  # No need to update frozen params
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch data to device
        if 'video' in batch:
            video = batch['video'].to(device)  # [B, 1, T, H, W]
        elif 'target_frames' in batch:
            video = batch['target_frames'].to(device)
        else:
            raise KeyError("Batch must contain either 'video' or 'target_frames'")
        
        # Extract first and last frames from the video sequence
        B, C, T, H, W = video.shape
        
        # Forward pass with autocast
        with torch.amp.autocast():
            # Get all losses from the model
            all_losses = model_ref(
                video=video,
                return_dict=True
            )
            
            # Extract individual losses
            total_loss = all_losses['total']
            
            # Record losses
            for key, value in all_losses.items():
                if isinstance(value, torch.Tensor):
                    losses[key].append(value.item())
        
        # Backward pass
        scaler.scale(total_loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config['training']['gradient_accumulation_steps'] == 0:
            # Unscale gradients
            if dit_optimizer is not None:
                scaler.unscale_(dit_optimizer)
            if vae_optimizer is not None and not model_ref._vae_frozen:
                scaler.unscale_(vae_optimizer)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['grad_clip']
            )
            
            # Update weights
            if dit_optimizer is not None:
                scaler.step(dit_optimizer)
            if vae_optimizer is not None and not model_ref._vae_frozen:
                scaler.step(vae_optimizer)
            
            scaler.update()
            
            # Zero gradients
            if dit_optimizer is not None:
                dit_optimizer.zero_grad(set_to_none=True)
            if vae_optimizer is not None:
                vae_optimizer.zero_grad(set_to_none=True)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': np.mean(losses['total'][-10:]) if losses['total'] else 0,
            'mem_gb': torch.cuda.max_memory_allocated() / 1e9
        })
        
        # Increment training step
        model_ref.training_step += 1
        
        # Clear cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Step schedulers
    if vae_scheduler is not None:
        vae_scheduler.step()
    if dit_scheduler is not None:
        dit_scheduler.step()
    
    # Average losses
    avg_losses = {k: np.mean(v) for k, v in losses.items() if v}
    
    # Log to wandb
    if use_wandb:
        wandb.log({f'train/{k}': v for k, v in avg_losses.items()}, step=epoch)
        wandb.log({
            'train/lr_vae': vae_optimizer.param_groups[0]['lr'] if vae_optimizer else 0,
            'train/lr_dit': dit_optimizer.param_groups[0]['lr'] if dit_optimizer else 0,
        }, step=epoch)
    
    return avg_losses


def validate_epoch(
    model,
    dataloader,
    epoch,
    config,
    device,
    use_wandb=False
):
    """Validate one epoch - Fixed to handle actual data structure"""
    model.eval()
    losses = defaultdict(list)
    
    model_ref = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Val Epoch {epoch}"):
            # Move batch data to device
            if 'video' in batch:
                video = batch['video'].to(device)
            elif 'target_frames' in batch:
                video = batch['target_frames'].to(device)
            else:
                raise KeyError("Batch must contain either 'video' or 'target_frames'")
            
            B, C, T, H, W = video.shape
            
            # Forward pass
            with torch.amp.autocast():
                all_losses = model_ref(
                    video=video,
                    return_dict=True
                )
                
                # Record losses
                for key, value in all_losses.items():
                    if isinstance(value, torch.Tensor):
                        losses[key].append(value.item())
    
    # Average losses
    avg_losses = {k: np.mean(v) for k, v in losses.items() if v}
    
    # Log to wandb
    if use_wandb:
        wandb.log({f'val/{k}': v for k, v in avg_losses.items()}, step=epoch)
    
    return avg_losses


def main():
    """Main training function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Lung CT FLF2V model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--csv-file', type=str, required=True, help='CSV metadata file')
    parser.add_argument('--data-root', type=str, 
                      default='/home/ubuntu/azureblob/4D-Lung-Interpolated/data/',
                      help='Root data directory')
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
    output_dir = Path(args.output_dir)
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
        csv_file=args.csv_file,
        split='train',
        data_root=args.data_root,
        augment=True,
        normalize=True,
        load_target_frames=True
    )
    
    val_dataset = LungCTDataset(
        csv_file=args.csv_file,
        split='val',
        data_root=args.data_root,
        augment=False,
        normalize=True,
        load_target_frames=True
    )
    
    # Create dataloaders
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
        collate_fn=lungct_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
        collate_fn=lungct_collate_fn
    )
    
    # Create model
    model = create_model(config, device)
    if args.distributed:
        model = DDP(model, device_ids=[local_rank])
    
    # Create optimizers
    model_ref = model.module if args.distributed else model
    vae_optimizer = optim.AdamW(
        model_ref.vae.parameters(),
        lr=config['training'].get('vae_lr', 1e-4),
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    
    dit_params = list(model_ref.dit.parameters()) + list(model_ref.flow_matching.parameters())
    dit_optimizer = optim.AdamW(
        dit_params,
        lr=config['training'].get('dit_lr', 1e-4),
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    
    # Add LR schedulers
    vae_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        vae_optimizer, 
        T_max=config['training'].get('num_epochs', 100)
    )
    dit_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        dit_optimizer,
        T_max=config['training'].get('num_epochs', 100)
    )
    
    # Setup mixed precision
    scaler = GradScaler()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if args.distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        dit_optimizer.load_state_dict(checkpoint['dit_optimizer_state_dict'])
        vae_scheduler.load_state_dict(checkpoint['vae_scheduler_state_dict'])
        dit_scheduler.load_state_dict(checkpoint['dit_scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resumed from checkpoint: {args.resume}, epoch {start_epoch}")
    
    # Training loop
    val_losses = {}  # Initialize val_losses
    
    for epoch in range(start_epoch, config['training'].get('num_epochs', 100)):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_losses = train_epoch(
            model, train_loader, vae_optimizer, dit_optimizer, 
            vae_scheduler, dit_scheduler,
            scaler, epoch, config, device, not args.no_wandb
        )
        
        # Validate
        if epoch % config['training'].get('val_freq', 5) == 0:
            val_losses = validate_epoch(
                model, val_loader, epoch, config, device, not args.no_wandb
            )
            
            logging.info(f"Epoch {epoch}: Train Loss: {train_losses['total']:.4f}, "
                        f"Val Loss: {val_losses['total']:.4f}")
        
        # Save checkpoint
        if rank == 0 and epoch % config['training'].get('save_freq', 10) == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                'dit_optimizer_state_dict': dit_optimizer.state_dict(),
                'vae_scheduler_state_dict': vae_scheduler.state_dict(),
                'dit_scheduler_state_dict': dit_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': config,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    if rank == 0:
        final_path = output_dir / 'final_model.pt'
        torch.save({
            'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
            'config': config
        }, final_path)
        logging.info(f"Saved final model: {final_path}")
    
    # Cleanup
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()