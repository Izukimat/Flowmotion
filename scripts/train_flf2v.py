#!/usr/bin/env python3
"""
Training script for Lung CT FLF2V model - Fixed version with debugging
Addresses all critical issues: proper imports, loss handling, schedulers, debugging
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


def setup_cuda_optimizations():
    """Setup CUDA-specific optimizations"""
    # Enable TensorFloat-32 (faster on H100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Optimize memory allocation
    torch.cuda.empty_cache()
    
    # Set memory allocation strategy
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Enable cudnn optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Faster but less reproducible


def check_model_health(model, step):
    """Check for NaN/Inf in model parameters"""
    nan_params = []
    inf_params = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)
    
    if nan_params or inf_params:
        logging.error(f"❌ Step {step}: Model corruption detected!")
        logging.error(f"   NaN params: {nan_params}")
        logging.error(f"   Inf params: {inf_params}")
        return False
    return True


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
    flow_matching = create_flow_matching_model({'config': config['model']['flow_matching_config']})
    
    # Create full model with loss weights
    loss_weights = {
        'velocity_weight': config['training'].get('velocity_weight', 1.0),
        'flf_weight': config['training'].get('flf_weight', 0.1),
        'vae_recon_weight': config['training'].get('vae_recon_weight', 1.0),
        'vae_kl_weight': config['training'].get('vae_kl_weight', 0.01),
        'vae_temporal_weight': config['training'].get('vae_temporal_weight', 0.1)
    }
    
    model = LungCTFLF2V(
        vae=vae,
        dit=dit,
        flow_matching=flow_matching,
        freeze_vae_after=config['model']['freeze_vae_after'],
        loss_weights=loss_weights
    )
    
    return model.to(device)


def debug_training_step(model, batch, vae_optimizer, dit_optimizer, scaler, step, config, device):
    """Debug version of training step with comprehensive checks"""
    
    # Extract video data
    if 'video' in batch:
        video = batch['video'].to(device, memory_format=torch.channels_last)
    elif 'target_frames' in batch:
        video = batch['target_frames'].to(device, memory_format=torch.channels_last)
    else:
        raise KeyError("Batch must contain either 'video' or 'target_frames'")
    
    # Check input data
    if torch.isnan(video).any() or torch.isinf(video).any():
        logging.error(f"❌ Step {step}: NaN/Inf in input data!")
        return None
    
    # Forward pass with debugging
    try:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            losses = model(video, return_dict=True)
        
        # Check all loss components
        total_loss = None
        valid_losses = {}
        
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    logging.error(f"  ❌ {key}: {value} (CORRUPTED)")
                    return None
                else:
                    valid_losses[key] = value.item()
                    if key == 'loss_total':
                        total_loss = value
        
        if total_loss is None or total_loss.item() == 0:
            logging.error(f"❌ Step {step}: Invalid total loss: {total_loss}")
            return None
        
        # Log losses every 10 steps
        if step % 10 == 0:
            loss_str = ", ".join([f"{k}: {v:.6f}" for k, v in valid_losses.items()])
            logging.info(f"Step {step} - {loss_str}")
        
        # Backward pass
        scaler.scale(total_loss).backward()
        
        # Gradient accumulation check
        if (step + 1) % config['training']['gradient_accumulation_steps'] == 0:
            # Unscale gradients for clipping
            model_ref = model.module if hasattr(model, 'module') else model
            
            if dit_optimizer is not None:
                scaler.unscale_(dit_optimizer)
            if vae_optimizer is not None and not model_ref._vae_frozen:
                scaler.unscale_(vae_optimizer)
            
            # Check gradients before clipping
            total_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logging.error(f"❌ Step {step}: NaN/Inf gradient in {name}")
                        return None
            
            total_norm = total_norm ** (1. / 2)
            
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['grad_clip']
            )
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logging.error(f"❌ Step {step}: Invalid gradient norm: {grad_norm}")
                return None
            
            # Log gradient norm every 50 steps
            if step % 50 == 0:
                logging.info(f"Step {step}: Gradient norm: {grad_norm:.6f}")
            
            # Optimizer steps
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
            
            # Check model health after update
            if not check_model_health(model, step):
                return None
        
        return valid_losses
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error(f"❌ Step {step}: CUDA OOM: {e}")
            torch.cuda.empty_cache()
        else:
            logging.error(f"❌ Step {step}: Runtime error: {e}")
        return None
    except Exception as e:
        logging.error(f"❌ Step {step}: Exception during training: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    """Memory-optimized training epoch with debugging"""
    model.train()
    losses = defaultdict(list)
    
    # Check if VAE should be frozen
    model_ref = model.module if hasattr(model, 'module') else model
    if model_ref.training_step >= model_ref.freeze_vae_after:
        model_ref._freeze_vae()
        logging.info(f"VAE frozen at step {model_ref.training_step}")
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Debug training step
        step_losses = debug_training_step(
            model, batch, vae_optimizer, dit_optimizer, 
            scaler, model_ref.training_step, config, device
        )
        
        if step_losses is None:
            logging.error(f"❌ Training failed at step {model_ref.training_step}. Stopping epoch.")
            break
        
        # Record losses
        for key, value in step_losses.items():
            losses[key].append(value)
        
        # Update progress bar with current loss
        current_loss = step_losses.get('loss_total', 0)
        progress_bar.set_postfix({
            'loss': f"{current_loss:.6f}",
            'mem_gb': torch.cuda.memory_allocated() / 1e9
        })
        
        # Increment training step
        model_ref.training_step += 1
        
        # Memory cleanup every 50 steps
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            
        # Emergency break if loss becomes 0 consistently
        if len(losses['loss_total']) > 10:
            recent_losses = losses['loss_total'][-10:]
            if all(l < 1e-6 for l in recent_losses):
                logging.error("❌ Loss has been 0 for 10 consecutive steps. Stopping training.")
                break
    
    # Step schedulers
    if vae_scheduler is not None:
        vae_scheduler.step()
    if dit_scheduler is not None:
        dit_scheduler.step()
    
    # Average losses
    avg_losses = {k: np.mean(v) for k, v in losses.items() if v}
    
    # Log to wandb
    if use_wandb and avg_losses:
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
    """Validate one epoch"""
    model.eval()
    losses = defaultdict(list)
    
    model_ref = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Val Epoch {epoch}"):
            try:
                # Move batch data to device
                if 'video' in batch:
                    video = batch['video'].to(device, memory_format=torch.channels_last)
                elif 'target_frames' in batch:
                    video = batch['target_frames'].to(device, memory_format=torch.channels_last)
                else:
                    continue
                
                # Forward pass
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    all_losses = model_ref(video, return_dict=True)
                    
                    # Record losses
                    for key, value in all_losses.items():
                        if isinstance(value, torch.Tensor) and not (torch.isnan(value) or torch.isinf(value)):
                            losses[key].append(value.item())
            
            except Exception as e:
                logging.warning(f"Validation batch failed: {e}")
                continue
    
    # Average losses
    avg_losses = {k: np.mean(v) for k, v in losses.items() if v}
    
    # Log to wandb
    if use_wandb and avg_losses:
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
    
    # Setup CUDA optimizations
    setup_cuda_optimizations()
    
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
    
    logging.info(f"🚀 Starting training with config: {args.config}")
    logging.info(f"📊 Batch size: {config['training']['batch_size']}")
    logging.info(f"🔧 Latent channels: {config['model']['latent_channels']}")
    
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
        augment=config['data'].get('augmentation', {}).get('enable_augmentation', True),
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
    
    logging.info(f"📚 Loaded {len(train_dataset)} train samples")
    logging.info(f"📚 Loaded {len(val_dataset)} val samples")
    
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
        collate_fn=lungct_collate_fn,
        persistent_workers=True,
        drop_last=True  # For compilation stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
        collate_fn=lungct_collate_fn,
        persistent_workers=True
    )
    
    # Create model
    model = create_model(config, device)
    
    # Apply torch.compile if enabled
    if config['training'].get('compile_model', False):
        logging.info("🚀 Compiling model components for speed...")
        try:
            model.dit = torch.compile(
                model.dit,
                mode='default',  # Start conservative for stability
                fullgraph=False
            )
            logging.info("✅ DiT compiled successfully")
        except Exception as e:
            logging.warning(f"⚠️ DiT compilation failed: {e}, proceeding without compilation")
    
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
    scheduler_type = config['training'].get('scheduler_type', 'cosine')
    if scheduler_type == 'cosine':
        vae_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            vae_optimizer, 
            T_max=config['training'].get('num_epochs', 100)
        )
        dit_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            dit_optimizer,
            T_max=config['training'].get('num_epochs', 100)
        )
    else:  # linear
        vae_scheduler = optim.lr_scheduler.LinearLR(
            vae_optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config['training'].get('num_epochs', 100)
        )
        dit_scheduler = optim.lr_scheduler.LinearLR(
            dit_optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config['training'].get('num_epochs', 100)
        )
    
    # Setup mixed precision
    scaler = GradScaler(enabled=config['training'].get('mixed_precision', True))
    
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
        
        # Check if training failed
        if not train_losses or train_losses.get('loss_total', 0) == 0:
            logging.error("❌ Training failed. Exiting.")
            break
        
        # Validate
        if epoch % config['training'].get('val_freq', 5) == 0:
            val_losses = validate_epoch(
                model, val_loader, epoch, config, device, not args.no_wandb
            )
            
            if val_losses:
                logging.info(f"Epoch {epoch}: Train Loss: {train_losses.get('loss_total', 0):.6f}, "
                            f"Val Loss: {val_losses.get('loss_total', 0):.6f}")
            else:
                logging.info(f"Epoch {epoch}: Train Loss: {train_losses.get('loss_total', 0):.6f}")
        
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
            logging.info(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    if rank == 0:
        final_path = output_dir / 'final_model.pt'
        torch.save({
            'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
            'config': config
        }, final_path)
        logging.info(f"💾 Saved final model: {final_path}")
    
    # Cleanup
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()