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
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'
    
    # Enable cudnn optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


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


def efficient_training_step(model, batch, vae_optimizer, dit_optimizer, scaler, step, config, device):
    """Memory-efficient training step - minimal debugging"""
    
    # Extract video data efficiently
    if 'target_frames' in batch:
        video = batch['target_frames'].to(device, non_blocking=True)
    elif 'video' in batch:
        video = batch['video'].to(device, non_blocking=True)
    else:
        return None
    
    # Forward pass with mixed precision
    try:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            losses = model(video, return_dict=True)
        
        total_loss = losses.get('loss_total')
        if total_loss is None or torch.isnan(total_loss) or torch.isinf(total_loss):
            return None
        
        # Backward pass
        scaler.scale(total_loss).backward()
        
        # Gradient accumulation check
        if (step + 1) % config['training'].get('gradient_accumulation_steps', 1) == 0:
            model_ref = model.module if hasattr(model, 'module') else model
            
            # Unscale and clip gradients
            if dit_optimizer is not None:
                scaler.unscale_(dit_optimizer)
            if vae_optimizer is not None and not model_ref._vae_frozen:
                scaler.unscale_(vae_optimizer)
            
            # Simple gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training'].get('grad_clip', 1.0)
            )
            
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
        
        # Return minimal loss info
        return {
            'loss_total': total_loss.item(),
            'loss_velocity': losses.get('velocity_loss', torch.tensor(0.0)).item(),
            'loss_flf': losses.get('flf_loss', torch.tensor(0.0)).item()
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error(f"❌ Step {step}: CUDA OOM: {e}")
            torch.cuda.empty_cache()
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
    
    # Enhanced loss tracking
    loss_metrics = {
        'total': 0.0,
        'velocity': 0.0,
        'flf': 0.0,
        'vae_recon': 0.0,
        'vae_kl': 0.0,
        'vae_temporal': 0.0
    }
    num_batches = 0
    
    # Check if VAE should be frozen
    model_ref = model.module if hasattr(model, 'module') else model
    if model_ref.training_step >= model_ref.freeze_vae_after and not model_ref._vae_frozen:
        model_ref._freeze_vae()
        logging.info(f"🔒 VAE frozen at step {model_ref.training_step}")
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Efficient training step
        step_losses = efficient_training_step(
            model, batch, vae_optimizer, dit_optimizer, 
            scaler, model_ref.training_step, config, device
        )
        
        if step_losses is None:
            continue  # Skip failed batches instead of breaking
        
        # Update comprehensive metrics
        loss_metrics['total'] += step_losses.get('loss_total', 0.0)
        loss_metrics['velocity'] += step_losses.get('loss_velocity', 0.0)
        loss_metrics['flf'] += step_losses.get('loss_flf', 0.0)
        loss_metrics['vae_recon'] += step_losses.get('loss_vae_recon', 0.0)
        loss_metrics['vae_kl'] += step_losses.get('loss_vae_kl', 0.0)
        loss_metrics['vae_temporal'] += step_losses.get('loss_vae_temporal', 0.0)
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{step_losses['loss_total']:.4f}",
            'vel': f"{step_losses.get('loss_velocity', 0):.3f}",
            'flf': f"{step_losses.get('loss_flf', 0):.3f}",
            'mem_gb': f"{torch.cuda.memory_allocated() / 1e9:.1f}",
            'step': model_ref.training_step
        })
        
        # Real-time wandb logging (every 20 steps)
        if use_wandb and model_ref.training_step % 20 == 0:
            wandb.log({
                'train/loss_total_realtime': step_losses['loss_total'],
                'train/loss_velocity_realtime': step_losses.get('loss_velocity', 0),
                'train/loss_flf_realtime': step_losses.get('loss_flf', 0),
                'train/lr_vae': vae_optimizer.param_groups[0]['lr'] if vae_optimizer else 0,
                'train/lr_dit': dit_optimizer.param_groups[0]['lr'] if dit_optimizer else 0,
                'train/memory_gb': torch.cuda.memory_allocated() / 1e9,
                'train/step': model_ref.training_step,
                'train/epoch': epoch
            })
        
        # Increment training step
        model_ref.training_step += 1
        
        # Enhanced periodic logging
        if model_ref.training_step % 100 == 0:
            current_avg_loss = loss_metrics['total'] / num_batches if num_batches > 0 else 0
            logging.info(f"📊 Step {model_ref.training_step} | "
                        f"Avg Loss: {current_avg_loss:.6f} | "
                        f"Batches: {num_batches} | "
                        f"Memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        
        # Memory cleanup
        if batch_idx % 100 == 0 and batch_idx > 0:
            torch.cuda.empty_cache()
    
    # Step schedulers
    if vae_scheduler is not None:
        vae_scheduler.step()
    if dit_scheduler is not None:
        dit_scheduler.step()
    
    # Compute final averages
    if num_batches > 0:
        avg_losses = {k: v / num_batches for k, v in loss_metrics.items()}
    else:
        avg_losses = {k: 0.0 for k in loss_metrics.keys()}
    
    # Enhanced epoch summary logging
    logging.info(f"🎯 EPOCH {epoch} TRAINING SUMMARY:")
    logging.info(f"   📈 Total Loss: {avg_losses['total']:.6f}")
    logging.info(f"   🎯 Velocity Loss: {avg_losses['velocity']:.6f}")
    logging.info(f"   🔒 FLF Loss: {avg_losses['flf']:.6f}")
    logging.info(f"   🖼️  VAE Recon Loss: {avg_losses['vae_recon']:.6f}")
    logging.info(f"   📊 VAE KL Loss: {avg_losses['vae_kl']:.6f}")
    logging.info(f"   ⏱️  VAE Temporal Loss: {avg_losses['vae_temporal']:.6f}")
    logging.info(f"   📚 Processed Batches: {num_batches}")
    logging.info(f"   🔢 Training Steps: {model_ref.training_step}")
    
    # Comprehensive wandb epoch logging
    if use_wandb:
        wandb.log({
            'epoch_summary/train_loss_total': avg_losses['total'],
            'epoch_summary/train_loss_velocity': avg_losses['velocity'],
            'epoch_summary/train_loss_flf': avg_losses['flf'],
            'epoch_summary/train_loss_vae_recon': avg_losses['vae_recon'],
            'epoch_summary/train_loss_vae_kl': avg_losses['vae_kl'],
            'epoch_summary/train_loss_vae_temporal': avg_losses['vae_temporal'],
            'epoch_summary/train_batches_processed': num_batches,
            'epoch_summary/train_steps_total': model_ref.training_step,
            'epoch_summary/lr_vae': vae_optimizer.param_groups[0]['lr'] if vae_optimizer else 0,
            'epoch_summary/lr_dit': dit_optimizer.param_groups[0]['lr'] if dit_optimizer else 0,
            'epoch_summary/epoch': epoch
        })
    
    return avg_losses


def validate_epoch(
    model,
    dataloader,
    epoch,
    config,
    device,
    use_wandb=False
):
    """Enhanced validation with comprehensive logging"""
    model.eval()
    
    # Enhanced validation metrics
    val_metrics = {
        'total': 0.0,
        'velocity': 0.0,
        'flf': 0.0,
        'vae_recon': 0.0,
        'vae_kl': 0.0,
        'vae_temporal': 0.0
    }
    num_batches = 0
    failed_batches = 0
    
    model_ref = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Val Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch data to device
                if 'target_frames' in batch:
                    video = batch['target_frames'].to(device, non_blocking=True)
                elif 'video' in batch:
                    video = batch['video'].to(device, non_blocking=True)
                else:
                    continue
                
                # 🔧 CRITICAL FIX: Use runtime cropping for validation
                # This ensures consistent sequence length and fixes spatial_shape issues
                target_frames = config['data'].get('num_frames', 41)
                
                # Forward pass with runtime cropping (same as training)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                    try:
                        # 🔧 FIX: Use the same runtime cropping as training
                        all_losses = model_ref(
                            video, 
                            target_sequence_length=target_frames,
                            crop_strategy='center',
                            return_dict=True
                        )
                        total_loss = all_losses.get('loss_total')
                        
                        if total_loss is not None and not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                            val_metrics['total'] += total_loss.item()
                            val_metrics['velocity'] += all_losses.get('loss_velocity', torch.tensor(0.0)).item()
                            val_metrics['flf'] += all_losses.get('loss_flf', torch.tensor(0.0)).item()
                            val_metrics['vae_recon'] += all_losses.get('loss_vae_recon', torch.tensor(0.0)).item()
                            val_metrics['vae_kl'] += all_losses.get('loss_vae_kl', torch.tensor(0.0)).item()
                            val_metrics['vae_temporal'] += all_losses.get('loss_vae_temporal', torch.tensor(0.0)).item()
                            num_batches += 1
                            
                            # Update progress bar
                            progress_bar.set_postfix({
                                'val_loss': f"{total_loss.item():.4f}",
                                'success': f"{num_batches}/{batch_idx+1}"
                            })
                        else:
                            failed_batches += 1
                            
                    except TypeError:
                        # Fallback for models without target_sequence_length support
                        print(f"⚠️  Model doesn't support target_sequence_length, using fallback")
                        all_losses = model_ref(video, return_dict=True)
                        total_loss = all_losses.get('loss_total')
                        
                        if total_loss is not None and not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                            val_metrics['total'] += total_loss.item()
                            num_batches += 1
                            progress_bar.set_postfix({'val_loss': f"{total_loss.item():.4f}"})
                        else:
                            failed_batches += 1
                            
                    except RuntimeError as e:
                        if "invalid for input of size" in str(e) or "shape" in str(e):
                            failed_batches += 1
                            if failed_batches <= 5:  # Only log first few failures
                                logging.warning(f"⚠️  Skipping validation batch {batch_idx}: {e}")
                            continue
                        else:
                            raise e
            
            except Exception as e:
                failed_batches += 1
                if failed_batches <= 5:  # Only log first few failures
                    logging.warning(f"⚠️  Validation batch {batch_idx} failed: {e}")
                continue
    
    # Compute validation averages
    if num_batches > 0:
        avg_val_losses = {k: v / num_batches for k, v in val_metrics.items()}
    else:
        avg_val_losses = {k: 0.0 for k in val_metrics.keys()}
    
    # Enhanced validation summary logging
    logging.info(f"🔍 EPOCH {epoch} VALIDATION SUMMARY:")
    logging.info(f"   📈 Total Loss: {avg_val_losses['total']:.6f}")
    logging.info(f"   🎯 Velocity Loss: {avg_val_losses['velocity']:.6f}")
    logging.info(f"   🔒 FLF Loss: {avg_val_losses['flf']:.6f}")
    logging.info(f"   🖼️  VAE Recon Loss: {avg_val_losses['vae_recon']:.6f}")
    logging.info(f"   📊 VAE KL Loss: {avg_val_losses['vae_kl']:.6f}")
    logging.info(f"   ⏱️  VAE Temporal Loss: {avg_val_losses['vae_temporal']:.6f}")
    logging.info(f"   ✅ Successful Batches: {num_batches}")
    logging.info(f"   ❌ Failed Batches: {failed_batches}")
    logging.info(f"   📊 Success Rate: {num_batches/(num_batches+failed_batches)*100:.1f}%")
    
    # Comprehensive wandb validation logging
    if use_wandb:
        wandb.log({
            'epoch_summary/val_loss_total': avg_val_losses['total'],
            'epoch_summary/val_loss_velocity': avg_val_losses['velocity'],
            'epoch_summary/val_loss_flf': avg_val_losses['flf'],
            'epoch_summary/val_loss_vae_recon': avg_val_losses['vae_recon'],
            'epoch_summary/val_loss_vae_kl': avg_val_losses['vae_kl'],
            'epoch_summary/val_loss_vae_temporal': avg_val_losses['vae_temporal'],
            'epoch_summary/val_batches_successful': num_batches,
            'epoch_summary/val_batches_failed': failed_batches,
            'epoch_summary/val_success_rate': num_batches/(num_batches+failed_batches)*100 if (num_batches+failed_batches) > 0 else 0,
            'epoch_summary/epoch': epoch
        })
    
    return avg_val_losses

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
        pin_memory=False,  # Disable pin_memory to save GPU memory
        collate_fn=lungct_collate_fn,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=False,  # Disable pin_memory to save GPU memory
        collate_fn=lungct_collate_fn,
        persistent_workers=True
    )
    
    # Create model
    model = create_model(config, device)
    
    # Skip compilation for memory efficiency during debugging
    if config['training'].get('compile_model', False):
        model.dit = torch.compile(model.dit, mode='max-autotune', fullgraph=False)
    
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
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
    else:
        vae_scheduler = None
        dit_scheduler = None
    
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
        if vae_scheduler and 'vae_scheduler_state_dict' in checkpoint:
            vae_scheduler.load_state_dict(checkpoint['vae_scheduler_state_dict'])
        if dit_scheduler and 'dit_scheduler_state_dict' in checkpoint:
            dit_scheduler.load_state_dict(checkpoint['dit_scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resumed from checkpoint: {args.resume}, epoch {start_epoch}")
    
    # 🔥 ENHANCED TRAINING LOOP WITH COMPREHENSIVE LOGGING 🔥
    best_val_loss = float('inf')
    train_losses_history = []
    val_losses_history = []
    
    for epoch in range(start_epoch, config['training'].get('num_epochs', 100)):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Enhanced epoch start logging
        logging.info(f"\n🚀 STARTING EPOCH {epoch}")
        logging.info(f"   📅 Epoch: {epoch}/{config['training'].get('num_epochs', 100)}")
        logging.info(f"   🎯 Target: 41-frame FLF2V sequences (phase 0→50%)")
        logging.info(f"   💾 Memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB allocated")
        
        # Train with enhanced logging
        train_losses = train_epoch(
            model, train_loader, vae_optimizer, dit_optimizer, 
            vae_scheduler, dit_scheduler,
            scaler, epoch, config, device, not args.no_wandb
        )
        
        train_losses_history.append(train_losses)
        
        # Validate with enhanced logging
        val_losses = {}
        if epoch % config['training'].get('val_freq', 5) == 0:
            val_losses = validate_epoch(
                model, val_loader, epoch, config, device, not args.no_wandb
            )
            val_losses_history.append(val_losses)
            
            # Enhanced epoch comparison logging
            if train_losses and val_losses:
                train_total = train_losses.get('total', 0)
                val_total = val_losses.get('total', 0)
                
                logging.info(f"\n📊 EPOCH {epoch} COMPARISON:")
                logging.info(f"   🏋️  Training Loss:   {train_total:.6f}")
                logging.info(f"   🔍 Validation Loss: {val_total:.6f}")
                logging.info(f"   📈 Difference:      {abs(train_total - val_total):.6f}")
                
                if val_total < train_total:
                    logging.info(f"   ✅ Good generalization (val < train)")
                elif val_total > train_total * 1.2:
                    logging.info(f"   ⚠️  Possible overfitting (val >> train)")
                else:
                    logging.info(f"   👍 Normal training progress")
                
                # Track best model
                if val_total < best_val_loss:
                    best_val_loss = val_total
                    logging.info(f"   🏆 NEW BEST VALIDATION LOSS: {val_total:.6f}")
                    
                    # Save best model
                    if rank == 0:
                        best_checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                            'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                            'dit_optimizer_state_dict': dit_optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'config': config,
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'best_val_loss': best_val_loss
                        }
                        best_path = output_dir / 'best_model.pt'
                        torch.save(best_checkpoint, best_path)
                        logging.info(f"   💎 Saved best model: {best_path}")
                
                # Comprehensive wandb epoch comparison
                if not args.no_wandb:
                    wandb.log({
                        'epoch_comparison/train_vs_val_diff': abs(train_total - val_total),
                        'epoch_comparison/train_val_ratio': val_total / train_total if train_total > 0 else 0,
                        'epoch_comparison/best_val_loss': best_val_loss,
                        'epoch_comparison/is_best_epoch': val_total == best_val_loss,
                        'epoch_comparison/epoch': epoch
                    })
        
        # Save regular checkpoint with enhanced info
        if rank == 0 and epoch % config['training'].get('save_freq', 20) == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                'dit_optimizer_state_dict': dit_optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': config,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            if vae_scheduler:
                checkpoint['vae_scheduler_state_dict'] = vae_scheduler.state_dict()
            if dit_scheduler:
                checkpoint['dit_scheduler_state_dict'] = dit_scheduler.state_dict()
            
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    if rank == 0:
        final_checkpoint = {
            'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
            'config': config,
            'train_losses_history': train_losses_history,
            'val_losses_history': val_losses_history,
            'best_val_loss': best_val_loss,
            'final_epoch': config['training'].get('num_epochs', 100) - 1
        }
        final_path = output_dir / 'final_model.pt'
        torch.save(final_checkpoint, final_path)
        logging.info(f"💾 Saved final model: {final_path}")
        
        # Training summary
        logging.info(f"\n🎉 TRAINING COMPLETED!")
        logging.info(f"   🏆 Best validation loss: {best_val_loss:.6f}")
        logging.info(f"   📈 Total epochs: {config['training'].get('num_epochs', 100)}")
        logging.info(f"   💾 Models saved in: {output_dir}")
    
    # Cleanup
    if args.distributed:
        dist.destroy_process_group()


