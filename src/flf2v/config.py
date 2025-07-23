"""
Configuration and example usage for Lung CT FLF2V
"""
import torch
import yaml
from pathlib import Path
from typing import Dict
import argparse


# Default configuration
DEFAULT_CONFIG = {
    # Model architecture
    'model': {
        # VAE settings
        'vae_base_model': 'medvae_4_1_3d',
        'latent_channels': 8,  # Increased from MedVAE's 1 channel
        'vae_temporal_weight': 0.1,
        'freeze_vae_after': 5000,  # Freeze VAE after this many steps
        
        # DiT settings
        'dit_config': {
            'latent_channels': 8,
            'latent_size': (5, 16, 16),  # After 8x compression of 40x128x128
            'hidden_dim': 768,  # ~0.6B params with 24 layers
            'depth': 24,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'use_rope': True
        },
        
        # Flow matching settings
        'flow_matching_config': {
            'num_sampling_steps': 50,
            'sigma_min': 1e-5,
            'use_ode_solver': 'euler',
            'time_sampling': 'uniform',
            'loss_weighting': 'uniform',
            'interpolation_method': 'optimal_transport'
        }
    },
    
    # Training settings
    'training': {
        # Optimization
        'vae_lr': 1e-4,
        'dit_lr': 1e-4,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        
        # Loss weights
        'velocity_weight': 1.0,
        'flf_weight': 0.1,
        'vae_recon_weight': 1.0,
        'vae_kl_weight': 0.01,
        'vae_temporal_weight': 0.1,
        
        # Schedule
        'scheduler_T0': 1000,
        'num_epochs': 100,
        'val_freq': 5,
        'save_freq': 10,
        
        # Data
        'batch_size': 2,  # Small batch for V100
        'num_workers': 4,
        'num_frames': 40,
        'image_size': 128,
        
        # Logging
        'use_wandb': True,
        'save_dir': './checkpoints',
        'run_name': 'lungct_flf2v_baseline'
    },
    
    # Data settings
    'data': {
        'data_dir': './data/lung_ct',
        'slice_indices': [10, 25, 40],  # Apex, middle, base
        'augmentation': {
            'intensity_scale': 0.1,
            'intensity_shift': 0.05,
            'rotation_degrees': 5,
            'noise_std': 0.01
        }
    },
    
    # Inference settings
    'inference': {
        'guidance_scale': 1.0,
        'num_sampling_steps': 50,
        'batch_size': 4
    }
}


def load_config(config_path: str = None) -> Dict:
    """Load configuration from file or use defaults"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Merge with defaults
        return merge_configs(DEFAULT_CONFIG, config)
    return DEFAULT_CONFIG.copy()


def merge_configs(default: Dict, custom: Dict) -> Dict:
    """Recursively merge configurations"""
    result = default.copy()
    for key, value in custom.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: Dict, path: str):
    """Save configuration to YAML file"""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# Example training script
def train_example():
    """Example training script"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Lung CT FLF2V model')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    
    # Import modules (assuming they're in the same package)
    from .model import create_flf2v_model, LungCTDataset, FLF2VTrainer
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    
    # Create data augmentation pipeline
    transform = transforms.Compose([
        transforms.RandomRotation(config['data']['augmentation']['rotation_degrees']),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        # Add intensity augmentation
        lambda x: x + torch.randn_like(x) * config['data']['augmentation']['noise_std'],
        # Normalize to [-1, 1] for flow matching
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = LungCTDataset(
        data_dir=config['data']['data_dir'],
        split='train',
        num_frames=config['training']['num_frames'],
        slice_indices=config['data']['slice_indices'],
        transform=transform
    )
    
    val_dataset = LungCTDataset(
        data_dir=config['data']['data_dir'],
        split='val',
        num_frames=config['training']['num_frames'],
        slice_indices=config['data']['slice_indices'],
        transform=transforms.Normalize(mean=[0.5], std=[0.5])
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = create_flf2v_model(config['model'])
    
    # Load checkpoint if resuming
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Create trainer
    trainer = FLF2VTrainer(
        model=model,
        config=config['training'],
        device=device
    )
    
    # Save config
    save_path = Path(config['training']['save_dir']) / config['training']['run_name']
    save_path.mkdir(exist_ok=True, parents=True)
    save_config(config, save_path / 'config.yaml')
    
    # Train
    trainer.train(train_loader, val_loader, num_epochs=config['training']['num_epochs'])


# Example inference script
def inference_example():
    """Example inference script"""
    parser = argparse.ArgumentParser(description='Generate videos with Lung CT FLF2V')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--first_frame', type=str, required=True, help='First frame path')
    parser.add_argument('--last_frame', type=str, required=True, help='Last frame path')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--num_frames', type=int, default=40, help='Number of frames')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='CFG scale')
    args = parser.parse_args()
    
    # Load model
    from .model import create_flf2v_model
    import nibabel as nib
    import numpy as np
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cuda')
    config = checkpoint['config']
    
    # Create model
    model = create_flf2v_model(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.cuda()
    
    # Load first and last frames (assuming NIfTI format)
    first_nii = nib.load(args.first_frame)
    last_nii = nib.load(args.last_frame)
    
    first_frame = torch.from_numpy(first_nii.get_fdata()).float()
    last_frame = torch.from_numpy(last_nii.get_fdata()).float()
    
    # Normalize to [-1, 1]
    first_frame = (first_frame - first_frame.mean()) / first_frame.std()
    last_frame = (last_frame - last_frame.mean()) / last_frame.std()
    
    # Add batch and channel dimensions
    first_frame = first_frame.unsqueeze(0).unsqueeze(0).cuda()
    last_frame = last_frame.unsqueeze(0).unsqueeze(0).cuda()
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            first_frame,
            last_frame,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale
        )
    
    # Save output
    generated_np = generated.squeeze().cpu().numpy()
    
    # Create 4D NIfTI (time series)
    output_nii = nib.Nifti1Image(generated_np.transpose(1, 2, 3, 0), first_nii.affine)
    nib.save(output_nii, args.output)
    
    print(f"Generated video saved to: {args.output}")


# Memory optimization tips for limited GPU
def get_memory_efficient_config() -> Dict:
    """Get configuration optimized for limited GPU memory"""
    config = DEFAULT_CONFIG.copy()
    
    # Reduce model size
    config['model']['dit_config']['hidden_dim'] = 512  # Smaller hidden dim
    config['model']['dit_config']['depth'] = 12  # Fewer layers
    
    # Reduce batch size
    config['training']['batch_size'] = 1
    
    # Enable gradient checkpointing
    config['training']['gradient_checkpointing'] = True
    
    # Mixed precision training
    config['training']['mixed_precision'] = True
    
    # Reduce number of frames during initial training
    config['training']['num_frames'] = 20
    
    return config


if __name__ == "__main__":
    # Example: save a memory-efficient config
    mem_config = get_memory_efficient_config()
    save_config(mem_config, "config_memory_efficient.yaml")
    
    # Example: save default config
    save_config(DEFAULT_CONFIG, "config_default.yaml")
    
    print("Configuration files saved!")
    print("\nTo train: python -m lungct_flf2v.config train --config config_default.yaml")
    print("To generate: python -m lungct_flf2v.config inference --checkpoint path/to/ckpt.pt --first_frame f1.nii --last_frame f2.nii --output out.nii")