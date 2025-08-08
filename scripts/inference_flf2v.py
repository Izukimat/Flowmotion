#!/usr/bin/env python3
"""
Inference script for Lung CT FLF2V model
Generates interpolated breathing sequences from first and last frames
"""

import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import yaml

# Import our modules (absolute package path)
from src.flf2v.lungct_vae import LungCTVAE
from src.flf2v.lungct_dit import create_dit_model
from src.flf2v.lungct_flow_matching import create_flow_matching_model, FlowMatchingConfig
from src.flf2v.lungct_flf2v_model import LungCTFLF2V


# ============= Inference Dataset =============

class InferenceDataset(Dataset):
    """
    Dataset for inference - provides first and last frames
    """
    
    def __init__(
        self,
        first_frame_paths: List[str],
        last_frame_paths: List[str],
        target_size: Tuple[int, int] = (128, 128),
        intensity_window: Tuple[float, float] = (-1000, 500)
    ):
        assert len(first_frame_paths) == len(last_frame_paths)
        
        self.first_frame_paths = first_frame_paths
        self.last_frame_paths = last_frame_paths
        self.target_size = target_size
        self.intensity_window = intensity_window
    
    def _load_and_preprocess(self, path: str) -> torch.Tensor:
        """Load and preprocess a single frame"""
        # Load NIfTI
        nii = nib.load(path)
        data = nii.get_fdata()
        
        # Handle different formats
        if data.ndim == 3:
            # If 3D, take middle slice
            data = data[:, :, data.shape[2] // 2]
        elif data.ndim > 3:
            # If 4D, take first timepoint and middle slice
            data = data[:, :, data.shape[2] // 2, 0]
        
        # Apply intensity window
        data = np.clip(data, self.intensity_window[0], self.intensity_window[1])
        
        # Normalize to [-1, 1]
        data = 2 * (data - self.intensity_window[0]) / (self.intensity_window[1] - self.intensity_window[0]) - 1
        
        # Resize if needed
        if data.shape != self.target_size:
            data = cv2.resize(data, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor [1, H, W]
        tensor = torch.from_numpy(data).float().unsqueeze(0)
        
        return tensor, nii.affine
    
    def __len__(self) -> int:
        return len(self.first_frame_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        first_frame, first_affine = self._load_and_preprocess(self.first_frame_paths[idx])
        last_frame, last_affine = self._load_and_preprocess(self.last_frame_paths[idx])
        
        return {
            'first_frame': first_frame,
            'last_frame': last_frame,
            'first_affine': first_affine,
            'last_affine': last_affine,
            'first_path': self.first_frame_paths[idx],
            'last_path': self.last_frame_paths[idx]
        }


# ============= Model Loading =============

def load_model(checkpoint_path: str, device: str = 'cuda') -> Tuple[LungCTFLF2V, Dict]:
    """Load model from checkpoint"""
    logging.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    vae = LungCTVAE(
        base_model_name=config['model']['vae_base_model'],
        latent_channels=config['model']['latent_channels'],
        temporal_weight=config['model']['vae_temporal_weight'],
        use_tanh_scaling=True,
        freeze_pretrained=True  # Keep frozen for inference
    )
    
    dit = create_dit_model(config['model']['dit_config'])
    
    fm_config = FlowMatchingConfig(**config['model']['flow_matching_config'])
    flow_matching = create_flow_matching_model({'config': config['model']['flow_matching_config']})
    
    model = LungCTFLF2V(
        vae=vae,
        dit=dit,
        flow_matching=flow_matching,
        freeze_vae_after=0  # Already frozen
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


# ============= Generation Functions =============

@torch.no_grad()
def generate_sequence(
    model: LungCTFLF2V,
    first_frame: torch.Tensor,          # [B, 1, H, W]
    last_frame:  torch.Tensor,          # [B, 1, H, W]
    num_frames:  int  = 40,
    guidance_scale:   float = 1.0,
    num_inference_steps: int = 50,
    device: str = "cuda",
) -> torch.Tensor:                       # returns [B, 1, T, H, W]
    """
    Generate a video sequence from first and last frames
    
    Args:
        model: Trained FLF2V model
        first_frame: First frame tensor [B, 1, H, W]
        last_frame: Last frame tensor [B, 1, H, W]
        num_frames: Number of frames to generate (including first/last)
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of denoising steps
    
    Returns:
        Generated sequence [B, 1, T, H, W]
    """
    first_frame = first_frame.to(device)
    last_frame = last_frame.to(device)
    
    # Update flow matching config
    model.flow_matching.config.num_sampling_steps = num_inference_steps
    
    # Generate
    generated = model.generate(
        first_frame,
        last_frame,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        decode=True  # Decode to pixel space
    )
    
    return generated


def save_as_nifti(
    volume: np.ndarray,
    affine: np.ndarray,
    output_path: str,
    time_axis: bool = True
):
    """Save volume as NIfTI file"""
    if time_axis:
        # Reorder from [C, T, H, W] to [H, W, T, C] for NIfTI
        if volume.ndim == 4:
            volume = np.transpose(volume, (2, 3, 1, 0))
        elif volume.ndim == 3:
            volume = np.transpose(volume, (1, 2, 0))
    
    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, output_path)
    logging.info(f"Saved to {output_path}")


def save_as_video(
    volume: np.ndarray,
    output_path: str,
    fps: int = 10,
    quality: int = 95
):
    """Save volume as MP4 video"""
    import imageio
    
    # Normalize to [0, 255]
    volume = ((volume + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    # If single channel, repeat for RGB
    if volume.shape[0] == 1:
        volume = np.repeat(volume, 3, axis=0)
    
    # Transpose to [T, H, W, C]
    volume = np.transpose(volume, (1, 2, 3, 0))
    
    # Write video
    imageio.mimwrite(output_path, volume, fps=fps, quality=quality)
    logging.info(f"Saved video to {output_path}")


def create_comparison_figure(
    first_frame: np.ndarray,
    last_frame: np.ndarray,
    generated: np.ndarray,
    output_path: str
):
    """Create comparison figure showing first, middle, last frames"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Original first and last
    axes[0, 0].imshow(first_frame.squeeze(), cmap='gray')
    axes[0, 0].set_title('First Frame (Input)')
    axes[0, 0].axis('off')
    
    axes[0, 2].imshow(last_frame.squeeze(), cmap='gray')
    axes[0, 2].set_title('Last Frame (Input)')
    axes[0, 2].axis('off')
    
    # Generated frames
    T = generated.shape[1]
    mid_idx = T // 2
    
    axes[0, 1].imshow(generated[0, mid_idx], cmap='gray')
    axes[0, 1].set_title(f'Generated Frame {mid_idx}')
    axes[0, 1].axis('off')
    
    # Show a few more generated frames
    for i, idx in enumerate([T//4, T//2, 3*T//4]):
        axes[1, i].imshow(generated[0, idx], cmap='gray')
        axes[1, i].set_title(f'Generated Frame {idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved comparison figure to {output_path}")


# ============= Batch Processing =============

def process_batch(
    model: LungCTFLF2V,
    batch: Dict,
    config: Dict,
    output_dir: Path,
    save_formats: List[str] = ['nifti', 'video', 'figure']
) -> Dict[str, float]:
    """Process a batch of samples"""
    device = next(model.parameters()).device
    
    # Generate sequences
    start_time = time.time()
    
    generated = generate_sequence(
        model,
        batch['first_frame'],
        batch['last_frame'],
        num_frames=config.get('num_frames', 40),
        guidance_scale=config.get('guidance_scale', 1.0),
        num_inference_steps=config.get('num_inference_steps', 50),
        device=device
    )
    
    generation_time = time.time() - start_time
    
    # Process each sample in batch
    batch_size = generated.shape[0]
    
    for i in range(batch_size):
        # Extract paths for naming
        first_path = Path(batch['first_path'][i])
        last_path = Path(batch['last_path'][i])
        
        sample_name = f"{first_path.stem}_to_{last_path.stem}"
        sample_dir = output_dir / sample_name
        sample_dir.mkdir(exist_ok=True, parents=True)
        
        # Get data
        first_np = batch['first_frame'][i].cpu().numpy()
        last_np = batch['last_frame'][i].cpu().numpy()
        generated_np = generated[i].cpu().numpy()
        
        # Save in different formats
        if 'nifti' in save_formats:
            save_as_nifti(
                generated_np,
                batch['first_affine'][i].numpy(),
                str(sample_dir / 'generated_sequence.nii.gz'),
                time_axis=True
            )
        
        if 'video' in save_formats:
            save_as_video(
                generated_np,
                str(sample_dir / 'generated_sequence.mp4'),
                fps=config.get('video_fps', 10)
            )
        
        if 'figure' in save_formats:
            create_comparison_figure(
                first_np,
                last_np,
                generated_np,
                str(sample_dir / 'comparison.png')
            )
        
        # Save individual frames if requested
        if config.get('save_individual_frames', False):
            frames_dir = sample_dir / 'frames'
            frames_dir.mkdir(exist_ok=True)
            
            for t in range(generated_np.shape[1]):
                frame = generated_np[0, t]
                frame_normalized = ((frame + 1) * 127.5).clip(0, 255).astype(np.uint8)
                cv2.imwrite(
                    str(frames_dir / f'frame_{t:03d}.png'),
                    frame_normalized
                )
    
    return {
        'generation_time': generation_time,
        'samples_processed': batch_size,
        'time_per_sample': generation_time / batch_size
    }


# ============= Evaluation Metrics =============

def compute_metrics(
    generated: torch.Tensor,
    first_frame: torch.Tensor,
    last_frame: torch.Tensor
) -> Dict[str, float]:
    """Compute evaluation metrics"""
    metrics = {}
    
    # Check first/last frame preservation
    first_error = torch.mean((generated[:, :, 0] - first_frame.squeeze(2))**2).item()
    last_error = torch.mean((generated[:, :, -1] - last_frame.squeeze(2))**2).item()
    
    metrics['first_frame_mse'] = first_error
    metrics['last_frame_mse'] = last_error
    
    # Temporal smoothness (first-order differences)
    temporal_diff = generated[:, :, 1:] - generated[:, :, :-1]
    metrics['temporal_smoothness'] = torch.mean(temporal_diff**2).item()
    
    # Acceleration (second-order differences)
    if generated.shape[2] > 2:
        acceleration = temporal_diff[:, :, 1:] - temporal_diff[:, :, :-1]
        metrics['temporal_acceleration'] = torch.mean(acceleration**2).item()
    
    return metrics


# ============= Main Inference Function =============

def main():
    parser = argparse.ArgumentParser(description='Generate lung CT sequences with FLF2V')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    
    # Input specification (multiple options)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-csv', type=str, help='CSV file with first/last frame paths')
    input_group.add_argument('--first-frame', type=str, help='Single first frame path')
    parser.add_argument('--last-frame', type=str, help='Single last frame path (required with --first-frame)')
    input_group.add_argument('--input-dir', type=str, help='Directory with paired frames')
    
    # Generation parameters
    parser.add_argument('--num-frames', type=int, default=40, help='Number of frames to generate')
    parser.add_argument('--guidance-scale', type=float, default=1.0, help='Classifier-free guidance scale')
    parser.add_argument('--num-inference-steps', type=int, default=50, help='Number of denoising steps')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    
    # Output options
    parser.add_argument('--save-formats', nargs='+', default=['nifti', 'video', 'figure'],
                        choices=['nifti', 'video', 'figure'], help='Output formats')
    parser.add_argument('--video-fps', type=int, default=10, help='FPS for video output')
    parser.add_argument('--save-individual-frames', action='store_true', help='Save individual frames as PNG')
    
    # Other options
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.first_frame and not args.last_frame:
        parser.error("--last-frame required when using --first-frame")
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, config = load_model(args.checkpoint, args.device)
    
    # Prepare inference config
    inference_config = {
        'num_frames': args.num_frames,
        'guidance_scale': args.guidance_scale,
        'num_inference_steps': args.num_inference_steps,
        'video_fps': args.video_fps,
        'save_individual_frames': args.save_individual_frames
    }
    
    # Save inference config
    with open(output_dir / 'inference_config.yaml', 'w') as f:
        yaml.dump(inference_config, f)
    
    # Prepare input data
    if args.input_csv:
        # Load from CSV
        df = pd.read_csv(args.input_csv)
        first_frames = df['first_frame'].tolist()
        last_frames = df['last_frame'].tolist()
        
    elif args.first_frame:
        # Single pair
        first_frames = [args.first_frame]
        last_frames = [args.last_frame]
        
    elif args.input_dir:
        # Load from directory (assumes naming convention)
        input_dir = Path(args.input_dir)
        first_frames = sorted(input_dir.glob('*_first.nii.gz'))
        last_frames = sorted(input_dir.glob('*_last.nii.gz'))
        
        # Match pairs
        first_frames = [str(f) for f in first_frames]
        last_frames = [str(f) for f in last_frames]
    
    # Create dataset
    dataset = InferenceDataset(
        first_frames,
        last_frames,
        target_size=tuple(config['data']['target_size'][:2]) if 'target_size' in config['data'] else (128, 128),
        intensity_window=tuple(config['data'].get('intensity_window', (-600, 900)))
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # NIfTI loading doesn't parallelize well
        pin_memory=True
    )
    
    # Process all samples
    total_metrics = []
    total_time = 0
    
    logging.info(f"Processing {len(dataset)} sample pairs...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating sequences")):
        # Process batch
        timing = process_batch(
            model,
            batch,
            inference_config,
            output_dir,
            save_formats=args.save_formats
        )
        
        total_time += timing['generation_time']
        
        # Compute metrics if requested
        if args.verbose:
            with torch.no_grad():
                generated = generate_sequence(
                    model,
                    batch['first_frame'],
                    batch['last_frame'],
                    num_frames=args.num_frames,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    device=args.device
                )
                
                metrics = compute_metrics(
                    generated,
                    batch['first_frame'],
                    batch['last_frame']
                )
                total_metrics.append(metrics)
    
    # Summary statistics
    logging.info(f"\nInference complete!")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average time per sample: {total_time/len(dataset):.2f}s")
    logging.info(f"Output saved to: {output_dir}")
    
    # Save metrics if computed
    if total_metrics:
        avg_metrics = {
            k: np.mean([m[k] for m in total_metrics])
            for k in total_metrics[0].keys()
        }
        
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump({
                'average_metrics': avg_metrics,
                'all_metrics': total_metrics
            }, f, indent=2)
        
        logging.info("\nAverage metrics:")
        for k, v in avg_metrics.items():
            logging.info(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
