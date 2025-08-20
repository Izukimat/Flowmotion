#!/usr/bin/env python3
"""
Model Evaluation Script for Lung CT FLF2V
Generates interpolations using trained model and evaluates against ground truth
Assumes base_evaluation.py exists for baseline comparison functions
"""

import os
import argparse
import json
import time
import warnings
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
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import yaml

# Import our modules
from src.flf2v.lungct_vae import LungCTVAE
from src.flf2v.lungct_dit import create_dit_model
from src.flf2v.lungct_flow_matching import create_flow_matching_model, FlowMatchingConfig
from src.flf2v.lungct_flf2v_model import LungCTFLF2V
from src.flf2v.datasets import LungCTDataset

# Import baseline evaluation functions (assumed to be renamed)
try:
    from base_evaluation import (
        compute_optical_flow_error,
        compute_temporal_smoothness,
        compute_anatomical_consistency,
        compute_motion_fidelity,
        create_evaluation_visualizations
    )
    BASELINE_AVAILABLE = True
except ImportError:
    logging.warning("base_evaluation.py not found. Some metrics will be unavailable.")
    BASELINE_AVAILABLE = False

warnings.filterwarnings('ignore')

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

# ============= Model Loading =============

def load_model(checkpoint_path: str, device: str = 'cuda') -> Tuple[LungCTFLF2V, Dict]:
    """Load model from checkpoint"""
    logging.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model components
    vae = LungCTVAE(
        base_model_name=config['model']['vae_base_model'],
        latent_channels=config['model']['latent_channels'],
        temporal_weight=config['model']['vae_temporal_weight'],
        use_tanh_scaling=True,
        freeze_pretrained=True  # Keep frozen for inference
    )
    
    dit = create_dit_model(config['model']['dit_config'])
    
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
    
    logging.info(f"Model loaded successfully. VAE frozen: {model._vae_frozen}")
    
    return model, config

# ============= Evaluation Dataset =============

class EvaluationDataset(Dataset):
    """
    Dataset for model evaluation - provides ground truth intermediate frames
    """
    
    def __init__(
        self,
        data_root: str,
        split_csv: str,
        split: str = 'test',
        target_size: Tuple[int, int] = (128, 128),
        intensity_window: Tuple[float, float] = (-1000, 500),
        max_samples: Optional[int] = None
    ):
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.intensity_window = intensity_window
        
        # Load split information
        split_df = pd.read_csv(split_csv)
        self.samples = split_df[split_df['split'] == split].to_dict('records')
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        logging.info(f"Loaded {len(self.samples)} samples for {split} evaluation")
    
    def _load_numpy_sequence(self, sample: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load complete sequence from numpy files"""
        # Construct paths based on data structure
        patient_dir = self.data_root / sample['patient_id']
        
        # Find matching experiment directory
        exp_dirs = list(patient_dir.glob('*'))
        if not exp_dirs:
            raise FileNotFoundError(f"No experiment directories found in {patient_dir}")
        
        exp_dir = exp_dirs[0]  # Take first available experiment
        series_dir = exp_dir / sample['series_id']
        slice_dir = series_dir / f"slice_{sample['slice_num']:04d}"
        phase_dir = slice_dir / sample['phase_range']
        
        # Load input and target frames
        input_path = phase_dir / 'input_frames.npy'
        target_path = phase_dir / 'target_frames.npy'
        
        if not input_path.exists() or not target_path.exists():
            raise FileNotFoundError(f"Required files not found in {phase_dir}")
        
        input_frames = np.load(input_path).astype(np.float32)    # [10, H, W]
        target_frames = np.load(target_path).astype(np.float32)  # [82, H, W]
        
        return input_frames, target_frames
    
    def _normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Normalize frames to [-1, 1] range"""
        # Apply intensity window
        frames = np.clip(frames, self.intensity_window[0], self.intensity_window[1])
        
        # Normalize to [-1, 1]
        frames = 2 * (frames - self.intensity_window[0]) / (
            self.intensity_window[1] - self.intensity_window[0]
        ) - 1
        
        return frames
    
    def _resize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Resize frames to target size"""
        if frames.shape[-2:] == self.target_size:
            return frames
        
        resized = np.zeros((frames.shape[0], *self.target_size), dtype=frames.dtype)
        for i in range(frames.shape[0]):
            resized[i] = cv2.resize(
                frames[i], self.target_size[::-1], interpolation=cv2.INTER_LINEAR
            )
        
        return resized
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        try:
            input_frames, target_frames = self._load_numpy_sequence(sample)
            
            # Normalize and resize
            input_frames = self._normalize_frames(input_frames)
            target_frames = self._normalize_frames(target_frames)
            
            input_frames = self._resize_frames(input_frames)
            target_frames = self._resize_frames(target_frames)
            
            # Convert to tensors
            input_tensor = torch.from_numpy(input_frames).float()    # [10, H, W]
            target_tensor = torch.from_numpy(target_frames).float()  # [82, H, W]
            
            # Extract first and last frames for model input
            first_frame = input_tensor[0:1]  # [1, H, W] - 0% phase
            last_frame = input_tensor[5:6]   # [1, H, W] - 50% phase
            
            # Extract ground truth intermediate frames (10%, 20%, 30%, 40%)
            # Assuming target frames have original phases at specific indices
            gt_intermediate = target_tensor[[9, 18, 27, 36]]  # Indices for 10%, 20%, 30%, 40%
            
            return {
                'first_frame': first_frame,
                'last_frame': last_frame,
                'gt_intermediate': gt_intermediate,
                'full_sequence': target_tensor,
                'sample_info': sample
            }
            
        except Exception as e:
            logging.error(f"Error loading sample {idx}: {e}")
            # Return dummy data to prevent batch loading failure
            dummy_shape = (1, *self.target_size)
            return {
                'first_frame': torch.zeros(dummy_shape),
                'last_frame': torch.zeros(dummy_shape),
                'gt_intermediate': torch.zeros(4, *self.target_size[1:]),
                'full_sequence': torch.zeros(10, *self.target_size[1:]),
                'sample_info': sample,
                'error': str(e)
            }

# ============= Generation Functions =============

@torch.no_grad()
def generate_interpolation(
    model: LungCTFLF2V,
    first_frame: torch.Tensor,      # [B, 1, H, W]
    last_frame: torch.Tensor,       # [B, 1, H, W]
    num_frames: int = 4,
    guidance_scale: float = 1.0,
    num_inference_steps: int = 50,
    device: str = "cuda"
) -> torch.Tensor:                  # [B, num_frames, H, W]
    """
    Generate interpolated frames between first and last frame
    """
    model.eval()
    
    B, C, H, W = first_frame.shape
    
    # Move to device
    first_frame = first_frame.to(device)
    last_frame = last_frame.to(device)
    
    # Encode first and last frames
    with torch.no_grad():
        first_encoded = model.encode_frames(first_frame)['latent']  # [B, C_latent, H', W']
        last_encoded = model.encode_frames(last_frame)['latent']
    
    # Add temporal dimension and create sequence
    # For 4 intermediate frames: [first, inter1, inter2, inter3, inter4, last]
    sequence_length = num_frames + 2  # Including first and last
    
    # Create latent sequence
    latent_shape = (B, first_encoded.shape[1], sequence_length, 
                   first_encoded.shape[2], first_encoded.shape[3])
    
    # Initialize with noise
    if hasattr(model.flow_matching.config, 'init_strategy') and \
       model.flow_matching.config.init_strategy == 'linear':
        # Linear initialization
        latent_sequence = torch.zeros(latent_shape, device=device)
        for t in range(sequence_length):
            alpha = t / (sequence_length - 1)
            latent_sequence[:, :, t] = (1 - alpha) * first_encoded + alpha * last_encoded
    else:
        # Noise initialization
        latent_sequence = torch.randn(latent_shape, device=device) * 0.01
    
    # Set endpoints
    latent_sequence[:, :, 0] = first_encoded
    latent_sequence[:, :, -1] = last_encoded
    
    # Flow matching sampling using model's inference
    generated_latents = model.flow_matching.sample(
        latent_sequence,
        first_encoded,
        last_encoded,
        model.dit,
        num_steps=num_inference_steps,
        guidance_scale=guidance_scale if guidance_scale != 1.0 else None
    )
    
    # Decode to pixel space
    # Extract intermediate frames (exclude first and last)
    intermediate_latents = generated_latents[:, :, 1:-1]  # [B, C, num_frames, H', W']
    
    # Reshape for decoding
    B, C_latent, T, H_latent, W_latent = intermediate_latents.shape
    intermediate_flat = intermediate_latents.permute(0, 2, 1, 3, 4).reshape(
        B * T, C_latent, H_latent, W_latent
    )
    
    # Decode
    decoded_flat = model.decode_frames(intermediate_flat)['reconstruction']
    
    # Reshape back
    decoded = decoded_flat.reshape(B, T, decoded_flat.shape[1], H, W)
    
    return decoded.squeeze(2)  # [B, T, H, W]

# ============= Evaluation Metrics =============

def compute_reconstruction_metrics(
    generated: torch.Tensor,     # [B, T, H, W]
    ground_truth: torch.Tensor   # [B, T, H, W]
) -> Dict[str, float]:
    """Compute reconstruction quality metrics"""
    
    # Convert to numpy and ensure same range
    gen_np = generated.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()
    
    # Normalize to [0, 1] for SSIM and PSNR
    gen_norm = (gen_np + 1) / 2
    gt_norm = (gt_np + 1) / 2
    
    metrics = {}
    
    # Compute metrics for each frame
    ssim_scores = []
    psnr_scores = []
    mse_scores = []
    
    B, T, H, W = gen_norm.shape
    
    for b in range(B):
        for t in range(T):
            # SSIM
            ssim_val = ssim(
                gt_norm[b, t], gen_norm[b, t],
                data_range=1.0
            )
            ssim_scores.append(ssim_val)
            
            # PSNR
            psnr_val = psnr(
                gt_norm[b, t], gen_norm[b, t],
                data_range=1.0
            )
            psnr_scores.append(psnr_val)
            
            # MSE
            mse_val = np.mean((gt_norm[b, t] - gen_norm[b, t]) ** 2)
            mse_scores.append(mse_val)
    
    metrics.update({
        'ssim_mean': np.mean(ssim_scores),
        'ssim_std': np.std(ssim_scores),
        'psnr_mean': np.mean(psnr_scores),
        'psnr_std': np.std(psnr_scores),
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores)
    })
    
    return metrics

def compute_temporal_metrics(
    generated: torch.Tensor,     # [B, T, H, W]
    ground_truth: torch.Tensor   # [B, T, H, W]
) -> Dict[str, float]:
    """Compute temporal consistency metrics"""
    
    gen_np = generated.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()
    
    metrics = {}
    
    # Temporal smoothness (frame-to-frame difference)
    gen_diff = np.diff(gen_np, axis=1)  # [B, T-1, H, W]
    gt_diff = np.diff(gt_np, axis=1)
    
    # Average temporal difference
    gen_temporal_var = np.mean(np.abs(gen_diff))
    gt_temporal_var = np.mean(np.abs(gt_diff))
    
    metrics.update({
        'temporal_variance_generated': gen_temporal_var,
        'temporal_variance_ground_truth': gt_temporal_var,
        'temporal_variance_ratio': gen_temporal_var / (gt_temporal_var + 1e-8)
    })
    
    # Temporal correlation
    correlations = []
    B, T, H, W = gen_np.shape
    
    for b in range(B):
        for t in range(T - 1):
            gen_flat = gen_np[b, t].flatten()
            gen_next_flat = gen_np[b, t + 1].flatten()
            
            gt_flat = gt_np[b, t].flatten()
            gt_next_flat = gt_np[b, t + 1].flatten()
            
            # Correlation between consecutive frames
            gen_corr = np.corrcoef(gen_flat, gen_next_flat)[0, 1]
            gt_corr = np.corrcoef(gt_flat, gt_next_flat)[0, 1]
            
            if not np.isnan(gen_corr) and not np.isnan(gt_corr):
                correlations.append(abs(gen_corr - gt_corr))
    
    if correlations:
        metrics['temporal_correlation_error'] = np.mean(correlations)
    
    return metrics

def compute_motion_metrics(
    generated: torch.Tensor,     # [B, T, H, W]
    ground_truth: torch.Tensor   # [B, T, H, W]
) -> Dict[str, float]:
    """Compute motion-related metrics using optical flow"""
    
    if not BASELINE_AVAILABLE:
        return {'motion_metrics': 'unavailable'}
    
    try:
        gen_np = generated.detach().cpu().numpy()
        gt_np = ground_truth.detach().cpu().numpy()
        
        # Convert to [0, 255] uint8 for optical flow
        gen_uint8 = ((gen_np + 1) * 127.5).astype(np.uint8)
        gt_uint8 = ((gt_np + 1) * 127.5).astype(np.uint8)
        
        motion_metrics = {}
        
        # Compute optical flow error
        flow_error = compute_optical_flow_error(gen_uint8, gt_uint8)
        motion_metrics['optical_flow_error'] = flow_error
        
        # Compute motion fidelity
        motion_fidelity = compute_motion_fidelity(gen_uint8, gt_uint8)
        motion_metrics.update(motion_fidelity)
        
        return motion_metrics
        
    except Exception as e:
        logging.warning(f"Could not compute motion metrics: {e}")
        return {'motion_metrics_error': str(e)}

# ============= Evaluation Pipeline =============

def evaluate_model(
    model: LungCTFLF2V,
    dataloader: DataLoader,
    device: str = 'cuda',
    save_visualizations: bool = True,
    output_dir: Path = None
) -> Dict[str, float]:
    """Evaluate model on dataset"""
    
    model.eval()
    
    all_metrics = []
    sample_count = 0
    
    if save_visualizations and output_dir:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            
            # Skip error samples
            if 'error' in batch:
                logging.warning(f"Skipping batch {batch_idx} due to loading error")
                continue
            
            first_frame = batch['first_frame'].to(device)
            last_frame = batch['last_frame'].to(device)
            gt_intermediate = batch['gt_intermediate'].to(device)
            
            # Generate interpolations
            try:
                generated = generate_interpolation(
                    model, first_frame, last_frame,
                    num_frames=4, guidance_scale=1.0, num_inference_steps=50, device=device
                )
                
                # Compute metrics
                recon_metrics = compute_reconstruction_metrics(generated, gt_intermediate)
                temporal_metrics = compute_temporal_metrics(generated, gt_intermediate)
                motion_metrics = compute_motion_metrics(generated, gt_intermediate)
                
                # Combine all metrics
                batch_metrics = {
                    **recon_metrics,
                    **temporal_metrics,
                    **motion_metrics
                }
                
                all_metrics.append(batch_metrics)
                sample_count += first_frame.shape[0]
                
                # Save visualizations for first few samples
                if save_visualizations and output_dir and batch_idx < 5:
                    save_batch_visualization(
                        first_frame, last_frame, generated, gt_intermediate,
                        vis_dir / f'sample_{batch_idx:03d}.png'
                    )
                
            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Aggregate metrics
    if not all_metrics:
        raise RuntimeError("No successful evaluations completed")
    
    aggregated_metrics = {}
    
    # Compute means and stds
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], (int, float)):
            values = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
            if values:
                aggregated_metrics[f'{key}_mean'] = np.mean(values)
                aggregated_metrics[f'{key}_std'] = np.std(values)
    
    aggregated_metrics['num_samples'] = sample_count
    aggregated_metrics['num_batches'] = len(all_metrics)
    
    return aggregated_metrics

def save_batch_visualization(
    first_frame: torch.Tensor,
    last_frame: torch.Tensor,
    generated: torch.Tensor,
    ground_truth: torch.Tensor,
    save_path: Path
):
    """Save visualization of a batch"""
    
    # Convert to numpy and normalize to [0, 1]
    first_np = ((first_frame[0, 0].cpu().numpy() + 1) / 2).clip(0, 1)
    last_np = ((last_frame[0, 0].cpu().numpy() + 1) / 2).clip(0, 1)
    
    gen_np = ((generated[0].cpu().numpy() + 1) / 2).clip(0, 1)  # [T, H, W]
    gt_np = ((ground_truth[0].cpu().numpy() + 1) / 2).clip(0, 1)
    
    # Create visualization
    fig, axes = plt.subplots(3, 6, figsize=(15, 9))
    
    # Row 1: Input frames and generated sequence
    axes[0, 0].imshow(first_np, cmap='gray')
    axes[0, 0].set_title('First Frame (0%)')
    axes[0, 0].axis('off')
    
    for i in range(4):
        axes[0, i + 1].imshow(gen_np[i], cmap='gray')
        axes[0, i + 1].set_title(f'Generated {(i+1)*10}%')
        axes[0, i + 1].axis('off')
    
    axes[0, 5].imshow(last_np, cmap='gray')
    axes[0, 5].set_title('Last Frame (50%)')
    axes[0, 5].axis('off')
    
    # Row 2: Ground truth sequence
    axes[1, 0].imshow(first_np, cmap='gray')
    axes[1, 0].set_title('First Frame (0%)')
    axes[1, 0].axis('off')
    
    for i in range(4):
        axes[1, i + 1].imshow(gt_np[i], cmap='gray')
        axes[1, i + 1].set_title(f'GT {(i+1)*10}%')
        axes[1, i + 1].axis('off')
    
    axes[1, 5].imshow(last_np, cmap='gray')
    axes[1, 5].set_title('Last Frame (50%)')
    axes[1, 5].axis('off')
    
    # Row 3: Difference maps
    axes[2, 0].axis('off')  # Empty
    
    for i in range(4):
        diff = np.abs(gen_np[i] - gt_np[i])
        im = axes[2, i + 1].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2, i + 1].set_title(f'|Gen - GT| {(i+1)*10}%')
        axes[2, i + 1].axis('off')
    
    axes[2, 5].axis('off')  # Empty
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============= Main Function =============

def main():
    parser = argparse.ArgumentParser(description='Evaluate FLF2V Model')
    
    # Model and data paths
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory of processed data')
    parser.add_argument('--split-csv', type=str, required=True,
                       help='CSV file with train/val/test splits')
    
    # Evaluation settings
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Which split to evaluate')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Generation parameters
    parser.add_argument('--num-frames', type=int, default=4,
                       help='Number of intermediate frames to generate')
    parser.add_argument('--guidance-scale', type=float, default=1.0,
                       help='Classifier-free guidance scale')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                       help='Number of inference steps')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--save-visualizations', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save evaluation config
    eval_config = vars(args)
    with open(output_dir / 'eval_config.json', 'w') as f:
        json.dump(eval_config, f, indent=2)
    
    # Load model
    logging.info("Loading model...")
    model, model_config = load_model(args.checkpoint, args.device)
    
    # Create dataset
    logging.info("Creating evaluation dataset...")
    dataset = EvaluationDataset(
        data_root=args.data_root,
        split_csv=args.split_csv,
        split=args.split,
        target_size=(128, 128),  # Match training size
        intensity_window=(-1000, 500),
        max_samples=args.max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Run evaluation
    logging.info(f"Starting evaluation on {len(dataset)} samples...")
    start_time = time.time()
    
    try:
        metrics = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=args.device,
            save_visualizations=args.save_visualizations,
            output_dir=output_dir
        )
        
        eval_time = time.time() - start_time
        
        # Add timing info
        metrics['evaluation_time_seconds'] = eval_time
        metrics['samples_per_second'] = metrics['num_samples'] / eval_time
        
        # Save results
        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print summary
        logging.info("\n" + "="*50)
        logging.info("EVALUATION RESULTS")
        logging.info("="*50)
        logging.info(f"Samples evaluated: {metrics['num_samples']}")
        logging.info(f"Evaluation time: {eval_time:.2f}s ({metrics['samples_per_second']:.2f} samples/s)")
        logging.info("")
        
        # Print key metrics
        key_metrics = [
            ('SSIM', 'ssim_mean_mean'),
            ('PSNR', 'psnr_mean_mean'),
            ('MSE', 'mse_mean_mean'),
            ('Temporal Variance Ratio', 'temporal_variance_ratio_mean'),
        ]
        
        for metric_name, metric_key in key_metrics:
            if metric_key in metrics:
                logging.info(f"{metric_name}: {metrics[metric_key]:.4f}")
        
        logging.info(f"\nDetailed results saved to: {results_path}")
        
        if args.save_visualizations:
            logging.info(f"Visualizations saved to: {output_dir / 'visualizations'}")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise
    
    logging.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()