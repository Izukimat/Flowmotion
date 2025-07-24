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

# Import our modules
from lungct_vae import LungCTVAE
from lungct_dit import create_dit_model
from lungct_flow_matching import create_flow_matching_model, FlowMatchingConfig
from lungct_flf2v_model import LungCTFLF2V


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
    flow_matching = create_flow_matching_model(dit, {'config': fm_config})
    
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
    # (1) move to GPU and add depth dimension
    first_frame = first_frame.to(device).unsqueeze(2)   # [B,1,1,H,W]
    last_frame  = last_frame .to(device).unsqueeze(2)   # [B,1,1,H,W]

    # (2) encode to latent
    with torch.no_grad():
        first_lat = model.encode_frames(first_frame)    # [B,C,1/8,H/8,W/8]
        last_lat  = model.encode_frames(last_frame )

    # (3) override sampler settings if caller asked
    fm_cfg = model.flow_matching.config
    fm_cfg.num_sampling_steps = num_inference_steps

    latent_video = model.flow_matching.sample(
        first_lat, last_lat,
        num_frames      = num_frames,
        guidance_scale  = guidance_scale,
        progress_bar    = False,
    )                                   # [B,C,T/8,H/8,W/8]

    # (4) decode to pixel space
    video = model.decode_frames(latent_video)           # [B,1,T,H,W]
    return video


# ============= CLI entry point  =============

def main():
    parser = argparse.ArgumentParser(description="FLF2V inference")
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--pairs-file",   required=True,
                        help="txt|csv: first_path,last_path per line")
    parser.add_argument("--output-dir",   default="./outputs_infer")
    parser.add_argument("--num-frames",   type=int, default=40)
    parser.add_argument("--steps",        type=int, default=50)
    parser.add_argument("--guidance",     type=float, default=1.0)
    parser.add_argument("--gpu",          type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # -------- 1. model ------------------------------------------------------
    model, _cfg = load_model(args.checkpoint, device)

    # -------- 2. build dataset / loader ------------------------------------
    first_paths, last_paths = [], []
    with open(args.pairs_file, "r") as f:
        for line in f:
            fp, lp = line.strip().split(",")
            first_paths.append(fp)
            last_paths .append(lp)

    ds = InferenceDataset(first_paths, last_paths,
                          target_size = tuple(_cfg["data"]["target_size"][:2]),
                          intensity_window = tuple(_cfg["data"]["intensity_window"]))
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    # -------- 3. loop -------------------------------------------------------
    model.eval()
    for i, batch in enumerate(tqdm(loader, desc="Generate")):
        vid = generate_sequence(
            model,
            batch["first_frame"],
            batch["last_frame"],
            num_frames          = args.num_frames,
            guidance_scale      = args.guidance,
            num_inference_steps = args.steps,
            device              = device,
        )                                              # [B,1,T,H,W]

        vid = vid.cpu().numpy()

        # save each sample
        for j in range(vid.shape[0]):
            out_name = (
                Path(batch["first_path"][j]).stem + "_to_" +
                Path(batch["last_path"][j]).stem + ".nii.gz"
            )
            affine = batch["first_affine"][j].numpy()
            nii = nib.Nifti1Image(vid[j,0].transpose(1,2,3,0), affine)
            nib.save(nii, Path(args.output_dir) / out_name)

    print(f"Done. Saved NIfTI sequences to {args.output_dir}")


if __name__ == "__main__":
    main()
