"""
Canonical LungCT Dataset for FLF2V model training
Single source of truth for data loading from NumPy arrays
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LungCTDataset(Dataset):
    """
    Canonical dataset for lung CT breathing sequences using preprocessed NumPy arrays
    Works with data structure: patient/experiment/series/slice_XXXX/phase_X-Y/
    """
    
    def __init__(
        self,
        csv_file: str,
        split: str = 'train',
        data_root: str = '/home/ragenius_admin/azureblob/4D-Lung-Interpolated/data/',
        augment: bool = True,
        normalize: bool = True,
        load_input_frames: bool = True,
        load_target_frames: bool = True,
        phase_filter: Optional[str] = None,
        experiment_filter: Optional[List[str]] = None
    ):
        self.csv_file = csv_file
        self.split = split
        self.data_root = Path(data_root)
        self.augment = augment and (split == 'train')
        self.normalize = normalize
        self.load_input_frames = load_input_frames
        self.load_target_frames = load_target_frames
        self.phase_filter = phase_filter
        self.experiment_filter = experiment_filter
        
        # Load and filter metadata
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split]
        
        # Apply filters if specified
        if phase_filter:
            self.df = self.df[self.df['phase_range'] == phase_filter]
        
        if experiment_filter:
            self.df = self.df[self.df['experiment_id'].isin(experiment_filter)]
        
        self.df = self.df.reset_index(drop=True)
        
        # Build sample list with verified paths
        self.samples = self.df.to_dict(orient="records")
        logging.info(f"Loaded manifest with {len(self.samples)} {split} samples")
    
    def _load_numpy_files(self, sample: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load NumPy arrays for a sample"""
        input_frames  = np.load(sample["input_path"])  if self.load_input_frames  else None
        target_frames = np.load(sample["target_path"]) if self.load_target_frames else None
        return input_frames, target_frames
    
    def _normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Apply lung windowing and normalization"""
        if not self.normalize:
            return frames.astype(np.float32)
        
        # Lung window parameters
        window_center = -600
        window_width = 1500
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        
        # Apply windowing
        frames_windowed = np.clip(frames, lower, upper)
        
        # Normalize to [-1, 1] for flow matching
        frames_norm = 2 * (frames_windowed - lower) / (upper - lower) - 1
        
        return frames_norm.astype(np.float32)
    
    def _augment_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation - Fix: use scalar random values"""
        if not self.augment:
            return tensor
        
        # Random intensity shifts - Fix: use random.random() for scalar
        if random.random() < 0.5:
            shift = torch.randn(()) * 0.05  # Fix: scalar tensor
            tensor = tensor + shift
        
        # Random intensity scaling
        if random.random() < 0.5:
            scale = 1 + torch.randn(()) * 0.05  # Fix: scalar tensor
            tensor = tensor * scale
        
        # Random noise
        if random.random() < 0.3:
            noise = torch.randn_like(tensor) * 0.01
            tensor = tensor + noise
        
        # Clamp to valid range
        tensor = torch.clamp(tensor, -1, 1)
        
        return tensor
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load NumPy arrays
        input_frames, target_frames = self._load_numpy_files(sample)
        
        # Basic sample info (strings and ints only for collate compatibility)
        result = {
            'patient_id': sample['patient_id'],
            'experiment_id': sample['experiment_id'],
            'series_id': sample['series_id'],
            'slice_num': sample['slice_num'],
            'phase_range': sample['phase_range']
        }
        
        # Process input frames if loaded
        if input_frames is not None:
            input_frames = self._normalize_frames(input_frames)
            input_tensor = torch.from_numpy(input_frames).unsqueeze(0)  # Add channel: (1, T, H, W)
            
            if self.augment:
                input_tensor = self._augment_tensor(input_tensor)
            
            result['input_frames'] = input_tensor
        
        # Process target frames if loaded
        if target_frames is not None:
            target_frames = self._normalize_frames(target_frames)
            target_tensor = torch.from_numpy(target_frames).unsqueeze(0)  # Add channel: (1, T, H, W)
            
            if self.augment:
                target_tensor = self._augment_tensor(target_tensor)
            
            result['target_frames'] = target_tensor
            # Also provide as 'video' for training compatibility
            result['video'] = target_tensor
        
        return result


def lungct_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for LungCT data
    Handles mixed tensor/string data properly
    """
    if not batch:
        return {}
    
    # Separate tensor and non-tensor data
    tensor_keys = []
    string_keys = []
    
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            tensor_keys.append(key)
        else:
            string_keys.append(key)
    
    # Build result dict
    result = {}
    
    # Stack tensor data
    for key in tensor_keys:
        result[key] = torch.stack([item[key] for item in batch])
    
    # Keep string/scalar data as lists
    for key in string_keys:
        result[key] = [item[key] for item in batch]
    
    return result