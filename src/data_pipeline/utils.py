"""
Utility classes and helper functions for data pipeline
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import numpy as np
from datetime import datetime


@dataclass
class Task:
    """Represents a single interpolation task"""
    patient_id: str
    experiment_id: str
    series_id: str
    slice_num: int
    phase_range: str  # '0-50' or '0-100'
    cycle_id: Optional[str] = None  # For tracking which breathing cycle
    
    def get_output_path(self, base_dir: Path) -> Path:
        """Construct output path for this task"""
        return (base_dir / 'data' / self.patient_id / self.experiment_id / 
                self.series_id / f'slice_{self.slice_num:04d}' / f'phase_{self.phase_range}')
    
    def get_input_data_paths(self, processed_dir: Path) -> Dict[str, Path]:
        """Get paths to required input phase volumes"""
        cycle_dir = processed_dir / self.patient_id / self.cycle_id
        return {
            'cycle_dir': cycle_dir,
            'phase_files': sorted(cycle_dir.glob('phase_*.npy'))
        }
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'patient_id': self.patient_id,
            'experiment_id': self.experiment_id,
            'series_id': self.series_id,
            'slice_num': self.slice_num,
            'phase_range': self.phase_range,
            'cycle_id': self.cycle_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        """Create Task from dictionary"""
        return cls(**data)


@dataclass
class DataSample:
    """Represents a training/inference sample with metadata"""
    patient_id: str
    experiment_id: str
    series_id: str
    slice_num: int
    phase_range: str
    split: str  # 'train', 'val', or 'test'
    input_path: Path
    target_path: Path
    metadata: Dict = field(default_factory=dict)
    
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load input and target arrays"""
        input_data = np.load(self.input_path)
        target_data = np.load(self.target_path)
        return input_data, target_data
    
    @property
    def is_3d(self) -> bool:
        """Check if this is 3D data (for future extension)"""
        return self.metadata.get('data_type', '2d') == '3d'


def create_directory_structure(base_dir: Path) -> None:
    """Create the base directory structure for the pipeline"""
    dirs = [
        base_dir / 'data',
        base_dir / 'splits',
        base_dir / 'experiment_configs',
        base_dir / 'logs'
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def load_patient_splits(splits_file: Path) -> Dict[str, List[str]]:
    """Load patient-level train/val/test splits"""
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_file}")
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    # Validate structure
    required_keys = {'train', 'val', 'test'}
    if not all(key in splits for key in required_keys):
        raise ValueError(f"Splits file must contain keys: {required_keys}")
    
    return splits


def save_patient_splits(splits: Dict[str, List[str]], splits_file: Path) -> None:
    """Save patient-level splits to JSON file"""
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)


def parse_phase_range(phase_range: str) -> Tuple[int, int]:
    """Parse phase range string (e.g., '0-50') into start and end values"""
    parts = phase_range.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid phase range format: {phase_range}")
    return int(parts[0]), int(parts[1])


def get_phase_indices(phase_percent: int, num_phases: int = 10) -> int:
    """Convert phase percentage to index (0% -> 0, 10% -> 1, etc.)"""
    return phase_percent // 10


def extract_slice_from_volume(volume_path: Path, slice_num: int) -> np.ndarray:
    """Extract a single 2D slice from a 3D volume"""
    volume = np.load(volume_path, mmap_mode='r')
    if slice_num >= volume.shape[0]:
        raise ValueError(f"Slice {slice_num} out of bounds for volume with {volume.shape[0]} slices")
    return volume[slice_num].copy()


def save_interpolation_result(output_dir: Path, input_frames: np.ndarray, 
                            target_frames: np.ndarray, metadata: dict,
                            save_video: bool = False) -> None:
    """Save interpolation results in standardized format"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.save(output_dir / 'input_frames.npy', input_frames)
    np.save(output_dir / 'target_frames.npy', target_frames)
    
    # Save metadata
    metadata['created_at'] = datetime.now().isoformat()
    metadata['input_shape'] = list(input_frames.shape)
    metadata['target_shape'] = list(target_frames.shape)
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Optionally save as video for visualization
    if save_video and target_frames.shape[0] > 10:
        try:
            import cv2
            video_path = output_dir / 'interpolated_sequence.mp4'
            height, width = target_frames[0].shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Assuming ~5 fps for breathing motion visualization
            out = cv2.VideoWriter(str(video_path), fourcc, 5.0, (width, height))
            
            for frame in target_frames:
                # Normalize to 0-255
                frame_norm = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
                # Convert grayscale to BGR for video
                frame_bgr = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)
                out.write(frame_bgr)
            
            out.release()
            metadata['video_path'] = str(video_path)
        except Exception as e:
            # Video saving is optional, don't fail the whole process
            pass


def get_patient_from_path(path: Path) -> str:
    """Extract patient ID from a file path"""
    # Assuming path structure includes patient ID
    parts = path.parts
    for part in parts:
        if part.startswith('Patient_') or part.startswith('RIDER-'):
            return part
    raise ValueError(f"Could not extract patient ID from path: {path}")


def estimate_storage_size(num_patients: int, num_experiments: int, 
                         avg_slices_per_patient: int, num_phase_ranges: int,
                         slice_size_mb: float = 2.0) -> float:
    """Estimate total storage requirements in GB"""
    total_combinations = num_patients * num_experiments * avg_slices_per_patient * num_phase_ranges
    total_mb = total_combinations * slice_size_mb
    return total_mb / 1024  # Convert to GB