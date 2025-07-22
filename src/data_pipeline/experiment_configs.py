"""
Experiment configurations for different interpolation methods
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class ExperimentConfig:
    """Configuration for an interpolation experiment"""
    name: str
    method: str  # 'linear', 'flow_matching', 'spline', 'optical_flow'
    input_phases: List[int]  # e.g., [0, 50] for 0% and 50% breathing
    target_phases: List[int]  # e.g., [10, 20, 30, 40] for intermediate phases
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Data selection parameters
    slice_selection: str = "all"  # 'all', 'middle_third', 'representative_levels', 'single_middle'
    priority: int = 1  # 1=highest priority, used for incremental processing
    
    def get_slice_indices(self, total_slices: int) -> List[int]:
        """Get slice indices based on selection strategy"""
        if self.slice_selection == "all":
            return list(range(total_slices))
        elif self.slice_selection == "single_middle":
            return [total_slices // 2]
        elif self.slice_selection == "middle_third":
            start = total_slices // 3
            end = 2 * total_slices // 3
            return list(range(start, end))
        elif self.slice_selection == "representative_levels":
            # 25%, 50%, 75% of lung height
            levels = [0.25, 0.5, 0.75]
            return [int(total_slices * level) for level in levels]
        else:
            raise ValueError(f"Unknown slice selection: {self.slice_selection}")
    
    def supports_phase_range(self, phase_range: str) -> bool:
        """Check if this experiment supports the given phase range"""
        start, end = map(int, phase_range.split('-'))
        # Check if input phases are within range
        return all(p >= start and p <= end for p in self.input_phases)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'method': self.method,
            'input_phases': self.input_phases,
            'target_phases': self.target_phases,
            'description': self.description,
            'params': self.params,
            'slice_selection': self.slice_selection,
            'priority': self.priority
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ExperimentConfig':
        """Create from dictionary"""
        return cls(**data)
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Preset experiment configurations
PRESET_EXPERIMENTS = {
    # High frame rate interpolation experiments
    'hfr_linear_8fps': ExperimentConfig(
        name='hfr_linear_8fps',
        method='linear',
        input_phases=list(range(0, 100, 10)),  # [0, 10, 20, ..., 90]
        target_phases=[],  # Will be populated dynamically
        description='High frame rate linear interpolation - 8 frames between each phase',
        params={'frames_per_interval': 8, 'mode': 'high_frame_rate'},
        slice_selection='representative_levels',
        priority=1
    ),
    
    'hfr_linear_10fps': ExperimentConfig(
        name='hfr_linear_10fps',
        method='linear',
        input_phases=list(range(0, 100, 10)),  # [0, 10, 20, ..., 90]
        target_phases=[],  # Will be populated dynamically
        description='High frame rate linear interpolation - 10 frames between each phase',
        params={'frames_per_interval': 10, 'mode': 'high_frame_rate'},
        slice_selection='representative_levels',
        priority=1
    ),
    
    'hfr_spline_8fps': ExperimentConfig(
        name='hfr_spline_8fps',
        method='spline',
        input_phases=list(range(0, 100, 10)),  # [0, 10, 20, ..., 90]
        target_phases=[],  # Will be populated dynamically
        description='High frame rate spline interpolation - 8 frames between each phase',
        params={'frames_per_interval': 8, 'mode': 'high_frame_rate'},
        slice_selection='representative_levels',
        priority=2
    ),
    
    # Original experiments (for testing flow matching model after training)
    'linear_0_50': ExperimentConfig(
        name='linear_0_50',
        method='linear',
        input_phases=[0, 50],
        target_phases=[10, 20, 30, 40],
        description='Linear interpolation between inspiration (0%) and expiration (50%)',
        slice_selection='representative_levels',
        priority=3
    ),
    
    'linear_0_100': ExperimentConfig(
        name='linear_0_100',
        method='linear',
        input_phases=[0, 100],
        target_phases=[10, 20, 30, 40, 50, 60, 70, 80, 90],
        description='Linear interpolation across full breathing cycle',
        slice_selection='representative_levels',
        priority=3
    ),
    
    # Test experiments for evaluating trained models (keep these for later)
    'test_sparse_0_50': ExperimentConfig(
        name='test_sparse_0_50',
        method='linear',
        input_phases=[0, 50],
        target_phases=[10, 20, 30, 40],
        description='Test interpolation with sparse inputs (for model evaluation)',
        slice_selection='representative_levels',
        priority=4
    ),
    
    'test_sparse_0_100': ExperimentConfig(
        name='test_sparse_0_100',
        method='linear',
        input_phases=[0, 100],
        target_phases=[10, 20, 30, 40, 50, 60, 70, 80, 90],
        description='Test interpolation across full cycle with only endpoints',
        slice_selection='representative_levels',
        priority=4
    ),
    
    # Spline interpolation
    'spline_3pt': ExperimentConfig(
        name='spline_3pt',
        method='spline',
        input_phases=[0, 50, 100],
        target_phases=[10, 20, 30, 40, 60, 70, 80, 90],
        description='Cubic spline interpolation with 3 control points',
        params={'spline_type': 'cubic'},
        slice_selection='representative_levels',
        priority=2
    ),
    
    # High frame rate optical flow
    'hfr_optical_flow_8fps': ExperimentConfig(
        name='hfr_optical_flow_8fps',
        method='optical_flow',
        input_phases=list(range(0, 100, 10)),  # [0, 10, 20, ..., 90]
        target_phases=[],  # Will be populated dynamically
        description='High frame rate optical flow interpolation - 8 frames between each phase',
        params={'frames_per_interval': 8, 'mode': 'high_frame_rate', 'flow_algorithm': 'farneback'},
        slice_selection='representative_levels',
        priority=2
    ),
    
    'hfr_optical_flow_5fps': ExperimentConfig(
        name='hfr_optical_flow_5fps',
        method='optical_flow',
        input_phases=list(range(0, 100, 10)),  # [0, 10, 20, ..., 90]
        target_phases=[],  # Will be populated dynamically
        description='High frame rate optical flow interpolation - 5 frames between each phase',
        params={'frames_per_interval': 5, 'mode': 'high_frame_rate', 'flow_algorithm': 'farneback'},
        slice_selection='representative_levels',
        priority=2
    ),
    
    # Standard optical flow (for testing)
    'optical_flow_0_50': ExperimentConfig(
        name='optical_flow_0_50',
        method='optical_flow',
        input_phases=[0, 50],
        target_phases=[10, 20, 30, 40],
        description='Optical flow guided interpolation',
        params={'flow_algorithm': 'farneback'},
        slice_selection='single_middle',
        priority=3
    ),
    

}


def get_experiment_by_priority(priority: int) -> List[ExperimentConfig]:
    """Get all experiments with given priority level"""
    return [exp for exp in PRESET_EXPERIMENTS.values() if exp.priority == priority]


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Validate experiment configuration"""
    # Check method is valid
    valid_methods = {'linear', 'spline', 'optical_flow'}
    if config.method not in valid_methods:
        raise ValueError(f"Invalid method: {config.method}. Must be one of {valid_methods}")
    
    # Check phases are in valid range
    all_phases = config.input_phases + config.target_phases
    if any(p < 0 or p > 100 for p in all_phases):
        raise ValueError("All phases must be between 0 and 100")
    
    # Check no overlap between input and target
    if set(config.input_phases) & set(config.target_phases):
        raise ValueError("Input and target phases must not overlap")
    
    # Check slice selection is valid
    valid_selections = {'all', 'middle_third', 'representative_levels', 'single_middle'}
    if config.slice_selection not in valid_selections:
        raise ValueError(f"Invalid slice_selection: {config.slice_selection}")