"""
Data Pipeline for 4D CT Interpolation
Handles interpolation experiments with patient-level splits and slice-level training
"""

from .pipeline_manager import PipelineManager
from .experiment_configs import ExperimentConfig, PRESET_EXPERIMENTS
from .data_processor import DataProcessor
from .manifest_manager import ManifestManager
from .utils import Task, DataSample

__all__ = [
    'PipelineManager',
    'ExperimentConfig', 
    'PRESET_EXPERIMENTS',
    'DataProcessor',
    'ManifestManager',
    'Task',
    'DataSample'
]

__version__ = '0.1.0'