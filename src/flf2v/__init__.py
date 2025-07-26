"""
FLF2V Package - First-Last Frame to Video for Lung CT
Canonical imports for all components
"""

# Import canonical dataset and utilities
from .datasets import LungCTDataset, lungct_collate_fn

# Import model components
from .lungct_flf2v_model import LungCTFLF2V

# Package metadata
__version__ = "0.1.0"
__author__ = "Lung CT FLF2V Team"

# Canonical exports - single source of truth
__all__ = [
    # Dataset and utilities
    'LungCTDataset',
    'lungct_collate_fn',
    
    # Model (no trainer - use scripts/train_flf2v.py)
    'LungCTFLF2V',
]