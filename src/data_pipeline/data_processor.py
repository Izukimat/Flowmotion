"""
Data processing module for various interpolation methods
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from scipy import interpolate
import logging
from concurrent.futures import ThreadPoolExecutor
import json

from .utils import Task, extract_slice_from_volume, save_interpolation_result, get_phase_indices
from .experiment_configs import ExperimentConfig


logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles the actual interpolation processing for different methods"""
    
    def __init__(self, processed_data_dir: Path, output_base_dir: Path):
        self.processed_data_dir = Path(os.path.expanduser(str(processed_data_dir)))
        self.output_base_dir = Path(os.path.expanduser(str(output_base_dir)))
        
        # Method dispatch table
        self.interpolation_methods = {
            'linear': self.linear_interpolation,
            'spline': self.spline_interpolation,
            'optical_flow': self.optical_flow_interpolation
        }
    
    def process_task(self, task: Task, experiment: ExperimentConfig) -> Tuple[bool, Optional[str]]:
        """
        Process a single interpolation task
        Returns: (success, error_message)
        """
        try:
            # Get output path
            output_path = task.get_output_path(self.output_base_dir)
            
            # Skip if already exists
            if output_path.exists() and (output_path / 'input_frames.npy').exists():
                logger.info(f"Skipping existing task: {output_path}")
                return True, None
            
            # Load input data
            input_data = self._load_input_data(task, experiment)
            if input_data is None:
                return False, "Failed to load input data"
            
            # Get interpolation method
            method = self.interpolation_methods.get(experiment.method)
            if method is None:
                return False, f"Unknown interpolation method: {experiment.method}"
            
            # Perform interpolation
            result = method(input_data, experiment)
            
            if result is None:
                return False, "Interpolation failed"
            
            input_frames, target_frames = result
            
            # Prepare metadata
            metadata = {
                'task': task.to_dict(),
                'experiment': experiment.to_dict(),
                'input_shape': input_frames.shape,
                'target_shape': target_frames.shape,
                'num_frames': target_frames.shape[0],
                'is_hfr': experiment.params.get('mode') == 'high_frame_rate',
                'phase_mapping': {
                    'input_phases': experiment.input_phases,
                    'target_phases': experiment.target_phases if not experiment.params.get('mode') == 'high_frame_rate' else 'generated'
                }
            }
            
            # Save results
            save_interpolation_result(output_path, input_frames, target_frames, metadata)
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            return False, str(e)
    
    def _load_input_data(self, task: Task, experiment: ExperimentConfig) -> Optional[Dict]:
        """Load the required input data for interpolation"""
        try:
            # Get paths to phase volumes
            paths = task.get_input_data_paths(self.processed_data_dir)
            cycle_dir = paths['cycle_dir']
            
            if not cycle_dir.exists():
                logger.error(f"Cycle directory not found: {cycle_dir}")
                return None
            
            # For high frame rate mode, we need to load input phases
            # Target phases will be generated dynamically
            phases_to_load = experiment.input_phases.copy()
            
            # Only add target phases if not in high frame rate mode
            if experiment.params.get('mode') != 'high_frame_rate':
                phases_to_load.extend(experiment.target_phases)
            
            # Load required phases
            phase_data = {}
            for phase_percent in phases_to_load:
                phase_idx = get_phase_indices(phase_percent)
                phase_file = cycle_dir / f'phase_{phase_idx:02d}.npy'
                
                if not phase_file.exists():
                    logger.error(f"Phase file not found: {phase_file}")
                    return None
                
                # Extract the specific slice
                slice_data = extract_slice_from_volume(phase_file, task.slice_num)
                phase_data[phase_percent] = slice_data
            
            return {
                'phase_data': phase_data,
                'slice_num': task.slice_num,
                'cycle_dir': cycle_dir
            }
            
        except Exception as e:
            logger.error(f"Error loading input data: {e}")
            return None
    
    def linear_interpolation(self, input_data: Dict, experiment: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linear interpolation - supports both standard and high frame rate modes
        """
        phase_data = input_data['phase_data']
        
        # Check if this is high frame rate mode
        if experiment.params.get('mode') == 'high_frame_rate':
            return self._high_frame_rate_interpolation(phase_data, experiment, method='linear')
        
        # Standard interpolation mode
        # Get input frames
        input_frames = []
        for phase in experiment.input_phases:
            input_frames.append(phase_data[phase])
        input_frames = np.stack(input_frames)
        
        # Interpolate target frames
        target_frames = []
        
        # For each target phase, interpolate between nearest input phases
        for target_phase in experiment.target_phases:
            # Find surrounding input phases
            lower_phase = max([p for p in experiment.input_phases if p <= target_phase], default=None)
            upper_phase = min([p for p in experiment.input_phases if p >= target_phase], default=None)
            
            if lower_phase is None or upper_phase is None:
                # Target phase is outside input range
                logger.warning(f"Target phase {target_phase} outside input range")
                # Use nearest phase
                if lower_phase is None:
                    target_frames.append(phase_data[upper_phase])
                else:
                    target_frames.append(phase_data[lower_phase])
            elif lower_phase == upper_phase:
                # Exact match
                target_frames.append(phase_data[lower_phase])
            else:
                # Linear interpolation
                alpha = (target_phase - lower_phase) / (upper_phase - lower_phase)
                interpolated = (1 - alpha) * phase_data[lower_phase] + alpha * phase_data[upper_phase]
                target_frames.append(interpolated)
        
        target_frames = np.stack(target_frames)
        
        return input_frames, target_frames
    
    def _high_frame_rate_interpolation(self, phase_data: Dict, experiment: ExperimentConfig, 
                                     method: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
        """
        High frame rate interpolation between consecutive phases
        Creates dense interpolation between each pair of adjacent phases
        """
        frames_per_interval = experiment.params.get('frames_per_interval', 8)
        input_phases = sorted(experiment.input_phases)
        
        # We'll return the original phases as "input" and all interpolated frames as "target"
        input_frames = []
        all_interpolated_frames = []
        frame_metadata = []  # Track which frames are original vs interpolated
        
        # Process each consecutive pair
        for i in range(len(input_phases) - 1):
            start_phase = input_phases[i]
            end_phase = input_phases[i + 1]
            
            start_frame = phase_data[start_phase]
            end_frame = phase_data[end_phase]
            
            # Add the start frame (original)
            if i == 0:  # Only add first frame once
                input_frames.append(start_frame)
                all_interpolated_frames.append(start_frame)
                frame_metadata.append({'phase': start_phase, 'original': True})
            
            # Interpolate frames between start and end
            for j in range(1, frames_per_interval + 1):
                alpha = j / (frames_per_interval + 1)
                
                if method == 'linear':
                    interpolated = (1 - alpha) * start_frame + alpha * end_frame
                elif method == 'spline' and len(phase_data) >= 4:
                    # For spline, we need more context
                    # Get 4 nearest phases for cubic interpolation
                    context_phases = self._get_spline_context(i, input_phases, 4)
                    context_frames = [phase_data[p] for p in context_phases]
                    
                    # Create interpolator
                    from scipy import interpolate
                    f = interpolate.interp1d(
                        context_phases, 
                        np.stack(context_frames).reshape(len(context_phases), -1),
                        kind='cubic', axis=0, fill_value='extrapolate'
                    )
                    
                    # Interpolate at target phase
                    target_phase = start_phase + alpha * (end_phase - start_phase)
                    interpolated = f(target_phase).reshape(start_frame.shape)
                else:
                    # Fallback to linear
                    interpolated = (1 - alpha) * start_frame + alpha * end_frame
                
                all_interpolated_frames.append(interpolated)
                frame_metadata.append({
                    'phase': start_phase + alpha * (end_phase - start_phase),
                    'original': False,
                    'interval': f"{start_phase}-{end_phase}"
                })
            
            # Add the end frame (original)
            input_frames.append(end_frame)
            all_interpolated_frames.append(end_frame)
            frame_metadata.append({'phase': end_phase, 'original': True})
        
        # Convert to numpy arrays
        input_frames = np.stack(input_frames)  # Original phases only
        all_frames = np.stack(all_interpolated_frames)  # All frames including interpolated
        
        # For training, we might want to return different combinations
        # Option 1: Return original frames as input, all frames as target
        # This helps the model learn to generate the full sequence
        
        # Option 2: Return pairs for supervised learning
        # Each interpolated frame paired with its surrounding context
        
        # For now, using Option 1 for simplicity
        return input_frames, all_frames
    
    def _get_spline_context(self, current_idx: int, phases: List[int], 
                           context_size: int = 4) -> List[int]:
        """Get surrounding phases for spline interpolation"""
        # Try to get equal number before and after
        half = context_size // 2
        start_idx = max(0, current_idx - half + 1)
        end_idx = min(len(phases), start_idx + context_size)
        
        # Adjust if we hit boundaries
        if end_idx - start_idx < context_size:
            if start_idx == 0:
                end_idx = min(len(phases), context_size)
            else:
                start_idx = max(0, end_idx - context_size)
        
        return phases[start_idx:end_idx]
    
    def spline_interpolation(self, input_data: Dict, experiment: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Spline-based interpolation through control points
        """
        phase_data = input_data['phase_data']
        
        # Check if this is high frame rate mode
        if experiment.params.get('mode') == 'high_frame_rate':
            return self._high_frame_rate_interpolation(phase_data, experiment, method='spline')
        
        # Standard spline interpolation
        # Get input frames
        input_frames = []
        for phase in experiment.input_phases:
            input_frames.append(phase_data[phase])
        input_frames = np.stack(input_frames)
        
        # Prepare for spline interpolation
        height, width = input_frames[0].shape
        target_frames = []
        
        # Create spline for each pixel
        for target_phase in experiment.target_phases:
            interpolated = np.zeros((height, width), dtype=np.float32)
            
            # Vectorized approach for efficiency
            # Flatten spatial dimensions
            input_flat = input_frames.reshape(len(experiment.input_phases), -1)
            
            # Create spline interpolator for all pixels at once
            if len(experiment.input_phases) >= 4:
                # Cubic spline
                f = interpolate.interp1d(experiment.input_phases, input_flat, 
                                       kind='cubic', axis=0, fill_value='extrapolate')
            else:
                # Linear if not enough points
                f = interpolate.interp1d(experiment.input_phases, input_flat, 
                                       kind='linear', axis=0, fill_value='extrapolate')
            
            # Interpolate
            interpolated_flat = f(target_phase)
            interpolated = interpolated_flat.reshape(height, width)
            
            # Ensure valid range
            interpolated = np.clip(interpolated, 0, input_frames.max())
            target_frames.append(interpolated)
        
        target_frames = np.stack(target_frames)
        
        return input_frames, target_frames
    
    def optical_flow_interpolation(self, input_data: Dict, experiment: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optical flow guided interpolation
        """
        phase_data = input_data['phase_data']
        
        # Check if this is high frame rate mode
        if experiment.params.get('mode') == 'high_frame_rate':
            return self._high_frame_rate_optical_flow(phase_data, experiment)
        
        # Standard optical flow interpolation
        # Get input frames
        input_frames = []
        for phase in experiment.input_phases:
            input_frames.append(phase_data[phase])
        input_frames = np.stack(input_frames)
        
        # For optical flow, we need at least 2 frames
        if len(experiment.input_phases) < 2:
            logger.error("Optical flow requires at least 2 input phases")
            return self.linear_interpolation(input_data, experiment)
        
        target_frames = []
        
        def normalize_frames_consistently(frame1, frame2):
            """Normalize both frames using the same scale"""
            frame1 = np.asarray(frame1, dtype=np.float64)
            frame2 = np.asarray(frame2, dtype=np.float64)
            
            global_min = min(frame1.min(), frame2.min())
            global_max = max(frame1.max(), frame2.max())
            
            if global_max <= global_min:
                return (np.full_like(frame1, 128, dtype=np.uint8), 
                       np.full_like(frame2, 128, dtype=np.uint8),
                       global_min, global_max)
            
            frame1_norm = (frame1 - global_min) / (global_max - global_min) * 255
            frame2_norm = (frame2 - global_min) / (global_max - global_min) * 255
            
            frame1_uint8 = np.clip(frame1_norm, 0, 255).astype(np.uint8)
            frame2_uint8 = np.clip(frame2_norm, 0, 255).astype(np.uint8)
            
            return frame1_uint8, frame2_uint8, global_min, global_max
        
        # Process each target phase
        for target_phase in experiment.target_phases:
            # Find nearest two input phases
            sorted_phases = sorted(experiment.input_phases)
            
            # Find bracketing phases
            lower_idx = 0
            upper_idx = len(sorted_phases) - 1
            
            for i in range(len(sorted_phases) - 1):
                if sorted_phases[i] <= target_phase <= sorted_phases[i + 1]:
                    lower_idx = i
                    upper_idx = i + 1
                    break
            
            lower_phase = sorted_phases[lower_idx]
            upper_phase = sorted_phases[upper_idx]
            
            if lower_phase == upper_phase:
                # Same phase
                target_frames.append(phase_data[lower_phase])
                continue
            
            try:
                # Normalize frames consistently
                frame1, frame2, global_min, global_max = normalize_frames_consistently(
                    phase_data[lower_phase], phase_data[upper_phase])
                
                # Check for sufficient contrast
                if (np.std(frame1) < 1.0 or np.std(frame2) < 1.0):
                    logger.warning(f"Low contrast frames for phases {lower_phase}-{upper_phase}, using linear interpolation")
                    alpha = (target_phase - lower_phase) / (upper_phase - lower_phase)
                    interpolated = (1 - alpha) * phase_data[lower_phase] + alpha * phase_data[upper_phase]
                    target_frames.append(interpolated)
                    continue
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    frame1, frame2, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                # Check if optical flow succeeded
                if flow is None or np.any(np.isnan(flow)) or np.any(np.isinf(flow)):
                    logger.warning(f"Optical flow failed for phases {lower_phase}-{upper_phase}, using linear interpolation")
                    alpha = (target_phase - lower_phase) / (upper_phase - lower_phase)
                    interpolated = (1 - alpha) * phase_data[lower_phase] + alpha * phase_data[upper_phase]
                    target_frames.append(interpolated)
                    continue
                
                # Interpolation factor
                alpha = (target_phase - lower_phase) / (upper_phase - lower_phase)
                
                # Apply weighted flow
                h, w = frame1.shape
                flow_scaled = flow * alpha
                
                # Create mesh grid
                row_coords, col_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                
                # Apply flow displacement
                displaced_rows = row_coords + flow_scaled[..., 1]
                displaced_cols = col_coords + flow_scaled[..., 0]
                
                # Remap the image
                interpolated_uint8 = cv2.remap(
                    frame1.astype(np.float32),
                    displaced_cols.astype(np.float32),
                    displaced_rows.astype(np.float32),
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )
                
                # Convert back to original scale using consistent global scale
                interpolated = interpolated_uint8 / 255.0 * (global_max - global_min) + global_min
                interpolated = np.clip(interpolated, global_min, global_max)
                
                target_frames.append(interpolated)
            
            except Exception as e:
                logger.error(f"Optical flow failed for phases {lower_phase}-{upper_phase}: {e}. Using linear interpolation.")
                # Fall back to linear interpolation
                alpha = (target_phase - lower_phase) / (upper_phase - lower_phase)
                interpolated = (1 - alpha) * phase_data[lower_phase] + alpha * phase_data[upper_phase]
                target_frames.append(interpolated)
        
        target_frames = np.stack(target_frames)
        
        return input_frames, target_frames
    
    def _high_frame_rate_optical_flow(self, phase_data: Dict, 
                                    experiment: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        High frame rate optical flow interpolation between consecutive phases
        More accurate than standard interpolation as phases are closer together
        """
        frames_per_interval = experiment.params.get('frames_per_interval', 8)
        flow_algorithm = experiment.params.get('flow_algorithm', 'farneback')
        input_phases = sorted(experiment.input_phases)
        
        input_frames = []
        all_interpolated_frames = []
        
        # Process each consecutive pair
        for i in range(len(input_phases) - 1):
            start_phase = input_phases[i]
            end_phase = input_phases[i + 1]
            
            start_frame = phase_data[start_phase]
            end_frame = phase_data[end_phase]
            
            # Add the start frame
            if i == 0:
                input_frames.append(start_frame)
                all_interpolated_frames.append(start_frame)
            
            # Robust normalization to uint8 for optical flow
            def normalize_frames_consistently(frame1, frame2):
                """
                Normalize both frames using the same scale to preserve relative intensities
                This is crucial for optical flow to work correctly
                """
                frame1 = np.asarray(frame1, dtype=np.float64)
                frame2 = np.asarray(frame2, dtype=np.float64)
                
                # Find global min/max across both frames
                global_min = min(frame1.min(), frame2.min())
                global_max = max(frame1.max(), frame2.max())
                
                # Handle edge cases
                if global_max <= global_min:
                    logger.warning(f"Frames have no dynamic range (min={global_min}, max={global_max})")
                    # Return mid-gray frames
                    return (np.full_like(frame1, 128, dtype=np.uint8), 
                           np.full_like(frame2, 128, dtype=np.uint8),
                           global_min, global_max)
                
                # Normalize both frames using the same scale
                frame1_norm = (frame1 - global_min) / (global_max - global_min) * 255
                frame2_norm = (frame2 - global_min) / (global_max - global_min) * 255
                
                # Ensure valid range and convert to uint8
                frame1_uint8 = np.clip(frame1_norm, 0, 255).astype(np.uint8)
                frame2_uint8 = np.clip(frame2_norm, 0, 255).astype(np.uint8)
                
                return frame1_uint8, frame2_uint8, global_min, global_max
            
            try:
                frame1_uint8, frame2_uint8, global_min, global_max = normalize_frames_consistently(start_frame, end_frame)
                
                # Check if frames are valid for optical flow (not completely uniform)
                if (np.std(frame1_uint8) < 1.0 or np.std(frame2_uint8) < 1.0):
                    logger.warning(f"Low contrast frames detected for phases {start_phase}-{end_phase}, using linear interpolation")
                    # Fall back to linear interpolation for this interval
                    for j in range(1, frames_per_interval + 1):
                        alpha = j / (frames_per_interval + 1)
                        interpolated = (1 - alpha) * start_frame + alpha * end_frame
                        all_interpolated_frames.append(interpolated)
                    continue
                
                # Calculate optical flow between consecutive phases
                if flow_algorithm == 'farneback':
                    flow = cv2.calcOpticalFlowFarneback(
                        frame1_uint8, frame2_uint8, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )
                else:
                    # Fallback to Farneback
                    flow = cv2.calcOpticalFlowFarneback(
                        frame1_uint8, frame2_uint8, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )
                
                # Check if optical flow calculation succeeded
                if flow is None or np.any(np.isnan(flow)) or np.any(np.isinf(flow)):
                    logger.warning(f"Optical flow failed for phases {start_phase}-{end_phase}, using linear interpolation")
                    # Fall back to linear interpolation
                    for j in range(1, frames_per_interval + 1):
                        alpha = j / (frames_per_interval + 1)
                        interpolated = (1 - alpha) * start_frame + alpha * end_frame
                        all_interpolated_frames.append(interpolated)
                    continue
                
                # Create mesh grid for remapping
                h, w = start_frame.shape
                row_coords, col_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                
                # Interpolate frames using optical flow
                for j in range(1, frames_per_interval + 1):
                    alpha = j / (frames_per_interval + 1)
                    
                    # Scale flow by interpolation factor
                    flow_scaled = flow * alpha
                    
                    # Apply flow displacement
                    displaced_rows = row_coords + flow_scaled[..., 1]
                    displaced_cols = col_coords + flow_scaled[..., 0]
                    
                    # Remap the image using the flow
                    interpolated_uint8 = cv2.remap(
                        frame1_uint8.astype(np.float32),
                        displaced_cols.astype(np.float32),
                        displaced_rows.astype(np.float32),
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT
                    )
                    
                    # Blend with linear interpolation for stability
                    # (pure optical flow can sometimes create artifacts)
                    linear_interp = (1 - alpha) * frame1_uint8.astype(np.float32) + alpha * frame2_uint8.astype(np.float32)
                    
                    # Weighted combination (more trust in optical flow for small displacements)
                    flow_weight = 0.8  # Can be tuned
                    interpolated_uint8 = flow_weight * interpolated_uint8 + (1 - flow_weight) * linear_interp
                    
                    # Convert back to original scale using the consistent global scale
                    interpolated = interpolated_uint8 / 255.0 * (global_max - global_min) + global_min
                    
                    # Ensure the result is in a reasonable range
                    interpolated = np.clip(interpolated, global_min, global_max)
                    
                    all_interpolated_frames.append(interpolated)
            
            except Exception as e:
                logger.error(f"Optical flow failed for phases {start_phase}-{end_phase}: {e}. Using linear interpolation.")
                # Fall back to linear interpolation for this entire interval
                for j in range(1, frames_per_interval + 1):
                    alpha = j / (frames_per_interval + 1)
                    interpolated = (1 - alpha) * start_frame + alpha * end_frame
                    all_interpolated_frames.append(interpolated)
            
            # Add the end frame
            input_frames.append(end_frame)
            all_interpolated_frames.append(end_frame)
        
        # Convert to numpy arrays
        input_frames = np.stack(input_frames)
        all_frames = np.stack(all_interpolated_frames)
        
        return input_frames, all_frames

    
    def process_batch(self, tasks: List[Task], experiment: ExperimentConfig, 
                     max_workers: int = 4) -> Dict[str, List[Task]]:
        """
        Process multiple tasks in parallel
        Returns dict with 'success' and 'failed' task lists
        """
        results = {'success': [], 'failed': []}
        
        def process_single(task):
            success, error = self.process_task(task, experiment)
            return task, success, error
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single, task) for task in tasks]
            
            for future in futures:
                task, success, error = future.result()
                if success:
                    results['success'].append(task)
                else:
                    results['failed'].append(task)
                    logger.error(f"Failed task: {task.to_dict()}, Error: {error}")
        
        return results