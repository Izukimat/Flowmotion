#!/usr/bin/env python
"""
Visualize interpolated HFR data as video or image sequence
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import cv2
from matplotlib.animation import FuncAnimation, PillowWriter
import sys

def load_interpolation_data(data_path):
    """Load interpolated frames and metadata"""
    input_frames = np.load(data_path / 'input_frames.npy')
    target_frames = np.load(data_path / 'target_frames.npy')
    
    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return input_frames, target_frames, metadata

def create_video(frames, output_path, fps=10):
    """Create MP4 video from frames"""
    height, width = frames[0].shape
    
    # Normalize frames to 0-255
    frames_normalized = []
    for frame in frames:
        frame_norm = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
        frames_normalized.append(frame_norm)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=False)
    
    for frame in frames_normalized:
        # Convert grayscale to BGR for video
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to: {output_path}")

def create_gif(frames, output_path, fps=10):
    """Create animated GIF from frames"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Normalize frames
    vmin = frames.min()
    vmax = frames.max()
    
    im = ax.imshow(frames[0], cmap='gray', vmin=vmin, vmax=vmax)
    ax.axis('off')
    
    def animate(i):
        im.set_array(frames[i])
        ax.set_title(f'Frame {i+1}/{len(frames)}')
        return [im]
    
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000/fps, blit=True)
    
    writer = PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer)
    plt.close()
    print(f"GIF saved to: {output_path}")

def create_montage(frames, output_path, max_frames=20):
    """Create a montage image showing multiple frames"""
    n_frames = min(len(frames), max_frames)
    
    # Sample frames evenly
    indices = np.linspace(0, len(frames)-1, n_frames, dtype=int)
    sampled_frames = frames[indices]
    
    # Calculate grid size
    cols = int(np.ceil(np.sqrt(n_frames)))
    rows = int(np.ceil(n_frames / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if n_frames > 1 else [axes]
    
    for i, (idx, frame) in enumerate(zip(indices, sampled_frames)):
        axes[i].imshow(frame, cmap='gray')
        axes[i].set_title(f'Frame {idx}')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_frames, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Montage saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize interpolated HFR data')
    parser.add_argument('data_path', type=Path, help='Path to interpolation data (contains .npy files)')
    parser.add_argument('--output-dir', type=Path, help='Output directory (default: same as input)')
    parser.add_argument('--format', choices=['video', 'gif', 'montage', 'all'], default='all',
                      help='Output format')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for video/gif')
    parser.add_argument('--show-info', action='store_true', help='Show information about the data')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.data_path.exists():
        print(f"Error: Path not found: {args.data_path}")
        sys.exit(1)
    
    # Load data
    try:
        input_frames, target_frames, metadata = load_interpolation_data(args.data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Show info if requested
    if args.show_info:
        print("\nInterpolation Data Info:")
        print(f"Input frames shape: {input_frames.shape}")
        print(f"Target frames shape: {target_frames.shape}")
        print(f"Experiment: {metadata['experiment']['name']}")
        print(f"Method: {metadata['experiment']['method']}")
        print(f"Is HFR: {metadata.get('is_hfr', False)}")
        print(f"Number of frames: {metadata.get('num_frames', target_frames.shape[0])}")
        print()
    
    # Set output directory
    output_dir = args.output_dir or args.data_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    if args.format in ['video', 'all']:
        video_path = output_dir / 'interpolated_sequence.mp4'
        create_video(target_frames, video_path, fps=args.fps)
    
    if args.format in ['gif', 'all']:
        gif_path = output_dir / 'interpolated_sequence.gif'
        create_gif(target_frames, gif_path, fps=args.fps)
    
    if args.format in ['montage', 'all']:
        montage_path = output_dir / 'frame_montage.png'
        create_montage(target_frames, montage_path)
    
    print("\nVisualization complete!")
    
    # Show comparison of methods if multiple experiments exist
    parent_dir = args.data_path.parent.parent.parent
    experiments = [d.name for d in parent_dir.iterdir() if d.is_dir() and 'hfr_' in d.name]
    if len(experiments) > 1:
        print(f"\nOther experiments available for comparison: {', '.join(experiments)}")

if __name__ == '__main__':
    main()