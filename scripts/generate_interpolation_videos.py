#!/usr/bin/env python3
"""
Generate high-frame-rate interpolation videos from a single 4D-Lung study.

Pipeline overview:
- Load one patient's single study from raw DICOMs (downloaded via NBIA).
- Detect respiratory-gated CT series and sort by breathing phase percent.
- Build 3D volumes per phase and select 3 representative axial slices
  (25%, 50%, 75% of the stack) as in the data pipeline.
- Perform high-frame-rate interpolation between consecutive phases using:
  1) Optical flow (Farneback), 2) Linear, 3) Spline.
- For each method, produce:
  - Full 0–90% sequence (82 frames when frames_per_interval=8 across 9 intervals).
  - A 0–50% video that tiles the 3 slices horizontally per frame.

Notes:
- This script mirrors the logic used in src/data_pipeline/data_processor.py
  for interpolation methods, and src/preprocess.py for DICOM -> volume.
- It does not write the per-phase volumes to disk; it works in-memory.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
import subprocess

import cv2
import numpy as np
import pydicom
from scipy import interpolate as sp_interpolate


def extract_breathing_phase(series_name: str) -> float | None:
    """Extract breathing phase percentage from a series/folder name.

    Accepts patterns like "50.0%", "70%", etc. Returns None if not found.
    """
    match = re.search(r"(\d+\.?\d*)%", series_name)
    if match:
        return float(match.group(1))
    return None


def is_respiratory_gated_ct_name(series_name: str) -> bool:
    """Loose filter for gated CT using only the folder/series name.

    Accepts names containing a percentage pattern (e.g., '0.0%') and not obviously
    excluded. We no longer require the literal word 'gated' because NBIA folders
    can be UIDs without descriptive names.
    """
    s = series_name.lower()
    if "%" not in s:
        return False
    exclude_keywords = ["planning", "scout", "localizer", "dose", "struct", "rtss"]
    if any(k in s for k in exclude_keywords):
        return False
    return extract_breathing_phase(series_name) is not None


def extract_phase_from_dicom(ds: pydicom.dataset.FileDataset) -> float | None:
    """Extract breathing phase percent from common DICOM text fields.

    Tries SeriesDescription, ProtocolName, StudyDescription, ImageComments.
    Returns None if not found.
    """
    candidates = [
        str(getattr(ds, 'SeriesDescription', '') or ''),
        str(getattr(ds, 'ProtocolName', '') or ''),
        str(getattr(ds, 'StudyDescription', '') or ''),
        str(getattr(ds, 'ImageComments', '') or ''),
    ]
    for text in candidates:
        phase = extract_breathing_phase(text)
        if phase is not None:
            return phase
    return None


def load_series_volume(series_dir: Path) -> np.ndarray:
    """Read a DICOM series directory into a 3D numpy volume [slices, H, W]."""
    dicom_files = sorted(series_dir.glob("*.dcm"))
    if not dicom_files:
        dicom_files = sorted(series_dir.rglob("*.dcm"))
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files in: {series_dir}")

    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    shapes = [s.pixel_array.shape for s in slices]
    if len(set(shapes)) > 1:
        raise ValueError(f"Non-uniform slice shapes found in {series_dir}: {set(shapes)}")

    volume = np.stack([s.pixel_array for s in slices], axis=0)
    return volume.astype(np.float32)


def select_representative_slices(num_slices: int) -> List[int]:
    """25%, 50%, 75% axial slice indices, clamped into valid range."""
    levels = [0.25, 0.5, 0.75]
    indices = [int(num_slices * lvl) for lvl in levels]
    indices = [min(max(0, i), num_slices - 1) for i in indices]
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for i in indices:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def build_phase_slice_dict(volumes: List[np.ndarray], slice_idx: int) -> Dict[int, np.ndarray]:
    """Map nominal breathing phase percent -> 2D slice image from corresponding volume.

    Assumes volumes are sorted by actual breathing phase percent ascending and
    correspond to nominal 0,10,20,... phases.
    """
    phase_percents = list(range(0, len(volumes) * 10, 10))
    data: Dict[int, np.ndarray] = {}
    for p, vol in zip(phase_percents, volumes):
        if slice_idx >= vol.shape[0]:
            raise ValueError(f"Slice {slice_idx} out of bounds for volume with {vol.shape[0]} slices")
        data[p] = vol[slice_idx].copy()
    return data


def hfr_linear_or_spline(phase_data: Dict[int, np.ndarray], frames_per_interval: int, method: str) -> Tuple[np.ndarray, np.ndarray]:
    """High-frame-rate interpolation using linear or spline between consecutive phases.

    Returns (input_frames, all_frames), where input_frames includes only the original
    phase frames, and all_frames contains originals + interpolated frames in order.
    """
    input_phases = sorted(phase_data.keys())
    input_frames = []
    all_frames = []

    for i in range(len(input_phases) - 1):
        start = input_phases[i]
        end = input_phases[i + 1]
        f0 = phase_data[start]
        f1 = phase_data[end]

        # Add the start frame once
        if i == 0:
            input_frames.append(f0)
            all_frames.append(f0)

        for j in range(1, frames_per_interval + 1):
            alpha = j / (frames_per_interval + 1)
            if method == "linear":
                interp = (1 - alpha) * f0 + alpha * f1
            elif method == "spline" and len(input_phases) >= 4:
                # Use 4-point cubic context when available
                ctx_idx = max(0, i - 1)
                ctx_phases = input_phases[ctx_idx:ctx_idx + 4]
                if len(ctx_phases) < 4:
                    ctx_phases = input_phases[max(0, len(input_phases) - 4):]
                ctx_frames = [phase_data[p] for p in ctx_phases]
                f = sp_interpolate.interp1d(
                    ctx_phases, np.stack(ctx_frames).reshape(len(ctx_phases), -1),
                    kind='cubic', axis=0, fill_value='extrapolate'
                )
                target_phase = start + alpha * (end - start)
                interp = f(target_phase).reshape(f0.shape)
            else:
                interp = (1 - alpha) * f0 + alpha * f1

            all_frames.append(interp)

        # Add the end frame
        input_frames.append(f1)
        all_frames.append(f1)

    return np.stack(input_frames), np.stack(all_frames)


def hfr_optical_flow(phase_data: Dict[int, np.ndarray], frames_per_interval: int) -> Tuple[np.ndarray, np.ndarray]:
    """High-frame-rate optical flow interpolation between consecutive phases (Farneback)."""
    def normalize_pair_to_uint8(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        gmin = min(a.min(), b.min())
        gmax = max(a.max(), b.max())
        if gmax <= gmin:
            return np.full_like(a, 128, np.uint8), np.full_like(b, 128, np.uint8), gmin, gmax
        a_u8 = np.clip((a - gmin) / (gmax - gmin) * 255.0, 0, 255).astype(np.uint8)
        b_u8 = np.clip((b - gmin) / (gmax - gmin) * 255.0, 0, 255).astype(np.uint8)
        return a_u8, b_u8, gmin, gmax

    input_phases = sorted(phase_data.keys())
    input_frames = []
    all_frames = []

    for i in range(len(input_phases) - 1):
        start = input_phases[i]
        end = input_phases[i + 1]
        f0 = phase_data[start]
        f1 = phase_data[end]

        # Add the start frame once
        if i == 0:
            input_frames.append(f0)
            all_frames.append(f0)

        try:
            f0_u8, f1_u8, gmin, gmax = normalize_pair_to_uint8(f0, f1)
            if np.std(f0_u8) < 1.0 or np.std(f1_u8) < 1.0:
                # Fallback to linear for low-contrast pairs
                for j in range(1, frames_per_interval + 1):
                    alpha = j / (frames_per_interval + 1)
                    interp = (1 - alpha) * f0 + alpha * f1
                    all_frames.append(interp)
                input_frames.append(f1)
                all_frames.append(f1)
                continue

            flow = cv2.calcOpticalFlowFarneback(
                f0_u8, f1_u8, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            if flow is None or np.any(np.isnan(flow)) or np.any(np.isinf(flow)):
                # Fallback to linear on failure
                for j in range(1, frames_per_interval + 1):
                    alpha = j / (frames_per_interval + 1)
                    interp = (1 - alpha) * f0 + alpha * f1
                    all_frames.append(interp)
                input_frames.append(f1)
                all_frames.append(f1)
                continue

            h, w = f0.shape
            rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

            for j in range(1, frames_per_interval + 1):
                alpha = j / (frames_per_interval + 1)
                flow_scaled = flow * alpha
                dr = rr + flow_scaled[..., 1]
                dc = cc + flow_scaled[..., 0]
                warped = cv2.remap(
                    f0_u8.astype(np.float32),
                    dc.astype(np.float32), dr.astype(np.float32),
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                )
                # Blend with linear interpolation for stability
                lin = (1 - alpha) * f0_u8.astype(np.float32) + alpha * f1_u8.astype(np.float32)
                out_u8 = 0.8 * warped + 0.2 * lin
                out = out_u8 / 255.0 * (gmax - gmin) + gmin
                out = np.clip(out, gmin, gmax)
                all_frames.append(out)

        except Exception:
            # Robust fallback: linear interpolation for the whole interval
            for j in range(1, frames_per_interval + 1):
                alpha = j / (frames_per_interval + 1)
                interp = (1 - alpha) * f0 + alpha * f1
                all_frames.append(interp)

        # Add the end frame
        input_frames.append(f1)
        all_frames.append(f1)

    return np.stack(input_frames), np.stack(all_frames)


def normalize_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Normalize a single 2D frame to 0–255 and convert to BGR for video."""
    f = frame.astype(np.float32)
    mn, mx = float(f.min()), float(f.max())
    if mx <= mn:
        g = np.zeros_like(f, dtype=np.uint8)
    else:
        g = ((f - mn) / (mx - mn) * 255.0).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def normalize_to_bgr_global(frame: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    f = frame.astype(np.float32)
    if vmax <= vmin:
        g = np.zeros_like(f, dtype=np.uint8)
    else:
        g = ((f - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _open_video_writer_try_codecs(path: Path, fps: float, size: Tuple[int, int], codec_prefs: List[str]) -> Tuple[cv2.VideoWriter | None, str | None, Path]:
    """Try opening a VideoWriter with a list of codecs, return (writer, codec_used, actual_path).

    If H.264/MP4 codecs fail, fallback to MJPG/AVI and adjust extension accordingly.
    """
    w, h = size
    # First attempt with provided codecs on the given path
    for c in codec_prefs:
        try:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*c), fps, (w, h))
            if writer.isOpened():
                return writer, c, path
        except Exception:
            pass

    # Fallback to MJPG in AVI container
    avi_path = path.with_suffix('.avi')
    try:
        writer = cv2.VideoWriter(str(avi_path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
        if writer.isOpened():
            return writer, 'MJPG', avi_path
    except Exception:
        pass

    return None, None, path


def _ffmpeg_transcode_to_h264(input_path: Path, output_path: Path, fps: float = 5.0) -> bool:
    """Transcode an existing video to H.264 (baseline, yuv420p) if ffmpeg is available.

    Returns True on success, False otherwise.
    """
    if shutil.which('ffmpeg') is None:
        return False
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', str(input_path),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-profile:v', 'baseline', '-level', '3.0',
        '-movflags', '+faststart',
        '-r', str(fps),
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False


def _ffmpeg_transcode_to_webm(input_path: Path, output_path: Path, fps: float = 5.0) -> bool:
    """Transcode an existing video to WebM (VP9) for maximum VS Code/Linux compatibility.

    Falls back to VP8 if libvpx-vp9 is unavailable. Returns True on success.
    """
    if shutil.which('ffmpeg') is None:
        return False
    # Try VP9 first
    cmd_vp9 = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', str(input_path),
        '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuv420p',
        '-b:v', '0', '-crf', '32',
        '-r', str(fps),
        str(output_path)
    ]
    try:
        subprocess.run(cmd_vp9, check=True)
        return True
    except Exception:
        pass

    # Fallback to VP8
    cmd_vp8 = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', str(input_path),
        '-c:v', 'libvpx', '-pix_fmt', 'yuv420p',
        '-b:v', '2M',
        '-r', str(fps),
        str(output_path)
    ]
    try:
        subprocess.run(cmd_vp8, check=True)
        return True
    except Exception:
        return False


def save_tiled_video(frames: List[np.ndarray], out_path: Path, fps: float = 5.0, codec_pref: str = 'avc1', transcode_h264: bool = False, write_webm: bool = False) -> Path:
    """Save a list of 3 grayscale frames (per time step) as a single horizontal-tiled video.

    Returns the actual output video path used (may switch to .avi if H.264 is unavailable).
    """
    assert len(frames) > 0
    # Each item in frames is [H, 3*W, 3] already stacked BGR
    h, w = frames[0].shape[:2]

    # Try web-friendly codecs first (H.264 variants), then mp4v, then MJPG/AVI
    codec_order = [codec_pref, 'H264', 'avc3', 'X264', 'mp4v']
    writer, used, actual_path = _open_video_writer_try_codecs(out_path, fps, (w, h), codec_order)
    if writer is None or not writer.isOpened():
        raise RuntimeError("Failed to open video writer for any supported codec.")

    for f in frames:
        writer.write(f)
    writer.release()

    print(f"Video saved to {actual_path.name} using codec {used}")

    # Optional: transcode to H.264 if current codec is not webview-friendly
    if transcode_h264 and used not in {'avc1', 'H264', 'avc3', 'X264'}:
        h264_path = actual_path.with_name(actual_path.stem + '_h264.mp4')
        ok = _ffmpeg_transcode_to_h264(actual_path, h264_path, fps=fps)
        if ok:
            print(f"Transcoded to H.264: {h264_path.name}")
            actual_path = h264_path
        else:
            print("ffmpeg not available or transcode failed; keeping original video.")

    # Optional: also write a WebM version for VS Code/Linux webview compatibility
    if write_webm:
        webm_path = actual_path.with_suffix('')
        webm_path = webm_path.with_name(webm_path.name + '.webm')
        ok = _ffmpeg_transcode_to_webm(actual_path, webm_path, fps=fps)
        if ok:
            print(f"Also wrote WebM: {webm_path.name}")
        else:
            print("WebM transcode failed (ffmpeg/libvpx missing?).")

    return actual_path


def save_decile_pngs_for_slice(
    out_dir: Path,
    method_name: str,
    slice_idx: int,
    decile_percents: List[int],
    frames_full: np.ndarray,
    frames_per_interval: int,
    original_decile_frames: List[np.ndarray] | None = None,
):
    """Save PNG images at each 10% decile for a single slice.

    - frames_full: full HFR sequence (originals + in-betweens) as [T, H, W]
    - decile_percents: e.g., [0,10,20,30,40,50]
    - original_decile_frames: if provided, also save the original phase frames for comparison.
    """
    dec_dir = out_dir / "deciles" / f"slice_{slice_idx:04d}"
    dec_dir.mkdir(parents=True, exist_ok=True)

    step = frames_per_interval + 1
    # Compute global min/max for consistent normalization across deciles
    picked_frames = []
    for p in decile_percents:
        idx = (p // 10) * step
        if idx < len(frames_full):
            picked_frames.append(frames_full[idx])
    if len(picked_frames) == 0:
        return
    vmin = float(min(f.min() for f in picked_frames))
    vmax = float(max(f.max() for f in picked_frames))
    if vmax <= vmin:
        vmin, vmax = 0.0, 1.0

    # Save method deciles
    for p in decile_percents:
        idx = (p // 10) * step
        if idx >= len(frames_full):
            continue
        f = frames_full[idx]
        # Normalize using global min/max
        g = ((f - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(dec_dir / f"{method_name}_{p:03d}.png"), g)

    # Optionally save originals
    if original_decile_frames is not None and len(original_decile_frames) > 0:
        ovmin = float(min(fr.min() for fr in original_decile_frames))
        ovmax = float(max(fr.max() for fr in original_decile_frames))
        if ovmax <= ovmin:
            ovmin, ovmax = 0.0, 1.0
        for p, fr in zip(decile_percents, original_decile_frames):
            g = ((fr - ovmin) / (ovmax - ovmin) * 255.0).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(dec_dir / f"original_{p:03d}.png"), g)


def main():
    parser = argparse.ArgumentParser(description="Generate HFR interpolation videos for 3 slices.")
    parser.add_argument("study_path", type=Path,
                        help="Path to the single study directory containing many series folders with DICOMs")
    parser.add_argument("output_dir", type=Path,
                        help="Directory to write outputs (videos and .npy arrays)")
    parser.add_argument("--frames-per-interval", type=int, default=8,
                        help="Number of interpolated frames between consecutive phases (default: 8)")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Frames per second for output videos")
    parser.add_argument("--phase-end", type=int, default=50,
                        help="Video shows frames from 0%% up to this phase percent (default: 50)")
    parser.add_argument("--save-deciles", action="store_true",
                        help="Also save PNGs for every 10% (0,10,...,phase_end) for each method and originals")
    parser.add_argument("--orig-decile-video", action="store_true",
                        help="Also export a video composed only of original deciles (no interpolation)")

    args = parser.parse_args()
    study_path: Path = args.study_path
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Discover gated CT series in this study
    series_info = []
    for series_dir in sorted([d for d in study_path.iterdir() if d.is_dir()]):
        name = series_dir.name
        phase = None

        # Strategy 1: detect from directory name (sometimes descriptive)
        if is_respiratory_gated_ct_name(name):
            phase = extract_breathing_phase(name)

        # Strategy 2: detect from DICOM metadata inside UID-named folders (NBIA layout)
        if phase is None:
            try:
                sample_dcm = next(series_dir.rglob('*.dcm'))
            except StopIteration:
                sample_dcm = None
            if sample_dcm is not None:
                try:
                    ds = pydicom.dcmread(sample_dcm, stop_before_pixels=True, force=True)
                    phase = extract_phase_from_dicom(ds)
                except Exception:
                    # Ignore unreadable DICOMs and continue
                    pass

        if phase is not None and 0.0 <= phase <= 100.0:
            series_info.append((phase, series_dir))

    if len(series_info) < 6:
        print(f"Found only {len(series_info)} candidate phase series.")
        print("Tip: Try a different study UID for this patient, or lower --phase-end to 40.")
        raise RuntimeError(f"Insufficient breathing phases in study: need at least 6 for 0–50%.")

    # Sort by actual breathing phase percent and report
    series_info.sort(key=lambda x: x[0])
    print("Discovered breathing phases:", [round(p,1) for p,_ in series_info])

    # Expect 10 phases for full 0–90% interpolation (82 frames with 8 per interval)
    if len(series_info) >= 10:
        series_info = series_info[:10]
    else:
        # Use what's available but warn if not 10
        print(f"Warning: {len(series_info)} phases available; full 0–90% HFR may produce fewer frames.")

    # 2) Load volumes per phase
    volumes: List[np.ndarray] = []
    percents: List[float] = []
    for p, d in series_info:
        vol = load_series_volume(d)
        volumes.append(vol)
        percents.append(p)

    # Sanity: enforce consistent shape across phases
    shapes = [v.shape for v in volumes]
    if len(set(shapes)) != 1:
        raise ValueError(f"Volumes have inconsistent shapes across phases: {set(shapes)}")

    num_slices = volumes[0].shape[0]
    h, w = volumes[0].shape[1:]
    slice_indices = select_representative_slices(num_slices)
    print(f"Selected slices (indices of axial stack): {slice_indices}")

    # Defaults: enable motion-friendly visualization without extra flags
    DO_GLOBAL_NORMALIZE = True
    DO_OVERLAY_PHASE = True
    DO_TRANSCODE_H264 = True
    DO_WRITE_WEBM = True

    # 3) Interpolate for each method and slice
    frames_per_interval = args.frames_per_interval
    methods = [
        ("optical_flow", lambda pd: hfr_optical_flow(pd, frames_per_interval)),
        ("linear",       lambda pd: hfr_linear_or_spline(pd, frames_per_interval, method="linear")),
        ("spline",       lambda pd: hfr_linear_or_spline(pd, frames_per_interval, method="spline")),
    ]

    # To build videos of 0–phase_end from the full HFR stream, compute cutoff length
    # For standard 10% steps: intervals = phase_end/10, frames = intervals*(N+1)+1
    intervals_for_video = args.phase_end // 10
    video_len = intervals_for_video * (frames_per_interval + 1) + 1

    for method_name, fn in methods:
        print(f"\n=== {method_name} ===")
        stacked_video_frames: List[np.ndarray] = []  # list of [H, 3*W, 3] frames

        # For each selected slice, run interpolation and save .npy
        per_slice_full_sequences: List[np.ndarray] = []
        per_slice_video_sequences: List[np.ndarray] = []

        for s_idx in slice_indices:
            # Map nominal 0,10,... to this slice
            phase_data = build_phase_slice_dict(volumes, s_idx)

            # Compute HFR sequence
            input_frames, all_frames = fn(phase_data)

            # Save full sequence (e.g., 82 frames for 0–90 with 8 per interval)
            npy_full_path = out_dir / f"{method_name}_slice_{s_idx:04d}_full.npy"
            np.save(npy_full_path, all_frames)

            # Take 0–phase_end segment for video
            video_seq = all_frames[:video_len]
            npy_video_path = out_dir / f"{method_name}_slice_{s_idx:04d}_0-{args.phase_end}.npy"
            np.save(npy_video_path, video_seq)

            # Optionally save decile PNGs for this slice and method
            if args.save_deciles:
                # Build original decile frames from the raw volumes
                deciles = list(range(0, args.phase_end + 1, 10))
                original_decile_frames = []
                # Volumes are in nominal order 0,10,...; map directly
                for p in deciles:
                    vol_idx = p // 10
                    if vol_idx < len(volumes):
                        original_decile_frames.append(volumes[vol_idx][s_idx])
                save_decile_pngs_for_slice(
                    out_dir,
                    method_name,
                    s_idx,
                    deciles,
                    all_frames,
                    frames_per_interval,
                    original_decile_frames=original_decile_frames,
                )

            per_slice_full_sequences.append(all_frames)
            per_slice_video_sequences.append(video_seq)

        # Build a single tiled video (3 slices side-by-side) for this method
        # Align by the shortest video length among slices (should be equal)
        min_len = min(seq.shape[0] for seq in per_slice_video_sequences)

        # Compute global min/max across the clip for consistent brightness
        vmin = float(min(seq[:min_len].min() for seq in per_slice_video_sequences))
        vmax = float(max(seq[:min_len].max() for seq in per_slice_video_sequences))

        for t in range(min_len):
            tiles = []
            for seq in per_slice_video_sequences:
                tiles.append(normalize_to_bgr_global(seq[t], vmin, vmax))
            tiled = np.concatenate(tiles, axis=1)  # [H, 3*W, 3]
            # Overlay phase percent text
            step = frames_per_interval + 1
            base = (t // step) * 10
            frac = (t % step) / (frames_per_interval + 1)
            phase_pct = base + frac * 10.0
            cv2.putText(tiled, f"~{phase_pct:.1f}%", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
            stacked_video_frames.append(tiled)

        video_path = out_dir / f"{method_name}_0-{args.phase_end}.mp4"
        actual_path = save_tiled_video(
            stacked_video_frames, video_path, fps=args.fps,
            codec_pref='avc1', transcode_h264=DO_TRANSCODE_H264,
            write_webm=DO_WRITE_WEBM
        )
        print(f"Saved video: {actual_path}")

        # Optional: also export original decile-only video (0,10,...,phase_end)
        if args.orig_decile_video:
            deciles = list(range(0, args.phase_end + 1, 10))
            orig_sequences: List[np.ndarray] = []
            for s_idx in slice_indices:
                frames = []
                for p in deciles:
                    vol_idx = p // 10
                    if vol_idx < len(volumes):
                        frames.append(volumes[vol_idx][s_idx])
                if frames:
                    orig_sequences.append(np.stack(frames))
            if orig_sequences:
                min_len2 = min(seq.shape[0] for seq in orig_sequences)
                vmin2 = float(min(seq[:min_len2].min() for seq in orig_sequences))
                vmax2 = float(max(seq[:min_len2].max() for seq in orig_sequences))
                stacked2: List[np.ndarray] = []
                for t in range(min_len2):
                    tiles2 = []
                    for seq in orig_sequences:
                        tiles2.append(normalize_to_bgr_global(seq[t], vmin2, vmax2))
                    tiled2 = np.concatenate(tiles2, axis=1)
                    cv2.putText(tiled2, f"{deciles[t]:.0f}%", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                    stacked2.append(tiled2)
                dec_path = out_dir / f"original_deciles_0-{args.phase_end}.mp4"
                actual2 = save_tiled_video(
                    stacked2, dec_path, fps=args.fps,
                    codec_pref='avc1', transcode_h264=DO_TRANSCODE_H264,
                    write_webm=DO_WRITE_WEBM
                )
                print(f"Saved original-decile video: {actual2}")

    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
