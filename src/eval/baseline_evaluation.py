#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline evaluation for dual-phase CT interpolation (0% -> 50%)
Baselines: linear, spline, optical flow (Farnebäck), DIR (BSpline→displacement→Warp)
- Multi-processing with a single shared pool across baselines
- LPIPS loaded once per worker via pool initializer
- Optional MS-SSIM + LPIPS
- Caching to skip already computed predictions
- CSV dumps for overall, per-slice, and phase-wise summaries
"""

import os
import json
import math
import time
import argparse
import logging
import warnings
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import cv2
import SimpleITK as sitk

from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

# Optional (installed by user)
try:
    from pytorch_msssim import ms_ssim as _torch_msssim
    import torch
    _HAS_MSSSIM = True
except Exception:
    _HAS_MSSSIM = False
    torch = None

try:
    import lpips as _lpips  # only imported inside worker init
    _HAS_LPIPS_MODULE = True
except Exception:
    _HAS_LPIPS_MODULE = False

# -------------------- Global config --------------------

PHASES_USE = [0, 10, 20, 30, 40, 50]
PHASES_EVAL = [10, 20, 30, 40]  # evaluate 10/20/30/40% between 0 and 50
T_LIST = [p / 50.0 for p in PHASES_EVAL]  # 0.2, 0.4, 0.6, 0.8

# LPIPS global (one per process)
_LPIPS_NET = None
_USE_LPIPS = False
_LPIPS_DEV = "cpu" 

def _pool_init(use_lpips: bool, sitk_threads: int, cv2_threads: int, blas_threads: int, lpips_device: str):
    """
    Runs once per *worker* process.
    - Sets thread caps (OpenCV / BLAS / Torch)
    - Builds LPIPS net if requested
    """
    # Quiet torchvision's pretrained deprecation noise
    warnings.filterwarnings(
        "ignore",
        message="The parameter 'pretrained' is deprecated since 0.13",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Arguments other than a weight enum or `None` for 'weights' are deprecated",
        category=UserWarning,
    )

    global _LPIPS_NET, _USE_LPIPS, _LPIPS_DEV
    _USE_LPIPS = bool(use_lpips)
    _LPIPS_DEV = lpips_device.lower().strip()

    # If LPIPS should be CPU, block CUDA before importing torch/torchvision
    if _LPIPS_DEV == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Thread controls
    try:
        cv2.setNumThreads(int(cv2_threads))
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(blas_threads))
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(sitk_threads))
    except Exception:
        pass

    # Import torch after CUDA visibility is set
    try:
        import torch as _torch_local
        # keep torch intra-op small per worker
        _torch_local.set_num_threads(max(1, int(blas_threads)))
    except Exception:
        pass

    # LPIPS model once per worker
    if _USE_LPIPS and _HAS_LPIPS_MODULE:
        try:
            import lpips as _lpips_local
            _net = _lpips_local.LPIPS(net='alex')
            if _LPIPS_DEV == "cuda":
                import torch as _torch_local
                if _torch_local.cuda.is_available():
                    _net = _net.cuda()
                else:
                    _LPIPS_DEV = "cpu"  # fallback if no CUDA
            _net.eval()
            _LPIPS_NET = _net
        except Exception as e:
            print(f"[worker-init] LPIPS failed to load: {e}")
            _LPIPS_NET = None

# -------------------- Utilities --------------------

def setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

def normalize_01(img: np.ndarray) -> np.ndarray:
    """
    Per-image baseline-like normalization to [0,1].
    If already [0,1], pass-through.
    Else use percentile [1,99].
    """
    img = img.astype(np.float32, copy=False)
    amin, amax = float(img.min()), float(img.max())
    if amin >= -1e-3 and amax <= 1.0 + 1e-3:
        return np.clip(img, 0.0, 1.0)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi > lo:
        x = (img - lo) / (hi - lo)
    else:
        x = np.zeros_like(img)
    return np.clip(x, 0.0, 1.0)

def psnr_ssim(pred01: np.ndarray, ref01: np.ndarray, use_gaussian_ssim: bool = True) -> Tuple[float, float]:
    p = float(sk_psnr(ref01, pred01, data_range=1.0))
    s = float(sk_ssim(ref01, pred01, data_range=1.0, gaussian_weights=use_gaussian_ssim))
    return p, s

def msssim(pred01: np.ndarray, ref01: np.ndarray) -> Optional[float]:
    if not _HAS_MSSSIM or torch is None:
        return None
    with torch.no_grad():
        # HxW [0,1] -> 1x1xHxW in [-1,1] range=2.0
        tx = torch.from_numpy(ref01).float().unsqueeze(0).unsqueeze(0)
        ty = torch.from_numpy(pred01).float().unsqueeze(0).unsqueeze(0)
        tx = tx * 2.0 - 1.0
        ty = ty * 2.0 - 1.0
        return float(_torch_msssim(ty, tx, data_range=2.0))

def lpips_score(pred01: np.ndarray, ref01: np.ndarray) -> Optional[float]:
    if not _USE_LPIPS or _LPIPS_NET is None:
        return None
    import torch
    with torch.no_grad():
        def to_tensor01(x):
            t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
            t = t * 2.0 - 1.0
            t = t.repeat(1, 3, 1, 1)  # 1x3xHxW
            if _LPIPS_DEV == "cuda" and torch.cuda.is_available():
                t = t.cuda(non_blocking=True)
            return t
        x = to_tensor01(ref01)
        y = to_tensor01(pred01)
        d = _LPIPS_NET(y, x)
        if isinstance(d, torch.Tensor):
            d = d.item()
        return float(d)

def resize_to(frames01: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """frames01: (H,W) or (T,H,W) -> resize to (H*,W*)."""
    Ht, Wt = size_hw
    if frames01.ndim == 2:
        return cv2.resize(frames01, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((frames01.shape[0], Ht, Wt), dtype=frames01.dtype)
    for i in range(frames01.shape[0]):
        out[i] = cv2.resize(frames01[i], (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    return out

# -------------------- Data discovery --------------------

def load_split_patients(azure_root: Path, split: str) -> List[str]:
    split_json = azure_root / "splits" / "patient_splits.json"
    if not split_json.exists():
        raise FileNotFoundError(f"Split file not found: {split_json}")
    with open(split_json, "r") as f:
        j = json.load(f)
    if split not in j:
        raise KeyError(f"Split '{split}' not in {split_json}")
    # filter out accidental junk that appeared in earlier message
    pts = [p for p in j[split] if "_HM" in p]
    return pts

def discover_samples(azure_root: Path, patients: List[str], slices: List[int]) -> List[Dict]:
    """
    Find (patient, series_id, slice_num, phase_dir).
    We use the 'hfr_linear_8fps' tree to get GT input_frames.npy for simplicity,
    since it contains identical input_frames.npy across experiment dirs.
    """
    data_dir = azure_root / "data"
    out = []
    for pid in patients:
        pdir = data_dir / pid
        if not pdir.exists():
            continue
        # pick any experiment dir that has series subdirs (linear or optical_flow)
        exp_dirs = [d for d in pdir.iterdir() if d.is_dir()]
        exp_dir = None
        for d in exp_dirs:
            # prefer linear for determinism
            if "linear" in d.name:
                exp_dir = d
                break
        if exp_dir is None and exp_dirs:
            exp_dir = exp_dirs[0]
        if exp_dir is None:
            continue

        for series_dir in sorted(exp_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            if not series_dir.name.startswith(pid):
                continue
            for s in slices:
                slice_dir = series_dir / f"slice_{s:04d}" / "phase_0-100"
                if (slice_dir / "input_frames.npy").exists() and (slice_dir / "metadata.json").exists():
                    out.append(dict(
                        patient_id=pid,
                        series_id=series_dir.name,
                        slice_num=s,
                        phase_dir=str(slice_dir)
                    ))
    return out

def load_gt_frames(phase_dir: str) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Load input_frames.npy and map phases -> indices.
    Returns frames [10,H,W] in [0,1], and a dict phase->idx
    """
    inp = Path(phase_dir) / "input_frames.npy"
    meta = Path(phase_dir) / "metadata.json"
    frames = np.load(str(inp)).astype(np.float32)  # [10,H,W]
    with open(meta, "r") as f:
        md = json.load(f)
    input_phases = md.get("experiment", {}).get("input_phases", list(range(0, 100, 10)))
    ph2idx = {ph: i for i, ph in enumerate(input_phases)}
    # normalize per-image to [0,1]
    frames01 = np.stack([normalize_01(frames[i]) for i in range(frames.shape[0])], axis=0)
    return frames01, ph2idx

# -------------------- Baselines --------------------

class BaseBaseline:
    name: str = "base"

    def synthesize(self, I0: np.ndarray, I50: np.ndarray, t_list: List[float]) -> Dict[float, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def _ensure01(x: np.ndarray) -> np.ndarray:
        return np.clip(x.astype(np.float32), 0.0, 1.0)

class LinearBaseline(BaseBaseline):
    name = "linear"
    def synthesize(self, I0, I50, t_list):
        out = {}
        for t in t_list:
            out[t] = self._ensure01((1.0 - t) * I0 + t * I50)
        return out

class SplineBaseline(BaseBaseline):
    name = "spline"
    # simple cubic smoothstep in time between endpoints
    @staticmethod
    def _smoothstep(t: float) -> float:
        return t * t * (3 - 2 * t)
    def synthesize(self, I0, I50, t_list):
        out = {}
        for t in t_list:
            s = self._smoothstep(float(t))
            out[t] = self._ensure01((1.0 - s) * I0 + s * I50)
        return out

class OpticalFlowBaseline(BaseBaseline):
    name = "optflow"
    def __init__(self, cache_root: Optional[Path] = None):
        self.cache_root = cache_root

    def synthesize(self, I0, I50, t_list):
        # compute forward flow I0->I50 using Farneback on uint8
        H, W = I0.shape
        key = None
        flow = None

        if self.cache_root is not None:
            key = self.cache_root / "optflow_flow.npy"
            if key.exists():
                flow = np.load(str(key))

        if flow is None:
            A = (I0 * 255.0).astype(np.uint8)
            B = (I50 * 255.0).astype(np.uint8)
            flow = cv2.calcOpticalFlowFarneback(
                A, B, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )  # (H,W,2) float32 (dx, dy)
            if key is not None:
                np.save(str(key), flow)

        # Warp I0 with t * flow using remap
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        out = {}
        for t in t_list:
            dx = t * flow[..., 0]
            dy = t * flow[..., 1]
            map_x = (grid_x + dx).astype(np.float32)
            map_y = (grid_y + dy).astype(np.float32)
            warped = cv2.remap(I0.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out[t] = self._ensure01(warped)
        return out

class DIRBaseline(BaseBaseline):
    name = "dir"
    def __init__(self, mesh_pixel: int = 32, iters: int = 200, metric: str = "mse", cache_root: Optional[Path] = None):
        self.mesh_px = int(mesh_pixel)
        self.iters = int(iters)
        self.metric = metric
        self.cache_root = cache_root

    def _register(self, I0: np.ndarray, I50: np.ndarray) -> sitk.Image:
        """
        Register moving=I0 to fixed=I50 with BSpline, return displacement field (VectorFloat64) on I0 grid.
        """
        H, W = I0.shape
        I0_s  = sitk.GetImageFromArray(I0.astype(np.float32))   # moving
        I50_s = sitk.GetImageFromArray(I50.astype(np.float32))  # fixed

        # consistent geometry
        for im in (I0_s, I50_s):
            im.SetOrigin((0.0, 0.0))
            im.SetSpacing((1.0, 1.0))
            im.SetDirection((1.0, 0.0, 0.0, 1.0))

        # BSpline transform domain (use fixed image domain is common)
        mesh_size = [max(1, W // self.mesh_px), max(1, H // self.mesh_px)]
        tx = sitk.BSplineTransformInitializer(image1=I50_s, transformDomainMeshSize=mesh_size, order=3)

        reg = sitk.ImageRegistrationMethod()
        if self.metric == "mse":
            reg.SetMetricAsMeanSquares()
        else:
            reg.SetMetricAsMattesMutualInformation(32)

        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=self.iters,
                                          convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([2, 1])
        reg.SetSmoothingSigmasPerLevel([1.0, 0.0])

        # IMPORTANT: set initial transform, then Execute with TWO args: (fixed, moving)
        reg.SetInitialTransform(tx, inPlace=False)
        final_tx = reg.Execute(I50_s, I0_s)  # fixed=I50, moving=I0

        # Create displacement field on the I0 grid (moving image) — robust across SITK versions
        try:
            df = sitk.TransformToDisplacementFieldFilter()
            df.SetReferenceImage(I0_s)
            df.SetOutputPixelType(sitk.sitkVectorFloat64)
            disp = df.Execute(final_tx)
        except Exception:
            # Fallbacks in case the filter API isn't available in your build
            try:
                disp = sitk.TransformToDisplacementField(final_tx, sitk.sitkVectorFloat64, I0_s)
            except Exception:
                disp = sitk.TransformToDisplacementField(
                    final_tx,
                    I0_s.GetSize(),
                    I0_s.GetOrigin(),
                    I0_s.GetSpacing(),
                    I0_s.GetDirection(),
                    sitk.sitkVectorFloat64,
                )

        return disp

    def synthesize(self, I0, I50, t_list):
        disp = None
        disp_path = None
        if self.cache_root is not None:
            self.cache_root.mkdir(parents=True, exist_ok=True)
            disp_path = self.cache_root / "dir_disp.nii.gz"
            if disp_path.exists():
                disp = sitk.ReadImage(str(disp_path), sitk.sitkVectorFloat64)

        if disp is None:
            disp = self._register(I0, I50)
            if disp_path is not None:
                sitk.WriteImage(disp, str(disp_path))

        I0_s = sitk.GetImageFromArray(I0.astype(np.float32))
        I0_s.CopyInformation(disp)

        outs = {}
        disp_np = sitk.GetArrayFromImage(disp)  # (H, W, 2), float64

        for t in t_list:
            # Resample expects a transform that maps OUTPUT -> INPUT.
            # Our displacement field is forward (I0 -> I50), so we use the NEGATIVE field for backward sampling.
            scaled_np = -(float(t) * disp_np)  # note the minus sign
            disp_t = sitk.GetImageFromArray(scaled_np.astype(np.float64), isVector=True)
            disp_t.CopyInformation(disp)

            tx = sitk.DisplacementFieldTransform(disp_t)

            # Resample on I0's grid using backward mapping
            warped = sitk.Resample(
                I0_s,           # input image to sample from
                I0_s,           # reference image (defines output grid = I0 grid)
                tx,             # transform (OUTPUT -> INPUT)
                sitk.sitkLinear,
                0.0,            # default pixel value
                sitk.sitkFloat32,
            )

            outs[t] = np.clip(sitk.GetArrayFromImage(warped).astype(np.float32), 0.0, 1.0)

        return outs

# -------------------- Worker task --------------------

def _eval_one(sample: Dict,
              baseline_name: str,
              use_msssim: bool,
              cache_dir: Optional[str],
              dir_mesh_px: int,
              dir_iters: int,
              dir_metric: str) -> Dict:
    """
    Compute predictions and metrics for a single sample for a single baseline.
    Returns a dict containing per-phase metrics rows + (optional) cache path info.
    """
    # unpack paths
    phase_dir = sample["phase_dir"]
    slice_num = int(sample["slice_num"])

    # cache directory for this sample/baseline
    cache_root = Path(cache_dir) if cache_dir else None
    smp_cache = (cache_root / baseline_name /
                 sample["patient_id"] / sample["series_id"] /
                 f"slice_{slice_num:04d}") if cache_root else None
    if smp_cache is not None:
        smp_cache.mkdir(parents=True, exist_ok=True)

    # load GT
    frames01, ph2idx = load_gt_frames(phase_dir)
    missing = [ph for ph in PHASES_USE if ph not in ph2idx]
    if missing:
        return {"rows": []}  # skip

    I0 = frames01[ph2idx[0]]
    I50 = frames01[ph2idx[50]]
    refs = {t: frames01[ph2idx[int(50*t)]] for t in T_LIST}  # refs at 10/20/30/40

    # build baseline
    if baseline_name == "linear":
        baseline = LinearBaseline()
    elif baseline_name == "spline":
        baseline = SplineBaseline()
    elif baseline_name == "optflow":
        baseline = OpticalFlowBaseline(cache_root=smp_cache)
    elif baseline_name == "dir":
        baseline = DIRBaseline(
            mesh_pixel=dir_mesh_px,
            iters=dir_iters,
            metric=dir_metric,
            cache_root=smp_cache)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    # check cache for preds
    preds_cache_path = smp_cache / "preds.npz" if smp_cache is not None else None
    preds = None
    if preds_cache_path is not None and preds_cache_path.exists():
        try:
            z = np.load(str(preds_cache_path))
            preds = {float(k): z[k] for k in z.files}
        except Exception:
            preds = None

    if preds is None:
        preds = baseline.synthesize(I0, I50, T_LIST)
        # save
        if preds_cache_path is not None:
            np.savez_compressed(str(preds_cache_path), **{str(t): preds[t] for t in preds})

    # metrics
    rows = []
    for t in T_LIST:
        pred01 = preds[t]
        ref01 = refs[t]

        ps, ss = psnr_ssim(pred01, ref01, use_gaussian_ssim=True)
        ms = msssim(pred01, ref01) if use_msssim else None
        lp = lpips_score(pred01, ref01)

        # “ssim_like” column will hold MS-SSIM if requested and available; else SSIM
        ssim_like = ms if (use_msssim and ms is not None) else ss

        rows.append(dict(
            baseline=baseline_name,
            slice_num=slice_num,
            phase=int(50*t),
            psnr=ps,
            ssim=ss,
            msssim=ms if ms is not None else np.nan,
            ssim_like=ssim_like,
            lpips=lp if lp is not None else np.nan
        ))

    return {"rows": rows}

# -------------------- Main --------------------

def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--azure-root", type=str, required=True, help="~/azureblob/4D-Lung-Interpolated")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--slices", type=str, default="12,25,37")
    ap.add_argument("--baselines", type=str, default="linear,spline,optflow,dir")
    ap.add_argument("--dir-mesh-px", type=int, default=32,
                    help="Approx BSpline grid cell size (pixels) for DIR baseline")
    ap.add_argument("--dir-iters", type=int, default=200,
                    help="Registration iterations for DIR baseline")
    ap.add_argument("--dir-metric", type=str, default="mse", choices=["mse","mi"],
                    help="DIR similarity metric")

    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sitk-threads", type=int, default=2)
    ap.add_argument("--cv2-threads", type=int, default=1)
    ap.add_argument("--blas-threads", type=int, default=1)

    ap.add_argument("--cache-root", type=str, default=None, help="Cache dir for flows/preds to skip recompute")
    ap.add_argument("--use-msssim", action="store_true")
    ap.add_argument("--use-lpips", action="store_true")
    ap.add_argument("--save-csv", action="store_true")
    ap.add_argument("--csv-dir", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--lpips-device", type=str, default="cpu", choices=["cpu","cuda"],
                help="Device for LPIPS (default cpu). CPU avoids heavy CUDA init per process.")


    args = ap.parse_args()
    setup_logging(args.verbose)

    azure_root = Path(args.azure_root)
    cache_root = Path(args.cache_root) if args.cache_root else None
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]

    slices = [int(s) for s in args.slices.split(",")]
    patients = load_split_patients(azure_root, args.split)
    samples = discover_samples(azure_root, patients, slices)

    print(f"Discovered {len(samples)} slices for split='{args.split}'.")
    print(f"Patients in set: {', '.join(sorted(set(patients)))}\n")
    print(f"Evaluating baselines: {','.join(baselines)}")
    metric_str = "PSNR, " + ("MS-SSIM" if args.use_msssim and _HAS_MSSSIM else "SSIM") + (", LPIPS" if args.use_lpips and _HAS_LPIPS_MODULE else "")
    print(f"(Metrics: {metric_str})")
    print(f"Cache root: {cache_root}")
    print(f"Workers: {args.workers} | SITK threads/proc: {args.sitk_threads} | OpenCV threads/proc: {args.cv2_threads} | BLAS threads/proc: {args.blas_threads}\n")

    if args.workers <= 1 and args.use_lpips and _HAS_LPIPS_MODULE:
        # Initialize LPIPS in main process (single-thread fallback)
        _pool_init(True, args.sitk_threads, args.cv2_threads, args.blas_threads, args.lpips_device)

    t0 = time.time()
    all_rows = []

    # ---- unified progress counters ----
    total = len(baselines) * len(samples)
    done = 0
    tick_every = max(1, total // 20)  # ~5% updates

    if args.workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_pool_init,
            initargs=(
                args.use_lpips and _HAS_LPIPS_MODULE,
                args.sitk_threads, args.cv2_threads, args.blas_threads,
                args.lpips_device if hasattr(args, "lpips_device") else None,
            ),
        ) as ex:
            futures = []
            for smp in samples:
                for bname in baselines:
                    futures.append(ex.submit(
                        _eval_one, smp, bname, args.use_msssim,
                        str(cache_root) if cache_root else None,
                        args.dir_mesh_px, args.dir_iters, args.dir_metric
                    ))

            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    logging.error(f"Worker task failed: {e}")
                    res = {"rows": []}
                all_rows.extend(res.get("rows", []))

                done += 1
                if args.verbose and (done % tick_every == 0 or done == total):
                    print(f"[progress] {done}/{total} ({100.0*done/total:.1f}%)")

    else:
        for bname in baselines:
            for smp in samples:
                try:
                    res = _eval_one(
                        smp, bname, args.use_msssim,
                        str(cache_root) if cache_root else None,
                        args.dir_mesh_px, args.dir_iters, args.dir_metric
                    )
                except Exception as e:
                    logging.error(f"Task failed: {e}")
                    res = {"rows": []}
                all_rows.extend(res.get("rows", []))

                done += 1
                if args.verbose and (done % tick_every == 0 or done == total):
                    print(f"[progress] {done}/{total} ({100.0*done/total:.1f}%)")

    dt = time.time() - t0


    if not all_rows:
        print("No results produced.")
        return

    df = pd.DataFrame(all_rows)

    # ---------------- Summaries ----------------
    # Overall by baseline
    metric_name = "MS-SSIM" if args.use_msssim and _HAS_MSSSIM else "SSIM"
    print("\n=== Summary (mean ± std over all samples × phases) ===")
    print(f"{'Baseline':<12}  {'PSNR':>18}  {metric_name:>18}  {'LPIPS':>12}")
    print("-" * 72)
    for bname, g in df.groupby("baseline"):
        ps_m, ps_s = g["psnr"].mean(), g["psnr"].std()
        ssim_like_col = "ssim_like"
        ss_m, ss_s = g[ssim_like_col].mean(), g[ssim_like_col].std()
        if "lpips" in g and not g["lpips"].isna().all():
            lp_m, lp_s = g["lpips"].mean(), g["lpips"].std()
            lp_str = f"{lp_m:.4f} ± {lp_s:.4f}"
        else:
            lp_str = "N/A"
        print(f"{bname:<12}  {ps_m:>9.4f} ± {ps_s:<7.4f}  {ss_m:>9.4f} ± {ss_s:<7.4f}  {lp_str:>12}")

    # Per-slice (mean SSIM/MS-SSIM / PSNR)
    print(f"\n=== Per-slice performance (mean {metric_name} / PSNR) ===")
    for bname, g in df.groupby("baseline"):
        ag = g.groupby("slice_num").agg(
            psnr_mean=("psnr", "mean"),
            ssim_mean=("ssim_like", "mean")
        )
        print(f"[{bname}]")
        for s in sorted(ag.index):
            print(f"  slice {s:02d}:  {ag.loc[s,'ssim_mean']:.3f} / {ag.loc[s,'psnr_mean']:.2f}")

    # Phase-specific PSNR
    print("\n=== Phase-specific PSNR (dB) ===")
    header = "Method           " + "    ".join([f"{ph:>4}%" for ph in PHASES_EVAL])
    print(header)
    print("-" * len(header))
    for bname, g in df.groupby("baseline"):
        row = []
        for ph in PHASES_EVAL:
            row.append(f"{g[g['phase']==ph]['psnr'].mean():.2f}")
        print(f"{bname:<16}" + "    ".join([f"{x:>6}" for x in row]))

    # ---------------- CSV dumps ----------------
    if args.save_csv:
        csv_dir = Path(args.csv_dir) if args.csv_dir else (cache_root if cache_root else Path.cwd())
        csv_dir.mkdir(parents=True, exist_ok=True)

        # overall by baseline
        rows = []
        for bname, g in df.groupby("baseline"):
            rows.append(dict(
                baseline=bname,
                psnr_mean=g["psnr"].mean(), psnr_std=g["psnr"].std(),
                ssim_like_mean=g["ssim_like"].mean(), ssim_like_std=g["ssim_like"].std(),
                lpips_mean=(g["lpips"].mean() if "lpips" in g and not g["lpips"].isna().all() else np.nan),
                lpips_std=(g["lpips"].std() if "lpips" in g and not g["lpips"].isna().all() else np.nan),
            ))
        pd.DataFrame(rows).to_csv(csv_dir / "overall.csv", index=False)

        # per-slice
        per_slice = df.groupby(["baseline","slice_num"]).agg(
            psnr_mean=("psnr","mean"),
            ssim_like_mean=("ssim_like","mean")
        ).reset_index()
        per_slice.to_csv(csv_dir / "per_slice.csv", index=False)

        # phase-wise PSNR
        per_phase = df.groupby(["baseline","phase"]).agg(
            psnr_mean=("psnr","mean")
        ).reset_index()
        per_phase.to_csv(csv_dir / "per_phase_psnr.csv", index=False)

        # raw detail
        df.to_csv(csv_dir / "per_sample_phase_detail.csv", index=False)

        print(f"\nCSV saved to: {csv_dir}")

    print(f"\nDone in {dt:.1f}s.")

if __name__ == "__main__":
    main()
