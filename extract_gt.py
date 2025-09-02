#!/usr/bin/env python3
import argparse, shutil, sys
from pathlib import Path

GT_FILES = ("input_frames.npy", "metadata.json")

def pick_experiment_dir(patient_dir: Path, prefer_substring: str) -> Path | None:
    exp_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
    if not exp_dirs:
        return None
    preferred = [d for d in exp_dirs if prefer_substring and prefer_substring in d.name.lower()]
    return preferred[0] if preferred else exp_dirs[0]

def iter_phase_dirs(exp_dir: Path, series_prefix: str) -> list[Path]:
    # series dirs inside the chosen experiment dir
    out = []
    for series_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        if not series_dir.name.startswith(series_prefix):
            continue
        # look for slice_XXXX/phase_0-100
        for slice_dir in series_dir.glob("slice_*"):
            phase_dir = slice_dir / "phase_0-100"
            if phase_dir.is_dir():
                out.append(phase_dir)
    return out

def ensure_gt_present(phase_dir: Path) -> bool:
    return all((phase_dir / f).exists() for f in GT_FILES)

def copy_or_move(src: Path, dst: Path, move: bool, overwrite: bool, dry_run: bool, verbose: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        if verbose:
            print(f"[skip] exists: {dst}")
        return "skipped"
    if dry_run:
        print(f"[dry-run] {'move' if move else 'copy'} {src} -> {dst}")
        return "dry-run"
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))
    if verbose:
        print(f"[ok] {'moved' if move else 'copied'} {src} -> {dst}")
    return "done"

def prune_empty_dirs(path: Path, stop_at: Path, dry_run: bool, verbose: bool):
    # remove empty dirs up to stop_at (exclusive)
    p = path
    while p != stop_at and p.is_dir():
        try:
            next(p.iterdir())
            break  # not empty
        except StopIteration:
            if dry_run:
                print(f"[dry-run] rmdir {p}")
            else:
                p.rmdir()
                if verbose:
                    print(f"[pruned] {p}")
        p = p.parent

def main():
    ap = argparse.ArgumentParser(description="Extract/move ONLY GT files (input_frames.npy, metadata.json).")
    ap.add_argument("--source-root", default="~/azureblob/4D-Lung-Interpolulated",
                    help="Root of dataset (default: ~/azureblob/4D-Lung-Interpolulated)")
    ap.add_argument("--dest-root", default="~/flowmotion/data",
                    help="Destination root (default: ~/flowmotion/data)")
    ap.add_argument("--prefer-exp", default="linear",
                    help="Substring to prefer for experiment dir selection (default: linear)")
    ap.add_argument("--move", action="store_true", help="Move files instead of copy")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite destination if exists")
    ap.add_argument("--prune-empty", action="store_true", help="After move, prune empty source dirs")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without changing files")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    args = ap.parse_args()

    src_root = Path(args.source_root).expanduser()
    # note: the real folder is '4D-Lung-Interpolated' (typo guard)
    if not src_root.exists():
        alt = Path(str(src_root).replace("Interpolulated", "Interpolated"))
        if alt.exists():
            src_root = alt

    data_root = src_root / "data"
    dst_root = Path(args.dest_root).expanduser()

    if not data_root.is_dir():
        print(f"ERROR: {data_root} not found", file=sys.stderr)
        sys.exit(1)

    total_phase_dirs = 0
    total_done = 0
    total_skip = 0
    total_missing = 0

    patients = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if args.verbose:
        print(f"Found {len(patients)} patient dirs under {data_root}")

    for pdir in patients:
        exp_dir = pick_experiment_dir(pdir, args.prefer_exp.lower() if args.prefer_exp else "")
        if exp_dir is None:
            if args.verbose:
                print(f"[warn] No experiment dir inside {pdir}")
            continue

        # series_prefix is typically the patient ID (e.g., '110_HM10395')
        series_prefix = pdir.name
        phase_dirs = iter_phase_dirs(exp_dir, series_prefix)
        if args.verbose:
            print(f"[{pdir.name}] using exp '{exp_dir.name}', phase dirs: {len(phase_dirs)}")
        for phase_dir in phase_dirs:
            total_phase_dirs += 1
            if not ensure_gt_present(phase_dir):
                total_missing += 1
                if args.verbose:
                    print(f"[warn] Missing GT in {phase_dir}")
                continue

            # Build destination mirror: <dest>/<patient>/<series>/slice_xxxx/phase_0-100
            series_dir = phase_dir.parent.parent  # .../<series>/slice_xxxx/phase_0-100
            dst_phase_dir = dst_root / pdir.name / series_dir.name / phase_dir.parent.name / phase_dir.name

            # copy/move ONLY the two GT files
            results = []
            for fname in GT_FILES:
                src = phase_dir / fname
                dst = dst_phase_dir / fname
                results.append(copy_or_move(src, dst, args.move, args.overwrite, args.dry_run, args.verbose))

            if args.move and args.prune_empty:
                # if both files were moved (not skipped), try pruning the now-empty phase dir upward
                if all(r in ("done", "dry-run") for r in results):
                    prune_empty_dirs(phase_dir, stop_at=pdir, dry_run=args.dry_run, verbose=args.verbose)

            if any(r == "done" for r in results):
                total_done += 1
            else:
                total_skip += 1

    print("\n=== Summary ===")
    print(f"Phase dirs visited : {total_phase_dirs}")
    print(f"GT missing         : {total_missing}")
    print(f"Dirs processed     : {total_done}")
    print(f"Dirs skipped       : {total_skip}")
    print(f"Mode               : {'MOVE' if args.move else 'COPY'}{' (dry-run)' if args.dry_run else ''}")
    print(f"Source root        : {data_root}")
    print(f"Destination root   : {dst_root}")

if __name__ == "__main__":
    main()
