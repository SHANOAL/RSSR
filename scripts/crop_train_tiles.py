#!/usr/bin/env python3
"""Crop paired large GeoTIFFs into training tiles.

Output folders, split by source image so that tiles of one scene stay in one subset:
    <out>/train/HR|LR/   400x400 / 40x40 tiles
    <out>/val/HR|LR/
    <out>/test/HR|LR/
    <out>/test_scenes/HR|LR/  complete test scenes (symbolic links) for complete-scene inference
    <out>/split.json

Pairing walks the HR root recursively and matches every HR file to the LR file whose name contains it.
Each HR window resamples the LR image over the same ground extent to a 40x40 grid (x10).
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import from_bounds
from rasterio.windows import Window, bounds as window_bounds
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.warp import reproject

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from scene_split import (
    assign_scene_splits,
    fill_empty_splits,
    link_test_scenes,
    make_split_dirs,
    print_split_assignment,
    write_split_json,
)

TIF_EXTS = {".tif", ".tiff"}
HR_SIZE = 400
LR_SIZE = 40
DN_SCALE = 10000.0


def list_tif_files(root: Path) -> List[Path]:
    """Collect every .tif/.tiff under root recursively, following symbolic links."""
    if not root.is_dir():
        raise FileNotFoundError(f"directory not found: {root}")
    seen_dirs = set()
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        resolved = Path(dirpath).resolve()
        if resolved in seen_dirs:
            dirnames[:] = []
            continue
        seen_dirs.add(resolved)
        for name in filenames:
            if Path(name).suffix.lower() in TIF_EXTS:
                files.append(Path(dirpath) / name)
    return sorted(files)


def rel_under(path: Path, root: Path) -> str:
    """Path of path relative to root, or its name if that fails."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def hr_rel_stem(hr_path: Path, hr_dir: Path) -> str:
    """HR stem relative to the root, keeping subfolders such as area1/re_det_xxx."""
    rel = Path(rel_under(hr_path, hr_dir))
    return str(rel.with_suffix(""))


def lr_name_contains_hr_name(hr_path: Path, lr_path: Path) -> bool:
    """Whether the LR file name contains the HR file name.

    For example ``S2A_all_re_det_xxx.tif`` contains ``re_det_xxx.tif``.
    The match must start at a boundary (the beginning or ``_``) so that ``a.tif`` does not match ``ba.tif``.
    """
    hr_stem = hr_path.stem.lower()
    lr_stem = lr_path.stem.lower()
    if lr_stem == hr_stem or lr_stem.endswith("_" + hr_stem):
        return True
    lr_name = lr_path.name.lower()
    needles = (hr_path.name.lower(), hr_stem + ".tif", hr_stem + ".tiff")
    for needle in needles:
        idx = lr_name.find(needle)
        if idx < 0:
            continue
        if idx == 0 or lr_name[idx - 1] in ("_", "-", ".", " "):
            return True
    return False


def pick_best_lr(hr_path: Path, candidates: Sequence[Path]) -> Path:
    """Prefer an identical name, then the LR name with the fewest extra characters."""
    hr_name = hr_path.name.lower()
    hr_stem = hr_path.stem.lower()

    def score(lp: Path) -> Tuple[int, int, str]:
        lr_name = lp.name.lower()
        lr_stem = lp.stem.lower()
        exact = 0 if lr_name == hr_name or lr_stem == hr_stem else 1
        extra = max(0, len(lr_stem) - len(hr_stem))
        return (exact, extra, lr_name)

    return min(candidates, key=score)


def match_hr_lr_pairs(hr_dir: Path, lr_dir: Path) -> dict:
    """Pair every HR image with the LR image whose file name contains its name."""
    hr_files = list_tif_files(hr_dir)
    lr_files = list_tif_files(lr_dir)
    remaining = list(lr_files)
    pairs: List[Tuple[Path, Path]] = []
    unmatched_hr: List[Path] = []
    ambiguous: List[Tuple[Path, Path, List[Path]]] = []
    hr_sorted = sorted(hr_files, key=lambda p: (-len(p.stem), -len(p.name), str(p)))
    for hr_path in hr_sorted:
        cands = [p for p in remaining if lr_name_contains_hr_name(hr_path, p)]
        if not cands:
            unmatched_hr.append(hr_path)
            continue
        best = pick_best_lr(hr_path, cands)
        if len(cands) > 1:
            ambiguous.append((hr_path, best, [p for p in cands if p != best]))
        pairs.append((hr_path, best))
        remaining.remove(best)
    order = {p: i for i, p in enumerate(hr_files)}
    pairs.sort(key=lambda item: order.get(item[0], 10**9))
    return {
        "hr_files": hr_files,
        "lr_files": lr_files,
        "pairs": pairs,
        "unmatched_hr": unmatched_hr,
        "unused_lr": remaining,
        "ambiguous": ambiguous,
    }


def print_pair_stats(info: dict, hr_dir: Path, lr_dir: Path, max_lines: int = 8) -> None:
    """Print the HR/LR pairs with their relative paths."""
    print(f"[crop] HR dir: {hr_dir}")
    print(f"[crop] LR dir: {lr_dir}")
    print(f"[crop] HR={len(info['hr_files'])}  LR={len(info['lr_files'])}  "
          f"pairs={len(info['pairs'])}  unmatched_HR={len(info['unmatched_hr'])}  "
          f"unused_LR={len(info['unused_lr'])}")
    for hr_path, lr_path in info["pairs"][:max_lines]:
        print(f"[crop]   {rel_under(hr_path, hr_dir)}  <->  {rel_under(lr_path, lr_dir)}")
    if len(info["pairs"]) > max_lines:
        print(f"[crop]   ... {len(info['pairs'])} pairs in total")
    for p in info["unmatched_hr"][:max_lines]:
        print(f"[crop] unmatched HR: {rel_under(p, hr_dir)}")
    for p in info["unused_lr"][:max_lines]:
        print(f"[crop] unused LR: {rel_under(p, lr_dir)}")
    for hr_path, best, skipped in info.get("ambiguous", [])[:max_lines]:
        extra = ", ".join(p.name for p in skipped[:3])
        print(f"[crop] ambiguous HR {rel_under(hr_path, hr_dir)} -> {best.name} (also {extra})")


def iter_tile_origins(height: int, width: int, tile: int, stride: int) -> Iterable[Tuple[int, int]]:
    """Top-left (row, col) of every tile that lies fully inside the image."""
    if height < tile or width < tile:
        return
    for row in range(0, height - tile + 1, stride):
        for col in range(0, width - tile + 1, stride):
            yield row, col


def _named_band_index(src: rasterio.DatasetReader, names: Sequence[str]) -> Optional[List[int]]:
    desc = [(d or "").upper() for d in (src.descriptions or ())]
    if not desc:
        return None
    indexes = []
    for name in names:
        if name not in desc:
            return None
        indexes.append(desc.index(name) + 1)
    return indexes


def get_hr_rgb_indexes(src: rasterio.DatasetReader) -> List[int]:
    """HR bands: RGB by color interpretation, otherwise the first three bands."""
    names = [c.name.lower() for c in src.colorinterp]
    if {"red", "green", "blue"}.issubset(set(names)):
        return [names.index("red") + 1, names.index("green") + 1, names.index("blue") + 1]
    if src.count < 3:
        raise ValueError(f"HR needs >=3 bands, got {src.count}: {src.name}")
    return [1, 2, 3]


def get_lr_rgb_indexes(src: rasterio.DatasetReader) -> List[int]:
    """LR bands: Sentinel-2 B4/B3/B2, RGB by color interpretation, or the first three bands."""
    s2_rgb = _named_band_index(src, ("B4", "B3", "B2"))
    if s2_rgb is not None:
        return s2_rgb
    names = [c.name.lower() for c in src.colorinterp]
    if {"red", "green", "blue"}.issubset(set(names)):
        return [names.index("red") + 1, names.index("green") + 1, names.index("blue") + 1]
    if src.count < 3:
        raise ValueError(f"LR needs >=3 bands, got {src.count}: {src.name}")
    return [1, 2, 3]


def normalize_reflectance(arr: np.ndarray, dn_scale: float = DN_SCALE) -> np.ndarray:
    """Normalize reflectance or Sentinel-2 DN to [0, 1]."""
    x = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.nanmax(x)) > 1.5:
        x = x / float(dn_scale)
    return np.clip(x, 0.0, 1.0)


def is_hr_tile_valid(hr_chw: np.ndarray, min_mean: float, min_valid_ratio: float) -> bool:
    """Reject dark or largely empty HR tiles."""
    x = hr_chw.astype(np.float32)
    if float(x.mean()) < min_mean:
        return False
    valid = (hr_chw.max(axis=0) > 0).mean()
    return float(valid) >= min_valid_ratio


def is_lr_tile_valid(lr_chw: np.ndarray, min_valid_ratio: float) -> bool:
    """Reject almost empty LR tiles."""
    valid = np.isfinite(lr_chw) & (lr_chw > 1e-6)
    return float(valid.any(axis=0).mean()) >= min_valid_ratio


def pad_window(window: Window, src_w: int, src_h: int, pad: int = 2) -> Window:
    """Pad the resampling window by one pixel and clip it to the raster."""
    padded = Window(
        window.col_off - pad,
        window.row_off - pad,
        window.width + 2 * pad,
        window.height + 2 * pad,
    )
    return padded.intersection(Window(0, 0, src_w, src_h))


def warp_lr_patch(
    lr_src: rasterio.DatasetReader,
    hr_bounds: Tuple[float, float, float, float],
    hr_crs,
    band_indexes: Sequence[int],
    lr_size: int,
) -> Optional[Tuple[np.ndarray, object]]:
    """Resample the LR image to an lr_size x lr_size grid over the extent of the HR tile."""
    src_window = window_from_bounds(*hr_bounds, transform=lr_src.transform)
    src_window = src_window.round_offsets(op="floor").round_lengths(op="ceil")
    src_window = pad_window(src_window, lr_src.width, lr_src.height)
    if src_window.width <= 0 or src_window.height <= 0:
        return None
    data = lr_src.read(list(band_indexes), window=src_window).astype(np.float32)
    src_transform = rasterio.windows.transform(src_window, lr_src.transform)
    dst = np.zeros((len(band_indexes), lr_size, lr_size), dtype=np.float32)
    dst_transform = from_bounds(*hr_bounds, lr_size, lr_size)
    for i in range(len(band_indexes)):
        reproject(
            source=data[i],
            destination=dst[i],
            src_transform=src_transform,
            src_crs=lr_src.crs,
            dst_transform=dst_transform,
            dst_crs=hr_crs,
            resampling=Resampling.bilinear,
        )
    return dst, dst_transform


def write_rgb_tif(
    path: Path,
    arr_chw: np.ndarray,
    transform,
    crs,
    dtype: str,
) -> None:
    """Write a three-band RGB GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(arr_chw.shape[1]),
        "width": int(arr_chw.shape[2]),
        "count": 3,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "photometric": "RGB",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr_chw)
        dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)


def tile_output_paths(out_dir: Path, rel_stem: str, row: int, col: int) -> Tuple[Path, Path]:
    """Write a tile pair, mirroring the HR subfolders under HR/ and LR/."""
    rel = Path(rel_stem)
    name = f"{rel.name}_r{row:05d}_c{col:05d}.tif"
    sub = rel.parent
    if str(sub) in (".", ""):
        return out_dir / "HR" / name, out_dir / "LR" / name
    return out_dir / "HR" / sub / name, out_dir / "LR" / sub / name


def crop_one_pair(job: dict) -> dict:
    """Crop one image pair and return the counts of written and skipped tiles."""
    hr_path = Path(job["hr_path"])
    lr_path = Path(job["lr_path"])
    stats = {
        "stem": job.get("hr_rel_stem", hr_path.stem),
        "split": job.get("split", ""),
        "written": 0,
        "skipped_exist": 0,
        "skipped_invalid": 0,
    }
    with rasterio.open(hr_path) as hr_src, rasterio.open(lr_path) as lr_src:
        hr_idx = get_hr_rgb_indexes(hr_src)
        lr_idx = get_lr_rgb_indexes(lr_src)
        origins = list(iter_tile_origins(hr_src.height, hr_src.width, job["hr_size"], job["stride"]))
        for row, col in origins:
            _crop_one_tile(hr_src, lr_src, hr_idx, lr_idx, row, col, job, stats)
    return stats


def _crop_one_tile(
    hr_src,
    lr_src,
    hr_idx: Sequence[int],
    lr_idx: Sequence[int],
    row: int,
    col: int,
    job: dict,
    stats: dict,
) -> None:
    hr_out, lr_out = tile_output_paths(
        Path(job["out_dir"]), job.get("hr_rel_stem", Path(job["hr_path"]).stem), row, col
    )
    if job["skip_existing"] and hr_out.is_file() and lr_out.is_file():
        stats["skipped_exist"] += 1
        return
    window = Window(col, row, job["hr_size"], job["hr_size"])
    hr_arr = hr_src.read(list(hr_idx), window=window)
    if hr_arr.dtype != np.uint8:
        hr_arr = np.clip(hr_arr, 0, 255).astype(np.uint8)
    if not is_hr_tile_valid(hr_arr, job["min_hr_mean"], job["min_valid_ratio"]):
        stats["skipped_invalid"] += 1
        return
    hr_bounds = window_bounds(window, hr_src.transform)
    warped = warp_lr_patch(lr_src, hr_bounds, hr_src.crs, lr_idx, job["lr_size"])
    if warped is None:
        stats["skipped_invalid"] += 1
        return
    lr_arr, lr_transform = warped
    lr_arr = normalize_reflectance(lr_arr, job["dn_scale"])
    if not is_lr_tile_valid(lr_arr, job["min_valid_ratio"]):
        stats["skipped_invalid"] += 1
        return
    hr_transform = rasterio.windows.transform(window, hr_src.transform)
    write_rgb_tif(hr_out, hr_arr, hr_transform, hr_src.crs, "uint8")
    write_rgb_tif(lr_out, lr_arr, lr_transform, hr_src.crs, "float32")
    stats["written"] += 1


def build_jobs(split_map: dict, args: argparse.Namespace) -> List[dict]:
    jobs: List[dict] = []
    for split_name in ("train", "val", "test"):
        for hr_path, lr_path in split_map.get(split_name, []):
            jobs.append({
                "hr_path": str(hr_path),
                "lr_path": str(lr_path),
                "hr_rel_stem": hr_rel_stem(hr_path, args.hr_dir),
                "out_dir": str(args.out / split_name),
                "split": split_name,
                "hr_size": args.hr_size,
                "lr_size": args.lr_size,
                "stride": args.stride,
                "min_hr_mean": args.min_hr_mean,
                "min_valid_ratio": args.min_valid_ratio,
                "dn_scale": args.dn_scale,
                "skip_existing": args.skip_existing,
            })
    return jobs


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Crop large HR/LR GeoTIFFs into VARSR train tiles.")
    parser.add_argument("--hr-dir", type=Path, default=Path("~/MachineLearing/GHplus/datas/HR").expanduser())
    parser.add_argument("--lr-dir", type=Path, default=Path("~/MachineLearing/GHplus/datas/LR").expanduser())
    parser.add_argument("--out", type=Path, default=repo_root / "datasets" / "train_ghplus")
    parser.add_argument("--hr-size", type=int, default=HR_SIZE)
    parser.add_argument("--lr-size", type=int, default=LR_SIZE)
    parser.add_argument("--stride", type=int, default=HR_SIZE, help="HR window stride, no overlap by default")
    parser.add_argument("--min-hr-mean", type=float, default=8.0, help="lower bound of the HR mean (0-255)")
    parser.add_argument("--min-valid-ratio", type=float, default=0.5, help="lower bound of the non-empty pixel ratio")
    parser.add_argument("--dn-scale", type=float, default=DN_SCALE)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the pairs and the split without writing files")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=3407)
    args = parser.parse_args()
    if args.hr_size != args.lr_size * 10:
        raise SystemExit(f"need 10x SR: hr_size must be lr_size*10, got {args.hr_size} vs {args.lr_size}")
    args.hr_dir = args.hr_dir.expanduser().resolve()
    args.lr_dir = args.lr_dir.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    return args


def run_jobs(jobs: List[dict], workers: int) -> List[dict]:
    """Run the cropping jobs serially or in a process pool."""
    results: List[dict] = []
    if workers <= 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs, 1):
            stats = crop_one_pair(job)
            _print_scene_stats(i, len(jobs), stats)
            results.append(stats)
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(crop_one_pair, job): job for job in jobs}
        done = 0
        for fut in as_completed(futs):
            stats = fut.result()
            done += 1
            _print_scene_stats(done, len(jobs), stats)
            results.append(stats)
    return results


def _print_scene_stats(done: int, total: int, stats: dict) -> None:
    print(
        f"[crop] {done}/{total} [{stats.get('split', '')}] {stats['stem']}: "
        f"written={stats['written']} skip_exist={stats['skipped_exist']} "
        f"skip_invalid={stats['skipped_invalid']}",
        flush=True,
    )


def _print_done(args: argparse.Namespace, written: int, skipped: int, moved: dict) -> None:
    print(f"[crop] done: tiles={written} skipped={skipped} "
          f"moved_to_val={moved.get('val', 0)} moved_to_test={moved.get('test', 0)}")
    print(f"[crop] train: {args.out / 'train'}")
    print(f"[crop] val:   {args.out / 'val'}")
    print(f"[crop] test:  {args.out / 'test'}")
    print(f"[crop] test scenes (full): {args.out / 'test_scenes'}")


def main() -> None:
    args = parse_args()
    info = match_hr_lr_pairs(args.hr_dir, args.lr_dir)
    print_pair_stats(info, args.hr_dir, args.lr_dir, max_lines=20 if args.dry_run else 8)
    pairs = info["pairs"]
    if args.max_scenes is not None:
        pairs = pairs[: args.max_scenes]
    if not pairs:
        raise SystemExit("[crop] no HR/LR pairs found")
    split_map = assign_scene_splits(
        pairs, args.train_ratio, args.val_ratio, args.test_ratio, args.split_seed,
    )
    print_split_assignment(split_map, args.hr_dir, args.lr_dir)
    if args.dry_run:
        print("[crop] dry-run, no files written")
        return
    make_split_dirs(args.out)
    write_split_json(args.out / "split.json", split_map, args)
    print(f"[crop] out={args.out}  HR {args.hr_size}x{args.hr_size}  "
          f"LR {args.lr_size}x{args.lr_size}  stride={args.stride}")
    results = run_jobs(build_jobs(split_map, args), args.workers)
    written = sum(s["written"] for s in results)
    skipped = sum(s["skipped_exist"] + s["skipped_invalid"] for s in results)
    moved = fill_empty_splits(args.out, args.val_ratio, args.test_ratio, args.split_seed)
    write_split_json(args.out / "split.json", split_map, args, moved_tiles=moved)
    link_test_scenes(split_map.get("test", []), args.out, args.hr_dir, args.lr_dir)
    _print_done(args, written, skipped, moved)


if __name__ == "__main__":
    main()
