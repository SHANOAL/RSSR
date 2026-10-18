"""Split source images into train/val/test, holding out val/test tiles from train when there are too few scenes."""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

TIF_EXTS = {".tif", ".tiff"}
Pair = Tuple[Path, Path]
SplitMap = Dict[str, List[Pair]]


def compute_split_sizes(
    n: int, train_ratio: float, val_ratio: float, test_ratio: float,
) -> Tuple[int, int, int]:
    """Number of scenes per subset; with at least 3 scenes val and test get one scene each."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1, got {total}")
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        rest = n - 1
        n_val = max(1, rest // 2)
        n_test = rest - n_val
    return n_train, n_val, n_test


def assign_scene_splits(
    pairs: Sequence[Pair],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitMap:
    """Split by source image pair so that tiles of one scene stay in one subset."""
    n_train, n_val, n_test = compute_split_sizes(
        len(pairs), train_ratio, val_ratio, test_ratio,
    )
    items = list(pairs)
    random.Random(seed).shuffle(items)
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:n_train + n_val + n_test],
    }


def print_split_assignment(
    split_map: SplitMap, hr_dir: Path, lr_dir: Path, max_lines: int = 12,
) -> None:
    """Print the number of scenes and the file names of every subset."""
    counts = {k: len(v) for k, v in split_map.items()}
    print(f"[split] scenes: train={counts.get('train', 0)}  "
          f"val={counts.get('val', 0)}  test={counts.get('test', 0)}")
    if counts.get("val", 0) == 0 or counts.get("test", 0) == 0:
        print("[split] fewer than 3 scenes: val/test tiles are held out from the train tiles after cropping "
              "(same-scene hold-out, spatially dependent)")
    for split in ("train", "val", "test"):
        pairs = split_map.get(split, [])
        print(f"[split] {split} ({len(pairs)} scenes)")
        for hr_path, lr_path in pairs[:max_lines]:
            print(f"[split]   {hr_path.name}  <->  {lr_path.name}")
        if len(pairs) > max_lines:
            print(f"[split]   ... {len(pairs)} scenes in total")


def split_pairs_to_json(split_map: SplitMap, hr_dir: Path, lr_dir: Path) -> dict:
    """Path as a serializable relative path."""
    scenes = {}
    for split, pairs in split_map.items():
        recs = []
        for hr_path, lr_path in pairs:
            recs.append({
                "hr": _rel_or_name(hr_path, hr_dir),
                "lr": _rel_or_name(lr_path, lr_dir),
            })
        scenes[split] = recs
    return scenes


def write_split_json(
    path: Path,
    split_map: SplitMap,
    args,
    moved_tiles: dict = None,
) -> None:
    """Write the split list for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": int(args.split_seed),
        "ratios": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": float(args.test_ratio),
        },
        "scene_counts": {k: len(v) for k, v in split_map.items()},
        "scenes": split_pairs_to_json(split_map, args.hr_dir, args.lr_dir),
        "moved_tiles": moved_tiles or {},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[split] manifest -> {path}")


def make_split_dirs(out_dir: Path) -> None:
    """Create the HR and LR folders of train/val/test."""
    for split in ("train", "val", "test"):
        (out_dir / split / "HR").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "LR").mkdir(parents=True, exist_ok=True)


def list_split_hr_tiles(split_dir: Path) -> List[Path]:
    """All HR tiles of one subset."""
    hr_root = split_dir / "HR"
    if not hr_root.is_dir():
        return []
    return sorted(
        p for p in hr_root.rglob("*")
        if p.is_file() and p.suffix.lower() in TIF_EXTS
    )


def move_tile_pair(hr_path: Path, src_split: Path, dst_split: Path) -> None:
    """Move an HR/LR tile pair from one subset to another, keeping the relative path."""
    rel = hr_path.relative_to(src_split / "HR")
    lr_src = src_split / "LR" / rel
    hr_dst = dst_split / "HR" / rel
    lr_dst = dst_split / "LR" / rel
    hr_dst.parent.mkdir(parents=True, exist_ok=True)
    lr_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(hr_path), str(hr_dst))
    if lr_src.is_file():
        shutil.move(str(lr_src), str(lr_dst))


def fill_empty_splits(
    out_dir: Path, val_ratio: float, test_ratio: float, seed: int,
) -> dict:
    """Hold out val/test tiles from train when the scenes cannot form their own subsets."""
    moved = {"val": 0, "test": 0}
    train_tiles = list_split_hr_tiles(out_dir / "train")
    random.Random(seed).shuffle(train_tiles)
    remain = train_tiles
    for split, ratio in (("val", val_ratio), ("test", test_ratio)):
        if list_split_hr_tiles(out_dir / split):
            continue
        if len(remain) < 3:
            print(f"[split] not enough train tiles to fill {split}")
            continue
        k = max(1, int(round(len(train_tiles) * ratio)))
        k = min(k, max(1, len(remain) // 2))
        take, remain = remain[:k], remain[k:]
        for hr_path in take:
            move_tile_pair(hr_path, out_dir / "train", out_dir / split)
        moved[split] = len(take)
        print(f"[split] fallback: moved {len(take)} tiles train -> {split}")
    return moved


def link_or_copy(src: Path, dst: Path) -> None:
    """Link the source image, or copy it if linking fails."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def link_test_scenes(test_pairs: Sequence[Pair], out_dir: Path, hr_dir: Path, lr_dir: Path) -> None:
    """Link the complete test images into test_scenes/ for complete-scene inference."""
    if not test_pairs:
        return
    root = out_dir / "test_scenes"
    for hr_path, lr_path in test_pairs:
        link_or_copy(hr_path, root / "HR" / _rel_or_name(hr_path, hr_dir))
        link_or_copy(lr_path, root / "LR" / _rel_or_name(lr_path, lr_dir))
    print(f"[split] original test scenes -> {root}")


def _rel_or_name(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name
