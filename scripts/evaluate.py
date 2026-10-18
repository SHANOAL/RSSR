#!/usr/bin/env python3
"""Evaluate existing RGB predictions with the archived pyiqa protocol."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import pyiqa
import torch
from torchvision.transforms.functional import to_tensor
from pyiqa.default_model_configs import DEFAULT_CONFIGS


def image_map(root):
    paths = {}
    for path in Path(root).iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            if path.stem in paths:
                raise RuntimeError(f"Duplicate scene ID: {path.stem}")
            paths[path.stem] = path
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    hr, pred = image_map(args.hr_dir), image_map(args.pred_dir)
    if set(hr) != set(pred) or len(hr) != args.expected_count:
        raise RuntimeError(
            f"Pair audit failed: HR={len(hr)}, pred={len(pred)}, expected={args.expected_count}, "
            f"missing={sorted(set(hr)-set(pred))[:5]}, extra={sorted(set(pred)-set(hr))[:5]}"
        )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = {name: pyiqa.create_metric(name, device="cuda:0") for name in ("psnr", "ssim", "lpips")}
    values = {name: [] for name in metrics}
    started = time.perf_counter()
    with output.with_suffix(".csv").open("w", newline="") as stream, torch.inference_mode():
        writer = csv.DictWriter(stream, fieldnames=["scene_id", "psnr", "ssim", "lpips"])
        writer.writeheader()
        for index, name in enumerate(sorted(hr), 1):
            with Image.open(pred[name]) as image:
                image = image.convert("RGB")
                if image.size != (args.expected_size, args.expected_size):
                    raise RuntimeError(f"Unexpected output size: {name} {image.size}")
                prediction = to_tensor(image).unsqueeze(0).cuda()
            with Image.open(hr[name]) as image:
                image = image.convert("RGB")
                if image.size != (args.expected_size, args.expected_size):
                    raise RuntimeError(f"Unexpected HR size: {name} {image.size}")
                reference = to_tensor(image).unsqueeze(0).cuda()
            row = {"scene_id": name}
            for metric_name, metric in metrics.items():
                score = float(metric(prediction, reference).item())
                if not np.isfinite(score):
                    raise RuntimeError(f"Nonfinite metric: {name} {metric_name}={score}")
                row[metric_name] = score
                values[metric_name].append(score)
            writer.writerow(row)
            if index % 100 == 0 or index == len(hr):
                stream.flush()
                print(json.dumps({"method": args.method, "evaluated": index, "total": len(hr),
                                  "running_means": {k: float(np.mean(v)) for k, v in values.items()}}), flush=True)
    # Release full-resolution metric intermediates before the FID extraction.
    del prediction, reference, metrics
    torch.cuda.empty_cache()
    with torch.inference_mode():
        fid = pyiqa.create_metric("fid", device="cuda:0")
        fid_score = float(fid(args.hr_dir, args.pred_dir).item())
    result = {
        "method": args.method, "pairs": len(hr),
        **{key: float(np.mean(scores)) for key, scores in values.items()},
        "fid": fid_score, "pyiqa_version": pyiqa.__version__,
        "image_size": [args.expected_size, args.expected_size],
        "input_channels": "RGB", "crop_border": 0, "output_color_correction": False,
        "metric_configs": {key: DEFAULT_CONFIGS[key] for key in ("psnr", "ssim", "lpips", "fid")},
        "metric_options": "pyiqa defaults; same as archived evaluate_pyiqa.py",
        "hr_dir": str(Path(args.hr_dir).resolve()), "pred_dir": str(Path(args.pred_dir).resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
