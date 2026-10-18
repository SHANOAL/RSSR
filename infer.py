"""Complete-scene inference with aligned window decoding.

Every window of the final 25x25 latent grid starts on the input pixel grid (overlap 5), is predicted and refined
on its own, decoded at the training size, and the decoded windows are blended with Gaussian weights. Writes
<test_root>/VARPrediction/<name> for every <test_root>/LR/<name>; metrics are computed by scripts/evaluate.py.

Options (see utils/arg_util.py): --refine_temperature (0 = deterministic residual refinement), --fusion
rgb | latent_window | latent | none, --overlap, --export_stages DIR, --report_json FILE.
"""
import glob
import json
import math
import os
import time

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import dist
from models import build_var
from myutils.vae_tiled_decode import fhat_to_img_tiled
from utils import arg_util
from utils.ckpt_compat import load_var_and_vae, load_var_checkpoint_and_cfg

FUSION_MODES = {'rgb': 'rgb_gaussian', 'latent_window': 'gaussian_latent_window', 'latent': 'gaussian_latent', 'none': 'no_overlap_latent'}
RSCALE = 10


def percentile_stretch01(img_hw3: np.ndarray, pmin=2.0, pmax=98.0) -> np.ndarray:
    """Per-channel percentile stretch of an (H, W, 3) image to [0, 1], for display only."""
    out = np.nan_to_num(img_hw3, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=True)
    for c in range(out.shape[-1]):
        lo, hi = np.nanpercentile(out[..., c], pmin), np.nanpercentile(out[..., c], pmax)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(out[..., c])), float(np.nanmax(out[..., c]))
            if hi <= lo:
                out[..., c] = np.clip(out[..., c], 0.0, 1.0)
                continue
        out[..., c] = (out[..., c] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def normalize_lr_to01(lr_np: np.ndarray, dn_scale: float = 10000.0) -> np.ndarray:
    """LR to [0, 1]: float reflectance passes through, int16 DN (x10000) is rescaled."""
    x = np.nan_to_num(lr_np.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.max() > 1.5:
        x = x / dn_scale
    return np.clip(x, 0.0, 1.0)


def read_lr_geotiff_rgb(image_path: str) -> np.ndarray:
    """The first three GeoTIFF bands (stored as R, G, B) as an (H, W, 3) float32 array."""
    with rasterio.open(image_path) as src:
        lr_np = src.read([1, 2, 3]).astype(np.float32)
    return np.transpose(lr_np, (1, 2, 0))


def numpy_to_pil(images: np.ndarray):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype('uint8')
    return [Image.fromarray(image) for image in images]


def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
    return images.cpu().permute(0, 2, 3, 1).float().numpy()


def gaussian_weights(tile_width, tile_height, nbatches):
    """Gaussian mask of weights for tile contributions (latent grid, 32 channels).

    numpy exp/sqrt are kept on purpose: math.exp differs in the last bit for some
    entries, which changes the fused latents by rounding.
    """
    var = 0.01
    midpoint = (tile_width - 1) / 2
    x_probs = [np.exp(-(x - midpoint) * (x - midpoint) / (tile_width * tile_width) / (2 * var)) / np.sqrt(2 * np.pi * var) for x in range(tile_width)]
    midpoint = tile_height / 2
    y_probs = [np.exp(-(y - midpoint) * (y - midpoint) / (tile_height * tile_height) / (2 * var)) / np.sqrt(2 * np.pi * var) for y in range(tile_height)]
    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device=dist.get_device()), (nbatches, 32, 1, 1))


def tile_offsets(index, count, size, tile_size, tile_overlap):
    return size - tile_size if index == count - 1 else max(index * (tile_size - tile_overlap), 0)


def grid_count(size, tile_size, tile_overlap):
    count, cur = 0, 0
    while cur < size:
        cur = max(count * tile_size - tile_overlap * count, 0) + tile_size
        count += 1
    return count


def main(args: arg_util.Args, test_root: str):
    ck, var_sd, ck_cfg = load_var_checkpoint_and_cfg(var_ckpt_path=args.var_test_path, fallback_patch_nums=args.patch_nums, verbose=True)
    args.patch_nums = tuple(ck_cfg['patch_nums'])
    args.depth = int(ck_cfg['depth'])
    vae, var = build_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums,
        depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
    )
    # formal evaluation never continues with partially loaded weights
    load_var_and_vae(vae, var, var_ckpt_path=args.var_test_path, vae_pretrain_path=args.vae_model_path, ck=ck, var_sd=var_sd)
    del ck, var_sd

    print(f'[INFO] Using test root: {test_root}')
    image_names = []
    for e in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.PNG', '*.JPG', '*.JPEG', '*.TIF', '*.TIFF'):
        image_names.extend(sorted(glob.glob(os.path.join(test_root, 'LR', e))))
    if not image_names:
        raise FileNotFoundError(f'No test images found under {os.path.join(test_root, "LR")}')
    report_json = args.report_json.strip()
    stage_export_dir = args.export_stages.strip()
    export_stages = bool(stage_export_dir)
    if args.fusion not in FUSION_MODES:
        raise ValueError(f'Unknown --fusion {args.fusion}; expected one of {sorted(FUSION_MODES)}')
    fusion_mode = FUSION_MODES[args.fusion]
    refine_temperature = float(args.refine_temperature)
    print(f'[INFO] fusion={args.fusion} overlap={args.overlap} refine_temperature={refine_temperature}', flush=True)

    model_seconds, generated_scenes = 0.0, 0
    wall_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tile_size = int(var.patch_nums[-1])     # tiles of the final token grid
    for image_name in tqdm(image_names):
        lr_np = normalize_lr_to01(read_lr_geotiff_rgb(image_name))
        H_lr, W_lr, _ = lr_np.shape
        ori_w, ori_h = W_lr * RSCALE, H_lr * RSCALE
        # round the HR size (not the LR size) to the VAE stride
        target_H = max(math.ceil((H_lr * RSCALE) / 16) * 16, tile_size * 16)
        target_W = max(math.ceil((W_lr * RSCALE) / 16) * 16, tile_size * 16)
        lr_inp = torch.from_numpy(lr_np).permute(2, 0, 1).unsqueeze(0).contiguous()
        lr_inp = F.interpolate(lr_inp, size=(target_H, target_W), mode='bicubic', align_corners=False) * 2.0 - 1.0
        lr_inp = lr_inp.to(dist.get_device(), non_blocking=True)

        B = lr_inp.shape[0]
        h, w = math.ceil(lr_inp.shape[2] / 16), math.ceil(lr_inp.shape[3] / 16)
        tile_overlap = 0 if fusion_mode == 'no_overlap_latent' else args.overlap
        if fusion_mode in {'gaussian_latent', 'rgb_gaussian', 'gaussian_latent_window'}:
            tile_weights = gaussian_weights(tile_size, tile_size, 1)
        else:
            tile_weights = torch.ones((1, 1, tile_size, tile_size), device=lr_inp.device, dtype=lr_inp.dtype)
        grid_rows, grid_cols = grid_count(h, tile_size, tile_overlap), grid_count(w, tile_size, tile_overlap)
        offsets = [[(tile_offsets(r, grid_rows, h, tile_size, tile_overlap), tile_offsets(c, grid_cols, w, tile_size, tile_overlap))
                    for c in range(grid_cols)] for r in range(grid_rows)]

        recon_pred = []
        stage_recon_pred = [] if export_stages else None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for row in range(grid_rows):
            # one batch per tile row
            lr4var = torch.cat([lr_inp[:, :, ox * 16:(ox + tile_size) * 16, oy * 16:(oy + tile_size) * 16] for ox, oy in offsets[row]], dim=0)
            with torch.inference_mode(), torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                result = var.autoregressive_infer(lr4var, top_k=1, top_p=0.75, return_stages=export_stages,
                                                  refine_temperature=refine_temperature)
            if export_stages:
                recon_pred.append(result[0])
                stage_recon_pred.append(torch.stack(result[1], dim=0))
            else:
                recon_pred.append(result)

        if fusion_mode != 'rgb_gaussian':
            preds = torch.zeros((B, 32, h, w), device=lr_inp.device)
            contributors = torch.zeros((B, 32, h, w), device=lr_inp.device)
            for row in range(grid_rows):
                for col, (ox, oy) in enumerate(offsets[row]):
                    preds[:, :, ox:ox + tile_size, oy:oy + tile_size] += recon_pred[row][col].unsqueeze(0) * tile_weights
                    contributors[:, :, ox:ox + tile_size, oy:oy + tile_size] += tile_weights
            preds /= contributors.clamp_min_(1e-8)
        if fusion_mode == 'gaussian_latent_window':
            # decode the fused latent in the same windows instead of in one pass
            recon_pred = [[preds[0, :, ox:ox + tile_size, oy:oy + tile_size] for ox, oy in offsets[row]] for row in range(grid_rows)]
        if fusion_mode in {'rgb_gaussian', 'gaussian_latent_window'}:
            rgb_preds = torch.zeros((B, 3, h * 16, w * 16), device=lr_inp.device)
            rgb_contributors = torch.zeros((B, 1, h * 16, w * 16), device=lr_inp.device)
            rgb_weights = F.interpolate(tile_weights[:, :1].float(), size=(tile_size * 16, tile_size * 16), mode='bilinear', align_corners=False)
            with torch.no_grad():
                for row in range(grid_rows):
                    for col, (ox, oy) in enumerate(offsets[row]):
                        tile_rgb = fhat_to_img_tiled(vae, recon_pred[row][col].unsqueeze(0)).float()
                        x0, x1, y0, y1 = ox * 16, (ox + tile_size) * 16, oy * 16, (oy + tile_size) * 16
                        rgb_preds[:, :, x0:x1, y0:y1] += tile_rgb * rgb_weights
                        rgb_contributors[:, :, x0:x1, y0:y1] += rgb_weights
            recon_B3HW = rgb_preds / rgb_contributors.clamp_min_(1e-8)
        else:
            with torch.no_grad():
                recon_B3HW = fhat_to_img_tiled(vae, preds)

        stem = os.path.splitext(os.path.basename(image_name))[0]
        if export_stages:
            os.makedirs(stage_export_dir, exist_ok=True)
            for stage_idx, pn in enumerate(var.patch_nums):
                stage_canvas = torch.zeros((B, 32, h, w), device=lr_inp.device)
                stage_contributors = torch.zeros((B, 32, h, w), device=lr_inp.device)
                for row in range(grid_rows):
                    for col, (ox, oy) in enumerate(offsets[row]):
                        stage_canvas[:, :, ox:ox + tile_size, oy:oy + tile_size] += stage_recon_pred[row][stage_idx, col].unsqueeze(0) * tile_weights
                        stage_contributors[:, :, ox:ox + tile_size, oy:oy + tile_size] += tile_weights
                stage_canvas /= stage_contributors.clamp_min_(1e-8)
                with torch.no_grad():
                    stage_rgb = (fhat_to_img_tiled(vae, stage_canvas) + 1).mul(0.5).clamp(0.0, 1.0)
                stage_pil = numpy_to_pil(pt_to_numpy(stage_rgb))[0].resize((ori_w, ori_h), Image.BICUBIC)
                stage_pil.save(os.path.join(stage_export_dir, f'{stem}_S{stage_idx + 1:02d}_{pn}x{pn}.png'))
            input_pil = Image.fromarray((percentile_stretch01(lr_np) * 255.0).round().astype(np.uint8), mode='RGB')
            input_pil.resize((ori_w, ori_h), Image.BICUBIC).save(os.path.join(stage_export_dir, f'{stem}_input.png'))
            hr_path = os.path.join(test_root, 'HR', os.path.basename(image_name))
            if os.path.isfile(hr_path):
                with rasterio.open(hr_path) as src:
                    hr_np = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)
                if hr_np.max() > 1.5:
                    hr_np /= 255.0
                Image.fromarray((np.clip(hr_np, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode='RGB').save(
                    os.path.join(stage_export_dir, f'{stem}_reference.png'))

        recon_B3HW = (recon_B3HW + 1).mul_(0.5).clamp_(0.0, 1.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - start_time
        model_seconds += duration
        recon_pils = numpy_to_pil(pt_to_numpy(recon_B3HW))
        print(f'[BENCH] {os.path.basename(image_name)} model_seconds={duration:.6f}', flush=True)

        output_dir = os.path.dirname(image_name).replace('/LR', '/VARPrediction/').replace('/HR', '/VARPrediction/')
        os.makedirs(output_dir, exist_ok=True)
        for idx in range(B):
            image = recon_pils[idx].resize((ori_w, ori_h), Image.BICUBIC)
            if export_stages:
                image.save(os.path.join(stage_export_dir, f'{stem}_final_diffusion.png'))
            image.save(os.path.join(output_dir, os.path.basename(image_name)))
            generated_scenes += 1

    wall_seconds = time.perf_counter() - wall_start
    if report_json:
        result = {
            'method': args.exp_name,
            'checkpoint': args.var_test_path,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'precision': 'fp16-autocast',
            'fusion': args.fusion,
            'tile_overlap': tile_overlap,
            'parameters': sum(p.numel() for p in var.parameters()) + sum(p.numel() for p in vae.parameters()),
            'requested_scenes': len(image_names),
            'generated_scenes': generated_scenes,
            'model_seconds': model_seconds,
            'wall_seconds': wall_seconds,
            'mean_model_ms': 1000.0 * model_seconds / max(generated_scenes, 1),
            'mean_end_to_end_ms': 1000.0 * wall_seconds / max(generated_scenes, 1),
            'peak_vram_mib': torch.cuda.max_memory_allocated() / (1024.0 ** 2) if torch.cuda.is_available() else None,
        }
        os.makedirs(os.path.dirname(os.path.abspath(report_json)), exist_ok=True)
        with open(report_json, 'w') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    torch.backends.cudnn.benchmark = False     # fixed convolution algorithms make the outputs reproducible
    main(args, test_root=os.path.abspath((args.test_root or 'datasets/test').strip()))
