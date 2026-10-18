# -*- coding: utf-8 -*-
"""Tiled decoding of large latents, which bounds the memory of the decoder attention."""
from __future__ import annotations

from typing import List, Tuple

import torch

VAE_DOWNSAMPLE = 16
# latents with at most this many positions are decoded in one pass (a 160x160 input gives a 100x100 latent)
MAX_FULL_DECODE_LATENT_PIXELS = 100 * 100
DEFAULT_DECODE_TILE_LAT = 64
DEFAULT_LATENT_OVERLAP = 4


def latent_slice_ranges(length: int, tile_size: int, overlap: int) -> List[Tuple[int, int]]:
    """Start and end of every tile along one latent axis; the last tile is aligned to the border."""
    if length <= tile_size:
        return [(0, length)]

    step = max(tile_size - overlap, 1)
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + tile_size, length)
        if end == length and end - start < tile_size:
            start = max(0, length - tile_size)
            end = length
        ranges.append((start, end))
        if end >= length:
            break
        start += step

    deduped: List[Tuple[int, int]] = []
    for item in ranges:
        if not deduped or deduped[-1] != item:
            deduped.append(item)
    return deduped


def fhat_to_img_tiled(
    vae,
    preds: torch.Tensor,
    decode_tile_lat: int = DEFAULT_DECODE_TILE_LAT,
    latent_overlap: int = DEFAULT_LATENT_OVERLAP,
) -> torch.Tensor:
    """
    Decode a latent to RGB; large latents are decoded in overlapping tiles and blended.

    Args:
        vae: the VQVAE
        preds: [B, C, h_lat, w_lat] latent
        decode_tile_lat: tile side in latent positions
        latent_overlap: tile overlap in latent positions

    Returns:
        [B, 3, h_lat*16, w_lat*16] in [-1, 1], as returned by vae.fhat_to_img
    """
    _, _, h_lat, w_lat = preds.shape
    if h_lat * w_lat <= MAX_FULL_DECODE_LATENT_PIXELS:
        return vae.fhat_to_img(preds)

    out_h = h_lat * VAE_DOWNSAMPLE
    out_w = w_lat * VAE_DOWNSAMPLE
    out = torch.zeros((preds.shape[0], 3, out_h, out_w), device=preds.device, dtype=torch.float32)
    weights = torch.zeros((preds.shape[0], 1, out_h, out_w), device=preds.device, dtype=torch.float32)

    row_ranges = latent_slice_ranges(h_lat, decode_tile_lat, latent_overlap)
    col_ranges = latent_slice_ranges(w_lat, decode_tile_lat, latent_overlap)

    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            chunk = preds[:, :, r0:r1, c0:c1]
            img_chunk = vae.fhat_to_img(chunk).float()

            or0, oc0 = r0 * VAE_DOWNSAMPLE, c0 * VAE_DOWNSAMPLE
            or1 = or0 + img_chunk.shape[2]
            oc1 = oc0 + img_chunk.shape[3]

            out[:, :, or0:or1, oc0:oc1] += img_chunk
            weights[:, :, or0:or1, oc0:oc1] += 1.0

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return (out / weights.clamp(min=1e-8)).to(dtype=preds.dtype)
