#!/usr/bin/env python3
"""Build the README slider GIFs and the images of the interactive demo page.

Usage: python scripts/make_demo.py --lr LR_DIR --pred PRED_DIR --hr HR_DIR --font ARIAL_TTF --out REPO_ROOT
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont

# scene, title, crop (x, y, size) on the 1-m grid
SCENES = [
    ("Tile_180_626", "Urban", (560, 760, 480)),
    ("Tile_252_458", "Water", (720, 700, 480)),
    ("Tile_180_459", "Transport", (590, 480, 480)),
    ("Tile_20_21", "Rural", (880, 670, 480)),
]


def read(path):
    with rasterio.open(path) as ds:
        return ds.read([1, 2, 3]).transpose(1, 2, 0)


def stretch(lr):
    x = lr.astype(np.float32)
    out = np.empty_like(x)
    for c in range(3):
        lo, hi = np.percentile(x[..., c], [2, 98])
        out[..., c] = (x[..., c] - lo) / max(hi - lo, 1e-6)
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))


def label(draw, xy, text, font, anchor):
    x, y = xy
    box = draw.textbbox((x, y), text, font=font, anchor=anchor)
    draw.rounded_rectangle((box[0] - 7, box[1] - 4, box[2] + 7, box[3] + 4), radius=5, fill=(0, 0, 0, 150))
    draw.text((x, y), text, font=font, fill=(255, 255, 255), anchor=anchor)


def frame(left, right, pos, font):
    w, h = left.size
    cut = int(round(pos * w))
    img = right.copy()
    img.paste(left.crop((0, 0, cut, h)), (0, 0))
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    if cut > 150:
        label(d, (12, 12), "Sentinel-2  10 m", font, "la")
    if w - cut > 150:
        label(d, (w - 12, 12), "CrossVAR  1 m", font, "ra")
    d.rectangle((cut - 1, 0, cut + 1, h), fill=(255, 255, 255, 235))
    cy = h // 2
    d.ellipse((cut - 15, cy - 15, cut + 15, cy + 15), fill=(255, 255, 255, 240), outline=(40, 40, 40, 200), width=1)
    d.polygon([(cut - 9, cy), (cut - 3, cy - 6), (cut - 3, cy + 6)], fill=(50, 50, 50, 255))
    d.polygon([(cut + 9, cy), (cut + 3, cy - 6), (cut + 3, cy + 6)], fill=(50, 50, 50, 255))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def slider_gif(left, right, path, font, n=36, hold=6):
    positions = [0.06 + 0.88 * (1 - np.cos(np.pi * i / (n - 1))) / 2 for i in range(n)]
    sweep = [0.06] * hold + positions + [0.94] * hold + positions[::-1]
    frames = [frame(left, right, p, font) for p in sweep]
    pal = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=55, loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ("lr", "pred", "hr", "font", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    gif_dir, page_dir = out / "assets" / "demo", out / "docs" / "images"
    gif_dir.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(a.font, 17)
    for name, title, (x, y, s) in SCENES:
        s2 = stretch(read(Path(a.lr) / f"{name}.tif"))
        pred = Image.fromarray(read(Path(a.pred) / f"{name}.tif"))
        ref = Image.fromarray(read(Path(a.hr) / f"{name}.tif"))
        size = pred.size
        s2_up = s2.resize(size, Image.Resampling.NEAREST)
        box = (x, y, x + s, y + s)
        slider_gif(s2_up.crop(box), pred.crop(box), gif_dir / f"{title.lower()}.gif", font)
        for img, tag in ((s2_up, "s2"), (pred, "crossvar"), (ref, "reference")):
            img.save(page_dir / f"{title.lower()}_{tag}.jpg", quality=88)
        print(title, "done")
    # wide overview: complete urban scene
    name = SCENES[0][0]
    s2 = stretch(read(Path(a.lr) / f"{name}.tif")).resize((640, 640), Image.Resampling.NEAREST)
    pred = Image.fromarray(read(Path(a.pred) / f"{name}.tif")).resize((640, 640), Image.Resampling.LANCZOS)
    slider_gif(s2, pred, gif_dir / "overview.gif", font)
    print("overview done")


if __name__ == "__main__":
    main()
