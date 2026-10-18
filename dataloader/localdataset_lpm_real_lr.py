import glob
import os
import random

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils import data


class LocalImageDataset_LPM(data.Dataset):
    """Training pairs: HR/*.tif (8-bit RGB) and LR/*.tif (float32 reflectance in [0, 1]) with the same names.

    Both images are flipped together; the LR is bicubically resized to the HR size and both are mapped to [-1, 1].
    """

    def __init__(self, pngtxt_dir, image_size=400, random_flip=True, resize_bak=True, scale=10, lr_size=40):
        super().__init__()
        self.resize_bak = bool(resize_bak)
        self.scale = int(scale)
        self.lr_size = int(lr_size)
        self.gt_size = self.lr_size * self.scale
        assert self.gt_size == image_size, f'image_size must be {self.lr_size}*{self.scale}'
        self.random_flip = bool(random_flip)
        hr_root = os.path.join(pngtxt_dir, 'HR') if os.path.isdir(os.path.join(pngtxt_dir, 'HR')) else pngtxt_dir
        found = glob.glob(os.path.join(hr_root, '**', '*.tif'), recursive=True)
        found += glob.glob(os.path.join(hr_root, '**', '*.tiff'), recursive=True)
        self.img_paths = sorted(set(found))
        if not self.img_paths:
            raise FileNotFoundError(f'No .tif/.tiff found under {hr_root}')
        print(f'[Dataset] HR count: {len(self.img_paths)} | LR will be derived via path replace (/HR/->/LR/)')

    def _sync_flip(self, hr_t, lr_t):
        if not self.random_flip:
            return hr_t, lr_t
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        if hflip:
            hr_t, lr_t = TF.hflip(hr_t), TF.hflip(lr_t)
        if vflip:
            hr_t, lr_t = TF.vflip(hr_t), TF.vflip(lr_t)
        return hr_t, lr_t

    def __getitem__(self, index):
        hr_path = self.img_paths[index]
        lr_path = hr_path.replace('/HR/', '/LR/')
        hr_pil = Image.open(hr_path).convert('RGB')
        if hr_pil.size != (self.gt_size, self.gt_size):
            raise ValueError(f'HR size mismatch: got {hr_pil.size}, expected {self.gt_size}x{self.gt_size} at {hr_path}')
        hr_t = TF.to_tensor(hr_pil)
        with rasterio.open(lr_path) as src:
            lr_t = torch.from_numpy(src.read([1, 2, 3]).astype(np.float32))
        if lr_t.shape[-2:] != (self.lr_size, self.lr_size):
            raise ValueError(f'LR size mismatch: got {tuple(lr_t.shape[-2:])}, expected {self.lr_size}x{self.lr_size} at {lr_path}')
        hr_t, lr_t = self._sync_flip(hr_t, lr_t)
        if self.resize_bak:
            lr_t = F.interpolate(lr_t.unsqueeze(0), size=(self.gt_size, self.gt_size), mode='bicubic', align_corners=False).squeeze(0)
        return {'conditioning_pixel_values': lr_t * 2.0 - 1.0, 'pixel_values': hr_t * 2.0 - 1.0}

    def __len__(self):
        return len(self.img_paths)
