import glob
import os

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils import data
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TestDataset(data.Dataset):
    """Validation pairs for the teacher-forced CE check during training, read like the training set without flips."""

    def __init__(self, pngtxt_dir, resize_bak=True):
        super().__init__()
        self.resize_bak = resize_bak
        self.img_preproc = transforms.Compose([transforms.ToTensor()])
        hr_root = os.path.join(pngtxt_dir, 'HR') if os.path.isdir(os.path.join(pngtxt_dir, 'HR')) else pngtxt_dir
        found = glob.glob(os.path.join(hr_root, '**', '*.tif'), recursive=True)
        found += glob.glob(os.path.join(hr_root, '**', '*.tiff'), recursive=True)
        self.img_paths = sorted(set(found))
        if not self.img_paths:
            raise FileNotFoundError(f'No .tif/.tiff found under {hr_root}')
        print(len(self.img_paths))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_t = self.img_preproc(Image.open(img_path).convert('RGB'))
        with rasterio.open(img_path.replace('/HR/', '/LR/')) as ds:
            lr_t = torch.from_numpy(ds.read(indexes=[1, 2, 3], out_dtype=np.float32))
        if self.resize_bak and lr_t.shape[-2:] != gt_t.shape[-2:]:
            lr_t = F.interpolate(lr_t.unsqueeze(0), size=tuple(gt_t.shape[-2:]), mode='bicubic', align_corners=False).squeeze(0)
        return {'pixel_values': gt_t * 2.0 - 1.0, 'conditioning_pixel_values': lr_t * 2.0 - 1.0}

    def __len__(self):
        return len(self.img_paths)
