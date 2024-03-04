import torch
import torch.nn.functional as F
import numpy as np
import os
import tifffile


class CellDataset(torch.utils.data.Dataset):

    def __init__(self, images_dir, masks_dir, padding=64):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.padding = padding

        self.image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.tif')])
        self.mask_paths = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.tif')])

        assert len(self.image_paths) == len(self.mask_paths), "The number of images and masks must be the same"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = tifffile.imread(self.image_paths[idx])
        mask = tifffile.imread(self.mask_paths[idx])

        image = image.astype(np.float32)
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(0)

        pad_size = (self.padding, self.padding, self.padding, self.padding, self.padding, self.padding)
        image = F.pad(image, pad=pad_size, mode='reflect')
        mask = F.pad(mask, pad=pad_size, mode='reflect')

        return image, mask