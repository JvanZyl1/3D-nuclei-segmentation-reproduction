import os, tifffile, numpy as np

import torch
import torch.nn.functional as F

from dataset_utils import DatasetUtils


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

    def __getitem__(self, idx):    # this is missing the interpolation and the mirror padding (but I don't think mirror padding is needed)
        image = tifffile.imread(self.image_paths[idx])
        mask = tifffile.imread(self.mask_paths[idx])

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min)

        desired_height, desired_width = 139, 140
        image = DatasetUtils().apply_padding(image, desired_height, desired_width)
        mask = DatasetUtils.apply_padding(mask, desired_height, desired_width)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(0)

        return image, mask

    def __iter__(self):
        for i in len(self):
            yield self[i]


if __name__ == "__main__":
    images_dir = os.path.join("data", "images", "train", "Images")
    ground_truth_dir = os.path.join("data", "GroundTruth", "train", "GroundTruth_NDN")

    dataset = CellDataset(images_dir=images_dir, masks_dir=ground_truth_dir)
    print(len(dataset))
    max_h, max_w = 0, 0
    
    # fixed ur dataset 
    items = [item for item in dataset]