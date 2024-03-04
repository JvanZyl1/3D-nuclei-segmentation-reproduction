import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile

from unet_3d import UNet3D


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
        # on top of that, add potentially resizing and add (for sure) interpolation for the x/y/z mismatch
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

        image = F.pad(image, pad=(0, 0, self.padding, self.padding), mode='reflect')
        mask = F.pad(mask, pad=(0, 0, self.padding, self.padding), mode='reflect')

        return image, mask

    def display_single_frame(self, idx, frame_number=25):
        """
        :param idx: index of the 3d volume in the dataset we want to display
        :param frame_number: out of the stack of frames, which frame index we would like to show (default: 25)
        """
        image, _ = self.__getitem__(idx)
        frame = image[0, frame_number, :, :]

        plt.imshow(frame, cmap='gray')
        plt.title('Frame 25')
        plt.axis('off')
        plt.show()

    def display_stacked(self, idx):
        image, _ = self.__getitem__(idx)
        mip_image = torch.max(image, dim=1).values.squeeze()

        mip_image_np = mip_image.cpu().detach().numpy()

        plt.imshow(mip_image_np, cmap='gray')
        plt.title('Maximum Intensity Projection')
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    images_dir = os.path.join("data", "images", "train", "Images")
    ground_truth_dir = os.path.join("data", "GroundTruth", "train", "GroundTruth_NDN")

    dataset = CellDataset(images_dir=images_dir, masks_dir=ground_truth_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # tensors are of different dimensions and this needs to be fixed

    n_channels = 1
    n_classes = 2

    model = UNet3D(n_channels, n_classes)
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        break


