import os, tifffile, numpy as np

import torch

from pprint import pprint 
from dataset_utils import DatasetUtils

import scipy
import matplotlib.pyplot as plt
from mpl_interactions import ipyplot as iplt

class PreProcessCellDataset(torch.utils.data.Dataset):

    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        self.image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.tif')])
        self.mask_paths = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.tif')])

        assert len(self.image_paths) == len(self.mask_paths), "The number of images and masks must be the same"
        assert self.check_images_same_size(), "Images are not the same size"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        ##do the same as in CellDataset but without the preprocessing
        image = tifffile.imread(self.image_paths[idx])
        mask = tifffile.imread(self.mask_paths[idx])

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(0)

        return image, mask
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    ##check images same size
    def check_images_same_size(self):
        prev_image_name = None
        for i in range(len(self)):
            image, mask = self[i]
            image_name = self.image_paths[i]
            if i == 0:
                max_h, max_w = image.shape[1], image.shape[2]
                prev_image_name = image_name
            else:
                if image.shape[1] != max_h or image.shape[2] != max_w:
                    print(f"Images are not the same size: {prev_image_name} and {image_name}")
                    return False
                prev_image_name = image_name
        return True

    def print_image(self, image, slice_index=0):
        if len(image.shape) == 4:
            image = image.squeeze(0)
        image_slice = image[slice_index]
        plt.imshow(image_slice, cmap='gray')
        plt.show()

    def print_image_3D(self, image, slice_index=0):
        if len(image.shape) == 4:
            image = image.squeeze(0)
        def func(slice_index):
            #returns slics of image
            return image[int(slice_index)]
        n_ind = image.shape[0]
        control = iplt.imshow(func, slice_index=(0, n_ind-1), cmap='gray')
        plt.show()




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

        mask_min = mask.min()
        mask_max = mask.max()
        mask = (mask - mask_min) / (mask_max - mask_min)

        
        desired_height, desired_width = 140, 140
        image = DatasetUtils().apply_padding(image, desired_height, desired_width)
        mask = DatasetUtils().apply_padding(mask, desired_height, desired_width)
        
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # image, mask = self.interpolate(image, type='bicubic'), self.interpolate(mask, type='nearest')
        # mirror padding
        # image = torch.nn.functional.pad(image, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        # mask = torch.nn.functional.pad(mask, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(0)

        return image, mask

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


    def interpolate(self, image, type='cubic'):
        image = image.squeeze(0)
        scale_factor = 2.1875
        if type == 'bicubic':
            #use bicubic interpolation to resize the image along z-axis by scale factor of 2.1875 (not available with pytorch)
            img_resized = scipy.ndimage.zoom(image, (scale_factor, 1, 1), order=3)
            img_resized = torch.from_numpy(img_resized).unsqueeze(0)#convert to tensor, re-add channel dimension
            return img_resized #shape: [1, 112, 139, 140]
        if type == 'nearest':
            #use nearest interpolation for resizing mask to avoid having pixels with values between 0 and 1
            img_resized = scipy.ndimage.zoom(image, (scale_factor, 1, 1), order=0)
            img_resized = torch.from_numpy(img_resized).unsqueeze(0)#convert to tensor, re-add channel dimension    
            return img_resized #shape: [1, 112, 139, 140]
            
        
    def print_image(self, image, slice_index=0):
        if len(image.shape) == 4:
            image = image.squeeze(0)
        image_slice = image[slice_index]
        plt.imshow(image_slice, cmap='gray')
        plt.show()

    def print_image_3D(self, image, slice_index=0):
        if len(image.shape) == 4:
            image = image.squeeze(0)
        def func(slice_index):
            #returns slics of image
            return image[int(slice_index)]
        n_ind = image.shape[0]
        control = iplt.imshow(func, slice_index=(0, n_ind-1), cmap='gray')
        plt.show()

    def augment_data(self, image, mask):
        # For each image in the dataset, we want 4 final images:
        # 1. Original image
        # 2. Image flipped along x-axis
        # 3. Image flipped along y-axis
        # 4. Image flipped along x-axis and y-axis
        # Image & Mask : [C, Z, Y, X]
        image_flipped_x = torch.flip(image, [3])
        image_flipped_y = torch.flip(image, [2])
        image_flipped_xy = torch.flip(image, [2, 3])

        mask_flipped_x = torch.flip(mask, [3])
        mask_flipped_y = torch.flip(mask, [2])
        mask_flipped_xy = torch.flip(mask, [2, 3])

        images = [image, image_flipped_x, image_flipped_y, image_flipped_xy]

        masks = [mask, mask_flipped_x, mask_flipped_y, mask_flipped_xy]
        return images, masks



def find_folder_paths():
    ##iterates through all subfolders in data directory and makes list of path names
    data_image_dir = os.path.join('data', 'Images')
    data_gt_dir = os.path.join('data', 'GroundTruth')
    image_paths = []    
    gt_paths = []
    for root, dirs, files in os.walk(data_image_dir):
        if not dirs:  #if directory has no subdirectories
            image_paths.append(root) 
    for root, dirs, files in os.walk(data_gt_dir):
        if not dirs: 
            gt_paths.append(root)
    paths = (image_paths, gt_paths)
    augmented_image_paths = [path.replace('data', 'data_augmented', 1) for path in image_paths]
    augmented_gt_paths = [path.replace('data', 'data_augmented', 1) for path in gt_paths]
    augmented_paths = (augmented_image_paths, augmented_gt_paths)
    return paths, augmented_paths



def create_preprocessing_images():
    ##Creates new folder 'data_augmented' which contains all the images after preprocessing
    ##note: each image in original data folder becomes 4 images in augmented folder due to data augmentation
    paths, augmented_paths = find_folder_paths()
    # Pair 'test' folders
    print("Image folder: ", paths[0][0])
    print("Ground truth folder: ", paths[1][0])
    process_folder_pair(paths[0][0], paths[1][0], augmented_paths[0][0], augmented_paths[1][0])

    # Pair 'train' folders
    for j in range(1, len(paths[1])):  # iterate over ground truth folders
        print("Image folder: ", paths[0][1])
        print("Ground truth folder: ", paths[1][j])
        process_folder_pair(paths[0][1], paths[1][j], augmented_paths[0][1], augmented_paths[1][j])

def process_folder_pair(images_dir, masks_dir, augmented_images_dir, augmented_masks_dir):
    dataset = CellDataset(images_dir=images_dir, masks_dir=masks_dir)
    for k in range(len(dataset)):
        item = dataset[k]  # this will call the __getitem__ function
        image, mask = item
        images_augmented, masks_augmented = dataset.augment_data(image, mask)
        
        # Save the augmented images and masks
        for l in range(len(images_augmented)):
            # Get the original image name and append the augmentation type
            original_image_name = os.path.splitext(os.path.basename(dataset.image_paths[k]))[0]
            original_mask_name = os.path.splitext(os.path.basename(dataset.mask_paths[k]))[0]
            augmentation_type = "_flipped_x" if l == 1 else "_flipped_y" if l == 2 else "_flipped_xy" if l == 3 else ""
            image_augmented_name = original_image_name + augmentation_type + ".tif"
            mask_augmented_name = original_mask_name + augmentation_type + ".tif"
            
            image_augmented_path = os.path.join(augmented_images_dir, image_augmented_name)
            mask_augmented_path = os.path.join(augmented_masks_dir, mask_augmented_name)
            
            # Create the directories if they do not exist
            os.makedirs(os.path.dirname(image_augmented_path), exist_ok=True)
            os.makedirs(os.path.dirname(mask_augmented_path), exist_ok=True)
            
            # Save the images
            tifffile.imwrite(image_augmented_path, images_augmented[l].numpy())
            tifffile.imwrite(mask_augmented_path, masks_augmented[l].numpy())



if __name__ == "__main__":
    """
    run create_preprocessing_images() to create new folder 'data_augmented' with preprocessed images
    leave commented out otherwise will run each time
    """
    
    #create_preprocessing_images()

    ############################################################################################################
    ##Following code is just to print an example image and mask (IGNORE)
    print_example_image = False
    if print_example_image:
        print("PRINTING PREPROCESSED IMAGE AND MASK")
        images_dir = os.path.join("data_augmented", "images", "train", "Images")
        ground_truth_dir = os.path.join("data_augmented", "GroundTruth", "train", "GroundTruth_NDN")
        dataset_pre = PreProcessCellDataset(images_dir=images_dir, masks_dir=ground_truth_dir)
        print(len(dataset_pre))

        items = [item for item in dataset_pre]
        for item in items[-1:]:
            image, mask = item
            pprint({'image': image.shape, 'mask': mask.shape})
            dataset_pre.print_image_3D(mask)
            dataset_pre.print_image_3D(image)
        
        print("PRINTING ORIGINAL IMAGE AND MASK")
        images_dir = os.path.join("data", "images", "train", "Images")
        ground_truth_dir = os.path.join("data", "GroundTruth", "train", "GroundTruth_NDN")
        dataset = CellDataset(images_dir=images_dir, masks_dir=ground_truth_dir)
        print(len(dataset))
        mask_path = 'data/GroundTruth/train/GroundTruth_QCANet/Emb01_t001.tif'
        image_path = 'data_augmented/images/train/Images/Emb01_t001.tif'
        max_h, max_w = 0, 0
        # fixed ur dataset 
        items = [item for item in dataset]  # load the entire thing into memory
        for item in items[-1:]:             # let's print the last 5

            image, mask = item 
            pprint({'image': image.shape, 'mask': mask.shape})
            dataset.print_image_3D(mask)


            # Augment data
            images_augmented, masks_augmented = dataset.augment_data(image, mask)
            # Verify augmentation ?
            verif_aug = True
            # Display each image in images and visually verify that the augmentation is correct
            if verif_aug:
                print("PRINTING AUGMENTED IMAGES AND MASKS")
                for i in range(len(images_augmented)):
                    dataset.print_image_3D(images_augmented[i])
                    #dataset.print_image(masks_augmented[i])
    