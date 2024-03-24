'''
Post-processing in QCANet
We performed (a) reinterpolation and (b) marker-based watershed
transformation on the semantic segmentation image output from NSN
and NDN. Reinterpolation restores the resolution of the image interpolated
for segmentation and identification. Marker-based watershed divides the
semantic segmentation region by watershed with the centre region of the
identified nucleus as a marker. Post-processing enables QCANet to execute
instance segmentation.
'''

from cell_dataset import PreProcessCellDataset
import os
import torch
import scipy
import tifffile


class PostProcessing():
    def __init__(self, image, mask):
        # Initialise image and mask
        self.image = image
        self.mask = mask

    def reinterpolation(self, image, type='cubic'):
        """
        Reinterpolation restores the resolution of the image interpolated
        for segmentation and identification.
        """
        image = image.squeeze(0)
        scale_factor = 1/2.1875
        if type == 'bicubic':
            img_resized = scipy.ndimage.zoom(image, (scale_factor, 1, 1), order=3)
            img_resized = torch.from_numpy(img_resized).unsqueeze(0)
            return img_resized
        if type == 'nearest':
            img_resized = scipy.ndimage.zoom(image, (scale_factor, 1, 1), order=0)
            img_resized = torch.from_numpy(img_resized).unsqueeze(0)
            return img_resized
        
    def remove_padding(self, path_image):
        """
        Remove padding from the image and mask
        """
        # 1) Find the bounding box of the mask
        # Remove the extras 
        image_augmented = path_image  # use the path_image string instead of self.image
        mask_augmented = self.mask
        path_original_image = path_image.replace("data_augmented", "data")

        # Find dimensions of the original image
        image = tifffile.imread(path_original_image)
        image_original_shape = image.shape

        # Find dimensions of the augmented image
        image_augmented = tifffile.imread(image_augmented)
        image_augmented_shape = image_augmented.shape
        # image_padding = mask_padding

        # Find the padding
        image_augmented_y = image_augmented_shape[2]
        image_augmented_x = image_augmented_shape[3]
        image_original_y = image_original_shape[1]
        image_original_x = image_original_shape[2]

        padding_y = (image_augmented_y - image_original_y) / 2
        padding_x = (image_augmented_x - image_original_x) / 2
        padding = (int(padding_y), int(padding_x))

        print(image_augmented.shape, mask_augmented.shape, padding_y, padding_x)

        # Crop the image and mask to the bounding box
        image_deaugmented = image_augmented[padding[0]:-padding[0], padding[1]:-padding[1]]
        mask_deaugmented = mask_augmented[padding[0]:-padding[0], padding[1]:-padding[1]]

        return image_deaugmented, mask_deaugmented


    def marker_based_watershed(self):
        """
        Marker-based watershed divides the
        semantic segmentation region by watershed with the centre region of the
        identified nucleus as a marker.
        """
        pass

    def post_processing(self, image, mask, path_image):
        self.image = image
        self.mask = mask
        self.image = self.reinterpolation(self.image)
        self.mask = self.reinterpolation(self.mask, type='nearest')

        self.image, self.mask = self.remove_padding(path_image)
        return self.image, self.mask
    
    def folder_post_processing(self, path_image, path_mask):
        """
        Perform post-processing on all images in a folder
        """
        # Save the image in a folder : data_post_processed\ .... same as the original folder
        new_image_folder_name = path_image.replace("data_augmented", "data_post_processed")
        new_mask_folder_name = path_mask.replace("data_augmented", "data_post_processed")
        os.makedirs(new_image_folder_name, exist_ok=True)
        os.makedirs(new_mask_folder_name, exist_ok=True)
        # Loop over each .tif image and mask in the folder
        image_files = os.listdir(path_image)
        mask_files = os.listdir(path_mask)
        idx = 0
        for image_file, mask_file in zip(image_files, mask_files):
            image_path = os.path.join(path_image, image_file)
            mask_path = os.path.join(path_mask, mask_file)
            self.image = tifffile.imread(image_path)
            self.mask = tifffile.imread(mask_path)
            image, mask = self.post_processing(self.image, self.mask, image_path)
            # Save the image and mask in the new folder
            tifffile.imwrite(new_image_folder_name, image)
            tifffile.imwrite(new_mask_folder_name, mask)
            print(idx)
            idx += 1
            
if __name__ == "__main__":
    path_image = os.path.join('data_augmented', 'Images', 'train', 'Images')
    path_mask = os.path.join('data_augmented', 'GroundTruth', 'train', 'GroundTruth_NSN')
    # Initial image and mask from folders
    image_initial = tifffile.imread(os.path.join(path_image, 'Emb01_t001.tif'))
    mask_initial = tifffile.imread(os.path.join(path_mask, 'Emb01_t001.tif'))
    post_processing = PostProcessing(image_initial, mask_initial)
    post_processing.folder_post_processing(path_image, path_mask)