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

class PostProcessing():
    def __init__(self, image_path, mask_path):
        # Initialise image and mask
        dataset = PreProcessCellDataset(image_path, mask_path)
        self.image, self.mask = dataset.__getitem__(0)

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
        image_augmented = self.image
        mask_augmented = self.mask
        image_original = image_augmented.split('_')[:2] # remove _flipped etc
        path_original_image = path_original_image.replace("data_augmented", "data")
        path_original_image = path_original_image + '/' + image_original + '.tif'

        # Find dimensions of the original image
        image = tifffile.imread(path_original_image)
        image_original_shape = image.shape

        # Find dimensions of the augmented image
        image_augmented = tifffile.imread(image_augmented)
        image_augmented_shape = image_augmented.shape
        # image_padding = mask_padding

        # Find the padding
        padding = (image_augmented_shape - image_original_shape) / 2

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

        idx = 0
        for image, mask in zip(path_image, path_mask):
            self.image = image
            self.mask = mask
            image, mask = self.post_processing(self.image, self.mask, path_image)
            # Save the image and mask in the new folder
            tifffile.imwrite(new_image_folder_name, image.numpy())
            tifffile.imwrite(new_mask_folder_name, mask.numpy())
            print(idx)
            idx += 1
            
if __name__ == "__main__":
    path_image = 'data_augmented\Images\train\Images'
    path_mask = 'data_augmented\GroundTruth\train\GroundTruth_NSN'
    post_processing = PostProcessing()
    post_processing.folder_post_processing(path_image, path_mask)

