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
import matplotlib.pyplot as plt
from mpl_interactions import ipyplot as iplt


def remove_mirror_padding(image, padding=64):
    """
    Remove padding from the image
    """
    image = image.squeeze(0)
    image = image[:, padding:-padding, padding:-padding]
    image = image.unsqueeze(0)
    return image

def reinterpolation(image, type='bicubic'):
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

def print_image_3D(image, slice_index=0):
    if len(image.shape) == 4:
        image = image.squeeze(0)
    def func(slice_index):
        #returns slics of image
        return image[int(slice_index)]
    n_ind = image.shape[0]
    control = iplt.imshow(func, slice_index=(0, n_ind-1), cmap='gray')
    plt.show()

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
    output_image_paths = [path.replace('data', 'data_output', 1) for path in image_paths]
    output_gt_paths = [path.replace('data', 'data_output', 1) for path in gt_paths]
    output_paths = (output_image_paths, output_gt_paths)
    return paths, output_paths

def post_processing(output_folder='data_output'):
    """
    Iterates through all the images in the folder where all output images of QCANet are saved
    It then undoes the interpolation and mirror padding (normal padding to make image square is left as it wont affect outputs)
    It then saves all the processed images to a new folder called data_post_processed
    """
    _, augmented_paths = find_folder_paths()
    #concatenate two lists in augmented paths
    augmented_paths = augmented_paths[0] + augmented_paths[1] #join the ground tuth and images paths into single list
    for folder_path in augmented_paths:
        
        #find amount of images in the folder
        list_image_paths = os.listdir(folder_path)
        list_image_paths = [path for path in list_image_paths if not 'flipped' in path]
        list_image_paths = sorted(list_image_paths)
        
        
        for image in list_image_paths:
            image_path = os.path.join(folder_path, image)
            ##Note: data_output is the folder were the output images of the neural net are saved (doesnt exist yet)
            new_path = image_path.replace(output_folder, 'data_post_processed')
            new_folder_path = folder_path.replace(output_folder, 'data_post_processed')
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            print(image_path, new_path)
            image = tifffile.imread(image_path)
            image = reinterpolation(image)
            image = remove_mirror_padding(image)
            tifffile.imwrite(new_path, image.numpy())
            