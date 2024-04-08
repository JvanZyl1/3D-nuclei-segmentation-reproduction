from cell_dataset import CellDataset
import os, torch, torch.optim as optim
from unet_3d import NSN, NDN
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt
from mayavi import mlab
import tifffile
#load trained models (NSN saved using pickle, unlike NDN)
n_channels = 1
ndn_model = NDN(n_channels=n_channels) 
nsn_model = NSN(n_channels=n_channels)


#note: model trained on CUDA, use map_location input to load model on CPU
ndn_model.load_state_dict(torch.load("replication/src/ndn (1).pth", map_location=torch.device('cpu')))
nsn_model.load_state_dict(torch.load("replication/src/nsn.pth", map_location=torch.device('cpu')))
#get test images and test ground truth
images_dir = os.path.join("data", "images", "test", "Images")
ground_truth_dir = os.path.join("data", "GroundTruth", "test",
                                    "GroundTruth_QCANet")

    
def run_watershed(nsn_output, ndn_output, slice):
    #normalise output as cv.threshold required range 0 to 255
    nsn_output_norm = ((nsn_output - nsn_output.min()) * (255 - 0) / (nsn_output.max() - nsn_output.min())) + 0
    nsn_output_norm = nsn_output_norm.astype(np.uint8)
    nsn_output_norm = nsn_output_norm[slice]
    ndn_output_norm = ((ndn_output - ndn_output.min()) * (255 - 0) / (ndn_output.max() - ndn_output.min())) + 0
    ndn_output_norm = ndn_output_norm.astype(np.uint8)
    ndn_output_norm = ndn_output_norm[slice]

    #threshold creates binary image 0 if cell, 255 if background
    ret, thresh = cv.threshold(nsn_output_norm,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 1)
    sure_bg = cv.dilate(opening,kernel,iterations=1)
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    #convert nsn and ndn to color
    nsn_output_norm = cv.cvtColor(nsn_output_norm, cv.COLOR_GRAY2BGR)
    ndn_output_norm = cv.cvtColor(ndn_output_norm, cv.COLOR_GRAY2BGR)
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv.watershed(nsn_output_norm,markers)
    nsn_output_norm[markers == -1] = [255,0,0]
    ndn_output_norm[markers == -1] = [255,0,0]
    return nsn_output_norm, ndn_output_norm

#function to plot 3d color image
def plot_image_3D(image, slice_index=0):
    if len(image.shape) == 4:
        image = image.squeeze(0)
    def func(slice_index):
        #returns slics of image
        return image[int(slice_index)]
    n_ind = image.shape[0]
    control = iplt.imshow(func, slice_index=(0, n_ind-1))
    plt.show()
    

if __name__ == "__main__":
    dataset = CellDataset(images_dir=images_dir, masks_dir=ground_truth_dir)

    image, mask = dataset[5]

    nsn_output = nsn_model.forward(image.unsqueeze(0)).squeeze(0)
    ndn_output = ndn_model.forward(image.unsqueeze(0)).squeeze(0)
    ndn_output = ndn_output.squeeze(0)
    nsn_output = nsn_output.squeeze(0)
    #convert to numpy array
    ndn_output = ndn_output.detach().numpy()
    nsn_output = nsn_output.detach().numpy()
    
    # Initialize nsn_watershed and ndn_watershed as 3D arrays
    nsn_watershed = np.zeros((*nsn_output.shape, 3), dtype=np.uint8)
    ndn_watershed = np.zeros((*ndn_output.shape, 3), dtype=np.uint8)

    print(nsn_watershed.shape)
    print(ndn_watershed.shape)

    for slice in range(nsn_output.shape[0]):
        nsn_watershed[slice], ndn_watershed[slice] = run_watershed(nsn_output, ndn_output, slice)

    ##save nsn_watershed and ndn_watershed as tiff files in folder called "watershed_images"
    #make directory if it does not exist
    if not os.path.exists("watershed_images"):
        os.makedirs("watershed_images")
    tifffile.imsave("watershed_images/nsn_watershed.tif", nsn_watershed)
    tifffile.imsave("watershed_images/ndn_watershed.tif", ndn_watershed)


    
    """
    slice = 25
    #print(run_watershed(nsn_output, ndn_output))
    img_watershed, imgwatershedndn = run_watershed(nsn_output, ndn_output, slice)
    padding = 32
    img_watershed = img_watershed[padding:-padding, padding:-padding]
    imgwatershedndn = imgwatershedndn[padding:-padding, padding:-padding]
    
    

    #plot img_watershed and nsn_output and ndn_outpu in subplot
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(nsn_output[slice], cmap='gray')
    axs[0].set_title("NSN output")
    axs[1].imshow(imgwatershedndn, cmap='gray')
    axs[1].set_title("NDN output")
    axs[2].imshow(img_watershed, cmap='gray')
    axs[2].set_title("Watershed output")
    plt.show()"""
    

    """
    print("initial image")
    dataset.print_image_3D(image)
    print("mask")
    dataset.print_image_3D(mask)
    print("ndn output")
    dataset.print_image_3D(ndn_output)
    print("nsn output")
    dataset.print_image_3D(nsn_output)
    """