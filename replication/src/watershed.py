from cell_dataset import CellDataset
import os, torch
from unet_3d import NSN, NDN
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import morphology, distance_transform_edt
from skimage.segmentation import watershed
from scipy.ndimage import label
from skimage.segmentation import find_boundaries
from skimage import img_as_ubyte
from skimage.color import gray2rgb


def run_watershed_3d(nsn_output, ndn_output):
    # Convert binary images to boolean
    nsn_output = nsn_output.astype(bool)
    ndn_output = ndn_output.astype(bool)

    # Apply 3D morphological operations
    kernel = np.ones((3,3,3))
    opening = morphology.binary_opening(nsn_output, kernel)
    sure_bg = morphology.binary_dilation(opening, kernel)
    dist_transform = distance_transform_edt(opening)
    sure_fg = dist_transform > 0.9 * dist_transform.max()
    unknown = sure_bg & ~sure_fg

    # Marker labelling
    markers, _ = label(sure_fg)  # Only keep the first item returned by label

    markers[unknown] = 0

    # Apply 3D watershed
    markers = watershed(-dist_transform, markers, mask=sure_bg)

    return markers

##PLOTTING FUNCTIONS##

def plot_2D_markers(nsn_output, ndn_output, mask, markers, slice=25):

    cmap = plt.cm.get_cmap('nipy_spectral', markers.max() + 1)
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(nsn_output[slice], cmap='gray')
    axs[0].set_title("NSN output")
    axs[1].imshow(ndn_output[slice], cmap='gray')
    axs[1].set_title("NDN output")
    axs[2].imshow(markers[slice], cmap=cmap)
    axs[2].set_title("Watershed output")
    axs[3].imshow(mask[slice])
    axs[3].set_title("Ground truth")
    plt.show()

def plot_3D_markers(marker_data, mask_data, sampling_rate=1):
    """
    Plots a 3D representation of an NSN watershed image with corresponding colors.
    
    Parameters:
    - marker_data: A 3D numpy array with shape [Z, X, Y], representing the 3D marker image.
    - mask_data: A 3D numpy array with shape [Z, X, Y], representing the 3D mask image.
    - sampling_rate: An integer to downsample the data for visualization. Higher values result in fewer points plotted.
    """
    
    # Ensure the data has the correct dimensions
    if marker_data.ndim != 3 or mask_data.ndim != 3:
        raise ValueError("Data must be a 3D numpy array.")
    
    fig = plt.figure(figsize=(20, 10))
    titles = ['Marker Image', 'Mask Image']

    for i, (data, title) in enumerate(zip([marker_data, mask_data], titles), start=1):
        # Extract the coordinates where the voxel should be plotted based on the sampling rate
        z, x, y = np.where(data > 0)
        z, x, y = z[::sampling_rate], x[::sampling_rate], y[::sampling_rate]
        
        # Extract corresponding colors for each voxel
        unique_markers = np.unique(data)
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_markers)))
        color_dict = {marker: color for marker, color in zip(unique_markers, colors)}
        voxel_colors = [color_dict[marker] for marker in data[z, x, y]]
        
        # Create a 3D plot
        ax = fig.add_subplot(1, 2, i, projection='3d')
        ax.set_title(title)  # Set the title for the subplot
        
        # Scatter plot, each point with its corresponding color
        ax.scatter(x, y, z, c=voxel_colors, marker='o')
        
        # Set labels
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    
    # Show the plot
    plt.savefig("watershed_images/3D_watershed_output.png")
    plt.show()
    #save image
    

    

if __name__ == "__main__":
    ##LOAD TRAINED NSN AND NDN MODELS##
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
    
    ##LOAD TEST IMAGES AND GROUND TRUTH FOR QCANet##
    dataset = CellDataset(images_dir=images_dir, masks_dir=ground_truth_dir)

    image, mask = dataset[5]

    nsn_output = nsn_model.forward(image.unsqueeze(0)).squeeze(0)
    ndn_output = ndn_model.forward(image.unsqueeze(0)).squeeze(0)

    ndn_output = ndn_output.squeeze(0)
    nsn_output = nsn_output.squeeze(0)
    mask = mask.squeeze(0)

    #convert raw output data values to probability between 0 and 1
    nsn_output = torch.sigmoid(nsn_output)
    ndn_output = torch.sigmoid(ndn_output)

    #convert the outputs to binary (0 for background, 1 for cell/cell center) 
    nsn_output = (nsn_output > 0.5).detach().numpy().astype(np.uint8)
    ndn_output = (ndn_output > 0.5).detach().numpy().astype(np.uint8)
    mask = mask.detach().numpy()
    
    #remove mirror padding
    mask = mask[:, 32:-32, 32:-32]
    nsn_output = nsn_output[:, 32:-32, 32:-32]
    ndn_output = ndn_output[:, 32:-32, 32:-32]
    

    ##RUN WATERSHED AND PLOT/SAVE RESULTS##

    markers = run_watershed_3d(nsn_output, ndn_output)
    
#   #save markers to tiff file
#   #make directory if it does not exist
#   if not os.path.exists("watershed_images"):
#       os.makedirs("watershed_images")
#   tifffile.imsave("watershed_images/watershed_output_3D.tif", markers)

    #plot 3D markers
    plot_3D_markers(markers, mask)

    slice = 25
    plot_2D_markers(nsn_output, ndn_output, mask, markers, slice=slice)