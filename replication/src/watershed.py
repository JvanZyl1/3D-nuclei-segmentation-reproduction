from cell_dataset import CellDataset
from metrics import Metrics, IoU, SEG, MuCov
import os, torch
from unet_3d import NSN, NDN
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import label, binary_opening, binary_dilation
from skimage.segmentation import watershed

def run_watershed_3d(nsn_output, ndn_output):
    # Convert to bool for morphology operations
    nsn_output = nsn_output.astype(bool)
    ndn_output = ndn_output.astype(bool)

    kernel = np.ones((3,3,3))
    #remove small unwqnted noise
    opening = binary_opening(nsn_output, kernel)
    sure_bg = binary_dilation(opening, kernel)

    #convert ndn_output so that each cell center has different int value
    markers, _ = label(ndn_output)

    markers = watershed(np.logical_not(nsn_output), markers, mask=sure_bg)

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
    plt.savefig("watershed_images/2D_slice.png")
    plt.show()

def plot_3D_markers(data, mask_data, sampling_rate=1, data_type='Watershed'):
    #sampling rate used to reduce points plotted

    #check dimesnions
    if data.ndim != 3 or mask_data.ndim != 3:
        raise ValueError("Data must be a 3D numpy array.")
    
    fig = plt.figure(figsize=(20, 10))
    titles = [data_type + ' Image', 'Mask Image']

    for i, (data, title) in enumerate(zip([data, mask_data], titles), start=1):
        z, x, y = np.where(data > 0)
        z, x, y = z[::sampling_rate], x[::sampling_rate], y[::sampling_rate]
        unique_markers = np.unique(data)
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_markers)))
        color_dict = {marker: color for marker, color in zip(unique_markers, colors)}
        voxel_colors = [color_dict[marker] for marker in data[z, x, y]]

        ax = fig.add_subplot(1, 2, i, projection='3d')
        ax.set_title(title)
        
        ax.scatter(x, y, z, c=voxel_colors, marker='o')
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    
    path = f"watershed_images/3D_{data_type}_output.png"
    plt.savefig(path)
    plt.show()
    

    

if __name__ == "__main__":
    ##LOAD TRAINED NSN AND NDN MODELS##
    n_channels = 1
    ndn_model = NDN(n_channels=n_channels) 
    nsn_model = NSN(n_channels=n_channels)


    #note: model trained on CUDA, use map_location input to load model on CPU
    ndn_model.load_state_dict(torch.load("replication/src/ndn (1).pth", map_location=torch.device('cpu')))
    nsn_model.load_state_dict(torch.load("replication/src/nsn.pth", map_location=torch.device('cpu')))
    #get test images and test ground truth (QCANet)
    images_dir = os.path.join("data", "images", "test", "Images")
    ground_truth_dir = os.path.join("data", "GroundTruth", "test",
                                        "GroundTruth_QCANet")
    
    ##LOAD TEST IMAGES AND GROUND TRUTH FOR QCANet##
    dataset = CellDataset(images_dir=images_dir, masks_dir=ground_truth_dir)
    #loop over all images in the test set
    iou_sum = 0
    seg_sum = 0
    mucov_sum = 0
    for i in range(len(dataset)):

        image, mask = dataset[i]

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
        
        plot = False
        if plot:
            plot_3D_markers(markers, mask, data_type='Watershed')

            #plot 3D nsn and ndn output 
            plot_3D_markers(nsn_output, mask, data_type='NSN')
            plot_3D_markers(ndn_output, mask, data_type='NDN')

            slice = 25
            plot_2D_markers(nsn_output, ndn_output, mask, markers, slice=slice)


        metrics = Metrics(IoU, SEG, MuCov)
        iou = metrics.iou(torch.tensor(markers), torch.tensor(mask))
        seg = metrics.seg(torch.tensor(markers), torch.tensor(mask))
        mucov = metrics.mucov(torch.tensor(markers), torch.tensor(mask))

        iou_sum += iou
        seg_sum += seg
        mucov_sum += mucov
    
    print(f"Average IoU: {iou_sum/len(dataset)}")
    print(f"Average SEG: {seg_sum/len(dataset)}")
    print(f"Average MuCov: {mucov_sum/len(dataset)}")