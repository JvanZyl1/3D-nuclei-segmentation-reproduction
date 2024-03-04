import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet_3d import UNet3D
from cell_dataset import CellDataset

if __name__ == "__main__":
    images_dir = os.path.join("data", "images", "train", "Images")
    ground_truth_dir = os.path.join("data", "GroundTruth", "train", "GroundTruth_NDN")

    dataset = CellDataset(images_dir=images_dir, masks_dir=ground_truth_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    n_channels = 1  # Number of input channels, e.g., 1 for grayscale
    n_classes = 2  # Number of output classes, e.g., 2 for binary segmentation
    model = UNet3D(n_channels, n_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define a loss function
    criterion = nn.CrossEntropyLoss()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs to train for
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for i, data in enumerate(dataloader):
            # Get inputs and labels from the data loader
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    print('Finished Training')

