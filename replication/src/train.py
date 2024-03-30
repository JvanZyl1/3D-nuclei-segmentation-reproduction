import os, torch, torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt

from unet_3d import NSN, NDN
from metrics import DiceLoss
from cell_dataset import CellDataset


def run_training_loop(images_dir, ground_truth_dir, criterion, optimizer, num_epochs, model):
    dataset = CellDataset(images_dir=images_dir, masks_dir=ground_truth_dir)

    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loss_per_epoch, val_loss_per_epoch = [], []

    for epoch in range(1, num_epochs + 1):
        print("epoch: " + str(epoch))

        model.train()
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if epoch == 50:
                plot_everything(inputs, labels, outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            print(loss)
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_dataloader)
        train_loss_per_epoch.append(epoch_loss)
        print(f'Train - Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    print('Finished Training')

    return train_loss_per_epoch, val_loss_per_epoch

def plot_everything(input_images, labels, predictions):
    input_image = input_images[0][0][24].detach().numpy()
    label = labels[0][0][24].detach().numpy()
    prediction = predictions[0][0][24].detach().numpy()

    fig, axs = plt.subplots(3)
    axs[0].imshow(input_image * 255)
    axs[1].imshow(label * 255)
    axs[2].imshow(prediction * 255)
    plt.show()

def train_ndn():
    images_dir = os.path.join("data", "images", "train", "Images")
    ground_truth_dir = os.path.join("data", "GroundTruth", "train",
                                    "GroundTruth_NDN")  # ndn is getting downsampeld too much from kernel size = 5

    n_channels = 1
    model = NDN(n_channels)

    learning_rate = 0.001
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 10

    run_training_loop(images_dir, ground_truth_dir, criterion, optimizer, num_epochs, model)


def train_nsn():
    images_dir = os.path.join("data", "images", "train", "Images")
    ground_truth_dir = os.path.join("data", "GroundTruth", "train", "GroundTruth_NSN")

    n_channels = 1
    model = NSN(n_channels)

    learning_rate = 0.001
    criterion = DiceLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    num_epochs = 10

    return run_training_loop(images_dir, ground_truth_dir, criterion, optimizer, num_epochs, model)

def plot_train_val_loss(train_loss, val_loss):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    train_loss, val_loss = train_nsn()
