import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from unet_3d import UNet3D, DiceLoss
from cell_dataset import CellDataset
from torch.utils.data.dataset import random_split

if __name__ == "__main__":
    images_dir = os.path.join("data", "images", "train", "Images")
    ground_truth_dir = os.path.join("data", "GroundTruth", "train", "GroundTruth_NSN")

    dataset = CellDataset(images_dir=images_dir, masks_dir=ground_truth_dir)

    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=False)

    n_channels = 1
    n_classes = 2
    model = UNet3D(n_channels, n_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = DiceLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        print("epoch: " + str(epoch))

        model.train()
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            print(loss)
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_dataloader)
        print(f'Train - Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for inputs, labels in eval_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                eval_loss += loss.item()
        eval_loss /= len(eval_dataloader)
        print(f'Eval - Epoch {epoch + 1}/{num_epochs}, Loss: {eval_loss:.4f}')

    print('Finished Training')

