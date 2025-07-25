import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def load_data(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_set = ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_set = ImageFolder(os.path.join(data_dir, "test"), transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return train_loader, val_loader, val_set.classes
