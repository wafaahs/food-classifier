import os
import time
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
import logging

# Paths and settings
DATA_DIR = "data/food-101/images_split"
MODEL_PATH = "models/food101_resnet.pth"
CHECKPOINT_DIR = "models/checkpoints"
LOG_DIR = "logs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Create log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"training_{timestamp}.log")

NUM_CLASSES = 101
BATCH_SIZE = 32
EPOCHS = 5

# Setup logging to file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
    logging.FileHandler(LOG_FILE),
    logging.StreamHandler()
])

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load datasets
def load_data():
    train_set = ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    val_set = ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    return train_loader, val_loader

# Train model
def train_model():
    start_time = time.time()
    local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Training started at local time: {local_time}\n")

    train_loader, val_loader = load_data()
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Resume from latest checkpoint if exists
    start_epoch = 0
    checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")])
    if checkpoint_files:
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
        model.load_state_dict(torch.load(latest_checkpoint))
        start_epoch = int(checkpoint_files[-1].split("_")[1].split(".")[0])
        logging.info(f"Resumed from checkpoint: {latest_checkpoint}, starting at epoch {start_epoch + 1}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            percent_complete = (i + 1) / total_batches * 100
            print(f"Epoch {epoch+1}, Batch {i+1}/{total_batches} - {percent_complete:.1f}% complete", end='\r')

        avg_loss = running_loss / total_batches
        logging.info(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")

        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logging.info(f"Validation Accuracy: {accuracy:.2f}%\n")



    # Save final model
    torch.save(model.state_dict(), MODEL_PATH)
    logging.info(f"Final model saved to {MODEL_PATH}")

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    logging.info(f"Training completed in {minutes}m {seconds}s")

if __name__ == '__main__':
    train_model()
