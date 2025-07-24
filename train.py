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
import platform
import csv

# Main settings
NUM_CLASSES = 101
BATCH_SIZE = 32
EPOCHS = 5

# Paths and settings
DATA_DIR = "data/food-101/images_split"
MODEL_DIR = f"models/checkpoints/e{EPOCHS}_b{BATCH_SIZE}"
MODEL_PATH = os.path.join(MODEL_DIR, "food101_resnet.pth")
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"training_{timestamp}.log")

CHECKPOINT_SUBDIR = os.path.join(MODEL_DIR, "checkpoints")
os.makedirs(CHECKPOINT_SUBDIR, exist_ok=True)

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
    return train_loader, val_loader, val_set.classes

# Train model
def train_model():
    start_time = time.time()
    local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Training started at local time: {local_time}\n")
    logging.info(f"Python: {platform.python_version()}, Torch: {torch.__version__}, Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logging.info(f"Config | Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: 0.001")

    train_loader, val_loader, class_names = load_data()
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    summary_file = os.path.join(LOG_DIR, f"summary_{timestamp}.csv")
    with open(summary_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])  # CSV header

    # Resume from latest checkpoint if exists
    start_epoch = 0
    checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_SUBDIR) if f.endswith(".pth")])
    if checkpoint_files:
        latest_checkpoint = os.path.join(CHECKPOINT_SUBDIR, checkpoint_files[-1])
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
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            print(f"Epoch {epoch+1}, Batch {i+1}/{total_batches} - {percent_complete:.1f}% complete | Time Elapsed: {mins}m {secs}s", end='\r')

            if (i + 1) % 100 == 0:
                logging.info(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / total_batches
        logging.info(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {avg_loss:.4f}")

        checkpoint_path = os.path.join(CHECKPOINT_SUBDIR, f"checkpoint_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_size = os.path.getsize(checkpoint_path) / 1024**2
        logging.info(f"Checkpoint saved: {checkpoint_path} ({checkpoint_size:.2f} MB)")

        model.eval()
        correct = 0
        total = 0
        val_batches = len(val_loader)
        val_loss = 0.0
        class_correct = [0] * NUM_CLASSES
        class_total = [0] * NUM_CLASSES
        with torch.no_grad():
            for j, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for idx in range(len(labels)):
                    label = labels[idx]
                    class_total[label] += 1
                    class_correct[label] += (predicted[idx] == label).item()

                val_percent = (j + 1) / val_batches * 100
                elapsed = time.time() - start_time
                mins, secs = divmod(int(elapsed), 60)
                print(f"Validation progress: {val_percent:.1f}% complete | Time Elapsed: {mins}m {secs}s", end='\r')

        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / val_batches
        logging.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")

        with open(summary_file, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch+1, avg_loss, avg_val_loss, val_accuracy])

        # Per-class accuracy logging
        for i in range(NUM_CLASSES):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                logging.info(f"Class {i} ({class_names[i]}): Accuracy {acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    logging.info(f"Final model saved to {MODEL_PATH}")

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    logging.info(f"Training completed in {minutes}m {seconds}s")

if __name__ == '__main__':
    train_model()
