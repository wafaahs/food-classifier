import os
import time
import argparse
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


# Argument parser for configurable parameters
parser = argparse.ArgumentParser(
    description="Train a ResNet18 model on food images with configurable hyperparameters.\n\n"
                "Example usage:\n"
                "  python train.py --epochs 10 --batch_size 64 --lr 0.0005 --optimizer sgd\n",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("--num_classes", type=int, default=101, help="Number of output classes (default: 101)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Optimizer to use (default: adam)")
args = parser.parse_args()

# Create timestamp and directories
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR = "data/food-101/images_split"
FINAL_MODEL_DIR = "models/final"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

# File naming setup
NUM_CLASSES = args.num_classes
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.lr
OPTIMIZER_NAME = args.optimizer
LR_TAG = str(LEARNING_RATE).replace('.', 'p')

MODEL_DIR = f"models/checkpoints/e{EPOCHS}_b{BATCH_SIZE}_{OPTIMIZER_NAME}_lr{LR_TAG}"
MODEL_PATH = os.path.join(FINAL_MODEL_DIR, f"e{EPOCHS}_b{BATCH_SIZE}_{OPTIMIZER_NAME}_lr{LR_TAG}_resnet.pth")
CHECKPOINT_SUBDIR = MODEL_DIR
os.makedirs(CHECKPOINT_SUBDIR, exist_ok=True)

# Setup logging
LOG_FILE = os.path.join(LOG_DIR, f"training_{timestamp}.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
    logging.FileHandler(LOG_FILE),
    logging.StreamHandler()
])

# Create CSV summary file
SUMMARY_CSV = os.path.join(LOG_DIR, f"summary_{timestamp}.csv")
with open(SUMMARY_CSV, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_data():
    train_set = ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    val_set = ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    return train_loader, val_loader, val_set.classes

def train_model():
    start_time = time.time()
    local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Training started at local time: {local_time}")
    logging.info(f"Python: {platform.python_version()}, Torch: {torch.__version__}, Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logging.info(f"Config | Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}, Optimizer: {OPTIMIZER_NAME}")

    train_loader, val_loader, class_names = load_data()
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()
    if OPTIMIZER_NAME == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Resume from latest checkpoint if available
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

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_SUBDIR, f"checkpoint_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_size = os.path.getsize(checkpoint_path) / 1024**2
        logging.info(f"Checkpoint saved: {checkpoint_path} ({checkpoint_size:.2f} MB)")

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        val_batches = len(val_loader)
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
        logging.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Write to CSV
        with open(SUMMARY_CSV, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch+1, avg_loss, avg_val_loss, val_accuracy])

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
