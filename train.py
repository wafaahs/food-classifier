import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import platform
import csv

from config import get_config
from utils.logger import setup_logging
from utils.model_builder import build_model
from utils.data_loader import load_data
from utils.validator import validate
from utils.trainer import train_one_epoch

# Load configuration
args = get_config()
NUM_CLASSES = args.num_classes
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.lr
OPTIMIZER_NAME = args.optimizer

# Timestamp and directories
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR = "data/food-101/images_split"
FINAL_MODEL_DIR = "models/final"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

# Path tags
LR_TAG = str(LEARNING_RATE).replace('.', 'p')
MODEL_DIR = f"models/checkpoints/e{EPOCHS}_b{BATCH_SIZE}_{OPTIMIZER_NAME}_lr{LR_TAG}"
MODEL_PATH = os.path.join(FINAL_MODEL_DIR, f"e{EPOCHS}_b{BATCH_SIZE}_{OPTIMIZER_NAME}_lr{LR_TAG}_resnet.pth")
CHECKPOINT_SUBDIR = MODEL_DIR
os.makedirs(CHECKPOINT_SUBDIR, exist_ok=True)

# Setup logging and summary file
LOG_FILE, SUMMARY_CSV = setup_logging(LOG_DIR, timestamp)

def train_model():
    start_time = time.time()
    local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Training started at local time: {local_time}")
    logging.info(f"Python: {platform.python_version()}, Torch: {torch.__version__}, Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logging.info(f"Config | Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}, Optimizer: {OPTIMIZER_NAME}")

    # Load data and model
    train_loader, val_loader, class_names = load_data(DATA_DIR, BATCH_SIZE)
    model = build_model(NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()
    if OPTIMIZER_NAME == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Resume from checkpoint if available
    start_epoch = 0
    checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_SUBDIR) if f.endswith(".pth")])
    if checkpoint_files:
        latest_checkpoint = os.path.join(CHECKPOINT_SUBDIR, checkpoint_files[-1])
        model.load_state_dict(torch.load(latest_checkpoint))
        start_epoch = int(checkpoint_files[-1].split("_")[1].split(".")[0])
        logging.info(f"Resumed from checkpoint: {latest_checkpoint}, starting at epoch {start_epoch + 1}")

    for epoch in range(start_epoch, EPOCHS):
        current_epoch_start_time = time.time()
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, EPOCHS, start_time, current_epoch_start_time
        )

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_SUBDIR, f"checkpoint_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_size = os.path.getsize(checkpoint_path) / 1024**2
        logging.info(f"Checkpoint saved: {checkpoint_path} ({checkpoint_size:.2f} MB)")

        # Validation
        avg_val_loss, val_accuracy = validate(
            model, val_loader, criterion, class_names,
            NUM_CLASSES, device, start_time
        )

        # Save epoch summary
        with open(SUMMARY_CSV, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch + 1, avg_loss, avg_val_loss, val_accuracy])

        elapsed_time_ce = time.time() - current_epoch_start_time
        minutes, seconds = divmod(int(elapsed_time_ce), 60)
        logging.info(f"Current Epoch completed in {minutes}m {seconds}s")

    torch.save(model.state_dict(), MODEL_PATH)
    logging.info(f"Final model saved to {MODEL_PATH}")

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    logging.info(f"Training completed in {minutes}m {seconds}s")

if __name__ == '__main__':
    train_model()
