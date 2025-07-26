import os
import logging
from datetime import datetime
import csv

def setup_logging(log_dir, timestamp):
    session_dir = os.path.join(log_dir, f"log_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    log_file = os.path.join(session_dir, "training.log")
    summary_csv = os.path.join(session_dir, "summary.csv")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    with open(summary_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Validation Accuracy"])

    return log_file, summary_csv

def get_logger(name, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(output_dir, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
