import os
import logging
import csv

def setup_logging(log_dir, timestamp):
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    summary_csv = os.path.join(log_dir, f"summary_{timestamp}.csv")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ])

    with open(summary_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

    return log_file, summary_csv
