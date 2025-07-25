import torch
import logging
import time

def validate(model, val_loader, criterion, class_names, num_classes, device, start_time, epoch=None, total_epochs=None):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    val_batches = len(val_loader)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

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

    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            logging.info(f"Class {i} ({class_names[i]}): Accuracy {acc:.2f}%")

    return avg_val_loss, val_accuracy
