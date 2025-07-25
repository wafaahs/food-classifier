import time
import logging
import torch

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs, start_time, current_epoch_start_time):
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
        elapsed_current_epoch = time.time() - current_epoch_start_time
        mins_ce, secs_ce = divmod(int(elapsed_current_epoch), 60)
        print(f"Epoch {epoch+1}/{total_epochs}, Batch {i+1}/{total_batches} "
              f"- {percent_complete:.1f}% complete | Current E Time Elapsed: {mins_ce}m {secs_ce}s | Total Time Elapsed: {mins}m {secs}s", end='\r')

        if (i + 1) % 100 == 0:
            logging.info(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")

    avg_loss = running_loss / total_batches
    logging.info(f"Epoch {epoch+1}/{total_epochs}, Training Loss: {avg_loss:.4f}")
    return avg_loss
