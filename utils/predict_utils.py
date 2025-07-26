import os
import csv
from PIL import Image
import torch
from torchvision import transforms

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict(image_path, model, class_names, device):
    input_tensor = load_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

def save_predictions_csv(output_dir, predictions):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "predictions.csv")
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "prediction", "ground_truth"])
        writer.writerows(predictions)
