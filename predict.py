import torch
import argparse
import os
from utils.model_builder import build_model
from utils.predict_utils import load_image, predict, save_predictions_csv
from utils.logger import get_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict food class from image or folder")
    parser.add_argument("--input", required=True, help="Path to input image or folder")
    parser.add_argument("--model", required=True, help="Path to trained model .pth file")
    parser.add_argument("--output", required=True, help="Path to the output root directory")
    args = parser.parse_args()

    logger = get_logger("predict", args.output)
    logger.info("Starting prediction...")

    # Load class names from food-101 metadata directory
    class_file = os.path.join("data", "food-101", "meta", "classes.txt")
    if not os.path.exists(class_file):
        logger.error(f"Expected class names at {class_file}")
        raise FileNotFoundError(f"Expected class names at {class_file}")

    with open(class_file, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    model.to(device)

    logger.info(f"Model loaded: {args.model}")
    predictions = []
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    output_dir = os.path.join(args.output, model_name)

    if os.path.isfile(args.input):
        true_label = os.path.basename(os.path.dirname(args.input))
        pred = predict(args.input, model, class_names, device)
        predictions.append([args.input, pred, true_label])
        logger.info(f"Predicted {pred} for {args.input}")

    elif os.path.isdir(args.input):
        supported_exts = (".jpg", ".jpeg", ".png")
        image_paths = []
        for root, _, files in os.walk(args.input):
            for fname in sorted(files):
                if fname.lower().endswith(supported_exts):
                    image_paths.append(os.path.join(root, fname))

        total = len(image_paths)
        for idx, fpath in enumerate(image_paths):
            true_label = os.path.basename(os.path.dirname(fpath))
            pred = predict(fpath, model, class_names, device)
            predictions.append([fpath, pred, true_label])
            if idx % 50 == 0 or idx == total - 1:
                logger.info(f"[{idx+1}/{total}] {fpath} -> {pred}")
    else:
        logger.error("--input must be a valid image file or directory")
        raise ValueError("--input must be a valid image file or directory")

    save_predictions_csv(output_dir, predictions)
    logger.info(f"Saved predictions to {os.path.join(output_dir, 'predictions.csv')}")
