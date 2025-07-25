import argparse

def get_config():
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
    return parser.parse_args()
