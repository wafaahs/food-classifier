import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_model(num_classes):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
