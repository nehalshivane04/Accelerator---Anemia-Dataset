import torch 
import torch.nn as nn
from torchvision import models

def get_cnn_model(num_classes=2, pretrained=True):
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    model.fc = nn.Linear(512, num_classes)
    return model