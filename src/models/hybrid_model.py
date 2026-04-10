import torch
import torch.nn as nn
from torchvision import models

from .attention_fusion import AttentionFusion

class HybridModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()
        backbone=models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.features =nn.Sequential(*list(backbone.children())[:-1])
        self.fusion = AttentionFusion(
            cnn_dim=512, hand_dim=14, hidden_dim=256,
            num_classes=num_classes, droput=dropout
        )

    def forward(self, images, hand_feat):
        cnn_feat=self.features(images)
        cnn_feat=cnn_feat.flatten(1)
        return self.fusion(cnn_feat, hand_feat)