import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    """
    Fuses CNN features (512) and handcrafted features (14) using learned attention.
    Attention weights determine how much each modality contributes per sample.
    """
    def __init__(self, cnn_dim=512, hand_dim=14, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, hidden_dim)
        self.hand_proj = nn.Linear(hand_dim, hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(cnn_dim + hand_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, cnn_feat, hand_feat):
        """
        Args:
            cnn_feat: (B, 512) from ResNet backbone
            hand_feat: (B, 14) handcrafted features
        Returns:
            logits: (B, num_classes)
        """
        combined = torch.cat([cnn_feat, hand_feat], dim=1)
        attn = self.attention(combined)
        cnn_proj = self.cnn_proj(cnn_feat)
        hand_proj = self.hand_proj(hand_feat)
        fused = attn[:, 0:1] * cnn_proj + attn[:, 1:2] * hand_proj
        return self.classifier(fused)