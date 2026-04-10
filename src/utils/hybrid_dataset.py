import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class HybridDataset(Dataset):
    """Dataset that returns (image, handcrafted_features, label) for hybrid model."""
    def __init__(self, image_paths, labels, handcrafted_features, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.hand_feat = handcrafted_features  # (N, 14) numpy array
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        feat = torch.tensor(self.hand_feat[idx], dtype=torch.float32)
        return img, feat, self.labels[idx]
