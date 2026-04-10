import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """Grad-CAM for ResNet backbone in HybridModel."""
   
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()
   
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
       
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
       
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
   
    def generate(self, image, hand_feat, target_class=None):
        """
        image: (1, 3, H, W) tensor
        hand_feat: (1, 14) tensor
        target_class: int or None (use predicted class)
        """
        self.model.eval()
        image.requires_grad_(True)
       
        # Forward
        cnn_feat = self.model.features(image)
        logits = self.model.fusion(cnn_feat.flatten(1), hand_feat)
       
        if target_class is None:
            target_class = logits.argmax(1).item()
       
        # Backward
        self.model.zero_grad()
        logits[0, target_class].backward()
       
        # Grad-CAM (resize with PyTorch, no cv2.resize)
        weights = self.gradients.mean(dim=(2, 3))
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam.squeeze().unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear')
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

def overlay_heatmap(img, heatmap, alpha=0.5):
    """
    img: numpy (H, W, 3) 0-255
    heatmap: numpy (H, W) 0-1
    """
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay
