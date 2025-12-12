"""
PyTorch model loader for Colab-trained EfficientNetB0 models.

Usage:
    The model will be automatically loaded from model/efficientnet_b0_best.pth
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models, transforms

# 7 skin cancer classes (must match training)
CLASS_NAMES = [
    "Melanoma",
    "Melanocytic_Nevus", 
    "Basal_Cell_Carcinoma",
    "Actinic_Keratosis",
    "Benign_Keratosis",
    "Dermatofibroma",
    "Vascular_Lesion"
]

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "efficientnet_b0_best.pth"


class EfficientNetB0Classifier(nn.Module):
    """EfficientNetB0 model for 7-class skin cancer classification."""
    
    def __init__(self, num_classes=7):
        super().__init__()
        # Load EfficientNetB0 without wrapper to match saved model structure
        self.features = models.efficientnet_b0(weights=None).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Get the number of features from the original classifier
        base_model = models.efficientnet_b0(weights=None)
        num_features = base_model.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_local_model(model_path: Optional[Path] = None, device: str = "cpu"):
    """
    Load a PyTorch model from disk.
    
    Args:
        model_path: Path to .pth model file (default: model/efficientnet_b0_best.pth)
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded PyTorch model in eval mode, or None if file doesn't exist
    """
    path = Path(model_path or DEFAULT_MODEL_PATH)
    if not path.exists():
        print(f"Model file not found: {path}")
        return None
    
    try:
        # Initialize model architecture
        model = EfficientNetB0Classifier(num_classes=len(CLASS_NAMES))
        
        # Load state dict
        checkpoint = torch.load(path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()  # Set to evaluation mode
        model.to(device)
        
        print(f"✅ Model loaded successfully from: {path}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def get_preprocessing_transform():
    """
    Get the image preprocessing transform for EfficientNetB0.
    Matches ImageNet normalization used during training.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
