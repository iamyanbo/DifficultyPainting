"""
Difficulty Predictor Model

A CNN that predicts 3D detection difficulty from 2D image patches.
This is the core of the Learned Detection Difficulty pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DifficultyPredictor(nn.Module):
    """
    CNN to predict detection difficulty from image patch.
    
    Input: 64x64 RGB image patch centered on object
    Output: Scalar difficulty score in [0, 1]
    
    Architecture: Simple VGG-style CNN for efficiency
    """
    
    def __init__(self, input_size=64):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 64 -> 32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 16 -> 8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4: 8 -> 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x, aux_features=None):
        """
        Args:
            x: (B, 3, 64, 64) RGB image patches
            aux_features: Optional, not used in V1
        Returns:
            difficulty: (B, 1) difficulty scores
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class DifficultyPredictorV2(nn.Module):
    """
    Enhanced difficulty predictor with auxiliary features.
    
    Input: 
        - 64x64 RGB image patch
        - Auxiliary features: [depth, bbox_width, bbox_height, truncation, occlusion]
    Output: Scalar difficulty score in [0, 1]
    
    The auxiliary features provide crucial context about 3D detectability
    that cannot be inferred from the 2D patch alone.
    """
    
    def __init__(self, input_size=64, num_aux_features=5):
        super().__init__()
        self.num_aux_features = num_aux_features
        
        # Image feature extractor (same as V1)
        self.features = nn.Sequential(
            # Block 1: 64 -> 32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 16 -> 8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4: 8 -> 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4)
        )
        
        # Auxiliary feature encoder
        self.aux_encoder = nn.Sequential(
            nn.Linear(num_aux_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )
        
        # Combined classifier (image features + aux features)
        # Image: 256*4*4 = 4096, Aux: 64, Total: 4160
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, aux_features):
        """
        Args:
            x: (B, 3, 64, 64) RGB image patches
            aux_features: (B, 5) [depth, bbox_w, bbox_h, truncation, occlusion]
        Returns:
            difficulty: (B, 1) difficulty scores
        """
        # Extract image features
        img_feats = self.features(x)
        img_feats = img_feats.view(img_feats.size(0), -1)  # Flatten
        
        # Encode auxiliary features
        aux_feats = self.aux_encoder(aux_features)
        
        # Concatenate and classify
        combined = torch.cat([img_feats, aux_feats], dim=1)
        out = self.classifier(combined)
        
        return out


class DifficultyPredictorV3(nn.Module):
    """
    Difficulty predictor using pretrained ResNet-18 backbone.
    
    Uses ImageNet-pretrained ResNet-18 for feature extraction,
    which provides better low-level features than training from scratch.
    
    Input: 
        - 64x64 RGB image patch (will be resized to 224x224 for ResNet)
        - Auxiliary features: [depth, bbox_width, bbox_height, truncation, occlusion]
    Output: Scalar difficulty score in [0, 1]
    """
    
    def __init__(self, num_aux_features=5, freeze_backbone=False):
        super().__init__()
        self.num_aux_features = num_aux_features
        
        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=True)
        
        # Remove the final FC layer, keep up to avgpool
        # ResNet-18 outputs 512-dim features after avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Optionally freeze backbone for faster training
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Auxiliary feature encoder
        self.aux_encoder = nn.Sequential(
            nn.Linear(num_aux_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )
        
        # Combined classifier (ResNet features + aux features)
        # ResNet-18: 512, Aux: 64, Total: 576
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Resize transform for ResNet (expects 224x224)
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
    
    def forward(self, x, aux_features):
        """
        Args:
            x: (B, 3, 64, 64) RGB image patches
            aux_features: (B, 5) [depth, bbox_w, bbox_h, truncation, occlusion]
        Returns:
            difficulty: (B, 1) difficulty scores
        """
        # Resize to 224x224 for ResNet
        x = self.resize(x)
        
        # Extract features with ResNet
        img_feats = self.backbone(x)
        img_feats = img_feats.view(img_feats.size(0), -1)  # (B, 512)
        
        # Encode auxiliary features
        aux_feats = self.aux_encoder(aux_features)
        
        # Concatenate and classify
        combined = torch.cat([img_feats, aux_feats], dim=1)
        out = self.classifier(combined)
        
        return out


class DifficultyPredictorV4(nn.Module):
    """
    Multi-output difficulty predictor using pretrained ResNet-18.
    
    Predicts 3 outputs simultaneously using multi-task learning:
    - Difficulty (1 - IoU with ground truth)
    - Truncation (how much of object is outside image)
    - Occlusion (how much of object is hidden)
    
    Multi-task learning helps the model learn richer representations
    that transfer across related prediction tasks.
    
    Input: 
        - 64x64 RGB image patch
        - Auxiliary features: [depth, bbox_width, bbox_height]
        (truncation/occlusion are now outputs, not inputs)
    Output: (difficulty, truncation, occlusion) scores in [0, 1]
    """
    
    def __init__(self, num_aux_features=3, freeze_backbone=False):
        super().__init__()
        self.num_aux_features = num_aux_features
        
        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Auxiliary feature encoder (only depth, bbox_w, bbox_h)
        self.aux_encoder = nn.Sequential(
            nn.Linear(num_aux_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )
        
        # Shared representation layer
        self.shared = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # Task-specific heads
        self.difficulty_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.truncation_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.occlusion_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
    
    def forward(self, x, aux_features):
        """
        Args:
            x: (B, 3, 64, 64) RGB image patches
            aux_features: (B, 3) [depth, bbox_w, bbox_h]
        Returns:
            difficulty: (B, 1)
            truncation: (B, 1) 
            occlusion: (B, 1)
        """
        x = self.resize(x)
        
        img_feats = self.backbone(x)
        img_feats = img_feats.view(img_feats.size(0), -1)
        
        aux_feats = self.aux_encoder(aux_features)
        
        combined = torch.cat([img_feats, aux_feats], dim=1)
        shared = self.shared(combined)
        
        difficulty = self.difficulty_head(shared)
        truncation = self.truncation_head(shared)
        occlusion = self.occlusion_head(shared)
        
        return difficulty, truncation, occlusion
    
    def predict_difficulty(self, x, aux_features):
        """Convenience method to get only difficulty prediction."""
        difficulty, _, _ = self.forward(x, aux_features)
        return difficulty



class DifficultyPredictorFinal(nn.Module):
    """
    Final Difficulty Predictor Model (V5/Winner).
    
    Configuration:
    - Backbone: ResNet-18 (Pretrained on ImageNet)
    - Aux Features: Depth, BBox Width, BBox Height (3 features)
    - Architecture: Concatenation after backbone
    
    This configuration won the ablation study with correlation 0.7526.
    """
    def __init__(self, freeze_backbone=False):
        super().__init__()
        
        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=True)
        # Remove FC layer, keeps 512-dim output
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Resize for ResNet input (224x224)
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        
        # Aux Encoder for 3 features [depth, w, h]
        self.aux_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, aux_features):
        x = self.resize(x)
        
        # Image Features
        img_feats = self.backbone(x)
        img_feats = img_feats.view(img_feats.size(0), -1)
        
        # Aux Features
        aux_feats = self.aux_encoder(aux_features)
        
        # Combine
        combined = torch.cat([img_feats, aux_feats], dim=1)
        out = self.classifier(combined)
        
        return out


class DifficultyPredictorFCN(nn.Module):
    """
    Fully Convolutional version for dense prediction.
    Can generate difficulty heatmaps for full images.
    """
    
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Downsample 4x
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB image
        Returns:
            difficulty_map: (B, 1, H, W) per-pixel difficulty
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    # Test the models
    model = DifficultyPredictor()
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    print(f"DifficultyPredictor: {x.shape} -> {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model_fcn = DifficultyPredictorFCN()
    x = torch.randn(1, 3, 384, 1280)
    out = model_fcn(x)
    print(f"DifficultyPredictorFCN: {x.shape} -> {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_fcn.parameters()):,}")
