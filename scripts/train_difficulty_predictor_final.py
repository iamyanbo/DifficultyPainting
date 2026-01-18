"""
Train Final Difficulty Predictor (V5/Winner)

Configuration:
- Model: DifficultyPredictorFinal
- Inputs: Image (ResNet-18) + Depth + BBox(w,h)
- Training Target: 2D Difficulty (1 - IoU)
- Epochs: 40 (for maximum convergence)

This script trains the configuration that won the ablation study (Corr: 0.7526).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import csv
from pathlib import Path
import argparse
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from difficulty_predictor import DifficultyPredictorFinal


class DifficultyDatasetFinal(Dataset):
    """Dataset for Final Model (Image + Depth + BBox)."""
    
    def __init__(self, csv_path, image_dir, patch_size=64, augment=True,
                 max_depth=70.0, image_width=1242, image_height=375):
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.max_depth = max_depth
        self.image_width = image_width
        self.image_height = image_height
        
        self.samples = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bbox = [
                    float(row['bbox2d_left']),
                    float(row['bbox2d_top']),
                    float(row['bbox2d_right']),
                    float(row['bbox2d_bottom'])
                ]
                
                depth = float(row['loc_z'])
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                
                self.samples.append({
                    'frame_id': row['frame_id'],
                    'bbox': bbox,
                    'difficulty': float(row['difficulty']),
                    'depth': depth,
                    'bbox_w': bbox_w,
                    'bbox_h': bbox_h,
                })
        
        print(f"Loaded {len(self.samples)} samples for final training")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img_path = self.image_dir / f"{sample['frame_id']}.png"
        image = cv2.imread(str(img_path))
        if image is None:
            img_path = self.image_dir / f"{sample['frame_id']}.jpg"
            image = cv2.imread(str(img_path))
        
        if image is None:
            return (torch.zeros(3, self.patch_size, self.patch_size),
                    torch.zeros(3), torch.tensor([0.5]))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bbox = sample['bbox']
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = (bbox[2] - bbox[0]) * 1.2
        h = (bbox[3] - bbox[1]) * 1.2
        
        x1 = max(0, int(cx - w/2))
        y1 = max(0, int(cy - h/2))
        x2 = min(image.shape[1], int(cx + w/2))
        y2 = min(image.shape[0], int(cy + h/2))
        
        patch = image[y1:y2, x1:x2]
        
        if patch.size == 0:
            return (torch.zeros(3, self.patch_size, self.patch_size),
                    torch.zeros(3), torch.tensor([0.5]))
        
        patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        
        if self.augment:
            if np.random.rand() > 0.5:
                patch = np.fliplr(patch).copy()
            alpha = 0.8 + np.random.rand() * 0.4
            beta = -20 + np.random.rand() * 40
            patch = np.clip(patch * alpha + beta, 0, 255).astype(np.uint8)
        
        patch = patch.astype(np.float32) / 255.0
        patch = torch.from_numpy(patch).permute(2, 0, 1)
        
        # Aux features: [depth, bbox_w, bbox_h]
        aux_features = torch.tensor([
            sample['depth'] / self.max_depth,
            sample['bbox_w'] / self.image_width,
            sample['bbox_h'] / self.image_height,
        ], dtype=torch.float32)
        
        difficulty = torch.tensor([sample['difficulty']], dtype=torch.float32)
        
        return patch, aux_features, difficulty


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for patches, aux_features, targets in tqdm(loader, desc="Training"):
        patches = patches.to(device)
        aux_features = aux_features.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(patches, aux_features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for patches, aux_features, targets in loader:
            patches = patches.to(device)
            aux_features = aux_features.to(device)
            targets = targets.to(device)
            
            outputs = model(patches, aux_features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    correlation = np.corrcoef(all_preds, all_targets)[0, 1]
    
    return total_loss / len(loader), correlation


def main():
    parser = argparse.ArgumentParser(description='Train Final Difficulty Predictor')
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='difficulty_predictor_final.pth')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--freeze_backbone', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    full_dataset = DifficultyDatasetFinal(args.csv_path, args.image_dir, augment=True)
    
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)
    
    # Final Model
    model = DifficultyPredictorFinal(freeze_backbone=args.freeze_backbone).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Final (ResNet-18 + Depth/BBox): {total_params:,} params")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_corr = -1
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_corr = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_corr={val_corr:.4f}")
        
        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(model.state_dict(), args.output)
            print(f"  -> Saved best model (corr={val_corr:.4f})")
    
    print(f"\nTraining complete. Best correlation: {best_corr:.4f}")
    print(f"Model saved to {args.output}")


if __name__ == '__main__':
    main()
