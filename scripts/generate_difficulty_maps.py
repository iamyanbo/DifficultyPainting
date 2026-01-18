"""
Generate Difficulty Maps using Trained Difficulty Predictor V5

This script uses the trained ResNet-based difficulty predictor to generate 
per-pixel or per-box difficulty scores for the dataset.

Modes:
1. 'ground_truth': Uses GT boxes (for training set analysis)
2. 'predicted': Uses 2D detections (for inference/painting)
"""

import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from difficulty_predictor import DifficultyPredictorFinal

def load_kitti_calib(calib_file):
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    P2 = np.array([float(x) for x in lines[2].strip().split(' ')[1:]]).reshape(3, 4)
    return P2

def process_dataset(args, model, device):
    image_dir = Path(args.image_dir)
    label_dir = image_dir.parent / 'label_2'
    calib_dir = image_dir.parent / 'calib'
    
    # Get list of images
    images = sorted(list(image_dir.glob('*.png')))
    if not images:
        images = sorted(list(image_dir.glob('*.jpg')))
    
    print(f"Found {len(images)} images")
    
    results = {}
    
    model.eval()
    
    with torch.no_grad():
        for img_path in tqdm(images, desc="Generating Maps"):
            frame_id = img_path.stem
            
            # Load Image
            image = cv2.imread(str(img_path))
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load Calib (for depth)
            calib_file = calib_dir / f"{frame_id}.txt"
            if not calib_file.exists(): continue
            P2 = load_kitti_calib(calib_file)
            
            # Load Boxes (GT or Predicted)
            # For now, implementing GT mode for training set generation
            label_file = label_dir / f"{frame_id}.txt"
            if not label_file.exists(): continue
            
            boxes = []
            depths = []
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    cls = parts[0]
                    if cls not in ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Person_sitting', 'Tram']:
                        continue
                        
                    # 2D Box
                    x1, y1, x2, y2 = [float(x) for x in parts[4:8]]
                    
                    # 3D Depth (loc_z)
                    z = float(parts[13])
                    
                    boxes.append([x1, y1, x2, y2])
                    depths.append(z)
            
            if not boxes:
                results[frame_id] = []
                continue
                
            frame_difficulties = []
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                z = depths[i]
                
                # Crop Patch
                cx, cy = (x1 + x2)/2, (y1 + y2)/2
                w, h = (x2 - x1) * 1.2, (y2 - y1) * 1.2
                
                px1 = max(0, int(cx - w/2))
                py1 = max(0, int(cy - h/2))
                px2 = min(image.shape[1], int(cx + w/2))
                py2 = min(image.shape[0], int(cy + h/2))
                
                patch = image[py1:py2, px1:px2]
                if patch.size == 0:
                    frame_difficulties.append(0.5) # Default
                    continue
                
                patch = cv2.resize(patch, (64, 64))
                patch_tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Aux Features [depth, w, h]
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                aux = torch.tensor([[z/70.0, bbox_w/1242.0, bbox_h/375.0]], dtype=torch.float32).to(device)
                
                # Predict
                difficulty = model(patch_tensor, aux)
                frame_difficulties.append(difficulty.item())

            # Store results
            if args.dense_map:
                # Generate a dense difficulty map (H, W)
                difficulty_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                
                # Draw boxes with difficulty values
                for i, diff in enumerate(frame_difficulties):
                    x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                    box_mask = np.zeros_like(difficulty_map)
                    cv2.rectangle(box_mask, (x1, y1), (x2, y2), float(diff), -1)
                    difficulty_map = np.maximum(difficulty_map, box_mask)
                
                # Save as uint8 PNG
                save_path = args.output_dir / f"{frame_id}.png"
                difficulty_img = (difficulty_map * 255).astype(np.uint8)
                cv2.imwrite(str(save_path), difficulty_img)
            else:
                results[frame_id] = list(zip(boxes, frame_difficulties))
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='difficulty_predictor_final.pth')
    parser.add_argument('--output_dir', type=str, default='difficulty_maps')
    parser.add_argument('--dense_map', action='store_true', help='Generate dense difficulty maps (images)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup output
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    model = DifficultyPredictorFinal()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded")
    
    # Generate
    results = process_dataset(args, model, device)
    
    if not args.dense_map:
        # Save pickle only for sparse mode
        with open(args.output_dir / 'difficulty_boxes.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved sparse difficulty maps to {args.output_dir / 'difficulty_boxes.pkl'}")
    else:
        print(f"Saved dense difficulty maps to {args.output_dir}")

if __name__ == '__main__':
    main()
