"""
Extract Detection Difficulty from OpenPCDet result.pkl

Modified version that works with OpenPCDet's result.pkl format.
Extracts per-object difficulty by matching detections to ground truth.
"""

import numpy as np
import pickle
from pathlib import Path
import argparse
from tqdm import tqdm
import csv


def load_kitti_label(label_path):
    """Load KITTI format label file."""
    objects = []
    with open(label_path, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            obj = {
                'idx': idx,
                'type': parts[0],
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox2d': [float(parts[4]), float(parts[5]), 
                          float(parts[6]), float(parts[7])],
                'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],
                'location': [float(parts[11]), float(parts[12]), float(parts[13])],
                'rotation_y': float(parts[14])
            }
            if obj['type'] in ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Person_sitting', 'Tram']:
                objects.append(obj)
    return objects


def compute_iou_3d_center(gt_loc, det_loc, gt_dims, det_dims, threshold=2.0):
    """
    Simplified IoU based on center distance and dimension overlap.
    """
    # Center distance
    center_dist = np.sqrt(
        (gt_loc[0] - det_loc[0])**2 + 
        (gt_loc[1] - det_loc[1])**2 + 
        (gt_loc[2] - det_loc[2])**2
    )
    
    # If too far, no match
    if center_dist > threshold:
        return 0.0
    
    # Simple dimension overlap
    dim_overlap = 1.0
    for i in range(3):
        overlap = min(gt_dims[i], det_dims[i]) / max(gt_dims[i], det_dims[i])
        dim_overlap *= overlap
    
    # Combine distance and dimension similarity
    dist_score = max(0, 1 - center_dist / threshold)
    iou_approx = dist_score * dim_overlap
    
    return iou_approx


def extract_difficulty_from_pkl(result_pkl, label_dir, output_csv):
    """
    Extract difficulty scores from OpenPCDet result.pkl.
    
    The pkl contains list of dicts with:
    - 'frame_id': frame identifier
    - 'boxes_lidar': (N, 7) predicted boxes [x, y, z, dx, dy, dz, heading]
    - 'score': (N,) confidence scores
    - 'pred_labels': (N,) class labels
    """
    with open(result_pkl, 'rb') as f:
        results = pickle.load(f)
    
    label_dir = Path(label_dir)
    
    all_difficulties = []
    
    for result in tqdm(results, desc="Matching detections to GT"):
        frame_id = result['frame_id']
        
        # Load GT labels
        label_path = label_dir / f'{frame_id}.txt'
        if not label_path.exists():
            continue
        
        gt_objects = load_kitti_label(label_path)
        
        if len(gt_objects) == 0:
            continue
        
        # Get detections (OpenPCDet format)
        # Keys: name, truncated, occluded, alpha, bbox, dimensions, location, 
        #       rotation_y, score, boxes_lidar, frame_id
        if 'boxes_lidar' in result and len(result['boxes_lidar']) > 0:
            det_names = result['name']  # Array of class names
            det_bboxes = result['bbox']  # [N, 4] 2D bboxes (left, top, right, bottom)
            det_scores = result['score']
        else:
            continue
        
        for gt in gt_objects:
            gt_bbox = gt['bbox2d']  # [left, top, right, bottom]
            
            # Find best matching detection of same class using 2D IoU
            best_iou = 0.0
            
            for i in range(len(det_names)):
                det_name = det_names[i]
                if det_name != gt['type']:
                    continue
                
                det_bbox = det_bboxes[i]
                
                # Compute 2D IoU
                x1 = max(gt_bbox[0], det_bbox[0])
                y1 = max(gt_bbox[1], det_bbox[1])
                x2 = min(gt_bbox[2], det_bbox[2])
                y2 = min(gt_bbox[3], det_bbox[3])
                
                if x2 <= x1 or y2 <= y1:
                    iou = 0.0
                else:
                    intersection = (x2 - x1) * (y2 - y1)
                    area_gt = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
                    area_det = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                    union = area_gt + area_det - intersection
                    iou = intersection / max(union, 1e-6)
                best_iou = max(best_iou, iou)
            
            # Difficulty = 1 - IoU
            difficulty = 1.0 - best_iou
            
            all_difficulties.append({
                'frame_id': frame_id,
                'object_idx': gt['idx'],
                'type': gt['type'],
                'bbox2d_left': gt['bbox2d'][0],
                'bbox2d_top': gt['bbox2d'][1],
                'bbox2d_right': gt['bbox2d'][2],
                'bbox2d_bottom': gt['bbox2d'][3],
                'loc_x': gt['location'][0],
                'loc_y': gt['location'][1],
                'loc_z': gt['location'][2],
                'difficulty': difficulty,
                'matched': 1 if best_iou > 0.3 else 0
            })
    
    # Save to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'frame_id', 'object_idx', 'type',
            'bbox2d_left', 'bbox2d_top', 'bbox2d_right', 'bbox2d_bottom',
            'loc_x', 'loc_y', 'loc_z',
            'difficulty', 'matched'
        ])
        writer.writeheader()
        writer.writerows(all_difficulties)
    
    # Statistics
    difficulties = [d['difficulty'] for d in all_difficulties]
    matched = sum(1 for d in all_difficulties if d['matched'])
    
    print(f"\nExtracted {len(all_difficulties)} objects")
    print(f"Difficulty: mean={np.mean(difficulties):.3f}, std={np.std(difficulties):.3f}")
    print(f"Matched: {matched}/{len(all_difficulties)} ({100*matched/len(all_difficulties):.1f}%)")
    print(f"Saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_pkl', type=str, required=True,
                        help='Path to OpenPCDet result.pkl')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Path to KITTI label_2 directory')
    parser.add_argument('--output', type=str, default='difficulty_labels.csv',
                        help='Output CSV file')
    args = parser.parse_args()
    
    extract_difficulty_from_pkl(args.result_pkl, args.label_dir, args.output)


if __name__ == '__main__':
    main()
