"""
Paint Difficulty Features onto LiDAR Point Clouds

This script reads the pre-generated difficulty map images (PNG) and 
paints the corresponding difficulty values onto the LiDAR points.

Process:
1. Load LiDAR point cloud (.bin)
2. Project points to Image coordinates (u, v)
3. Load Difficulty Map (Gray PNG)
4. Sample difficulty at (u, v)
5. Append difficulty to point features [x, y, z, intensity, difficulty]
6. Save as .npy or .bin
"""

import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import os

def load_kitti_calib(calib_file):
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    P2 = np.array([float(x) for x in lines[2].strip().split(' ')[1:]]).reshape(3, 4)
    
    # Rectification matrix (R0_rect)
    R0_rect = np.array([float(x) for x in lines[4].strip().split(' ')[1:]]).reshape(3, 3)
    R0_rect = np.insert(R0_rect, 3, values=[0,0,0], axis=0)
    R0_rect = np.insert(R0_rect, 3, values=[0,0,0,1], axis=1)

    # Tr_velo_to_cam
    Tr_velo_to_cam = np.array([float(x) for x in lines[5].strip().split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0,0,0,1], axis=0)
    
    return P2, R0_rect, Tr_velo_to_cam

def project_velo_to_image(pts_3d, P2, R0_rect, Tr_velo_to_cam):
    """
    Project 3D points to image plane
    """
    # 1. Velo -> Cam
    pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    pts_curr = np.dot(pts_3d_hom, Tr_velo_to_cam.T)
    
    # 2. Rectification
    pts_rect = np.dot(pts_curr, R0_rect.T)
    
    # 3. Projection
    # Only keep points in front of camera (z > 0)
    # But we project all to check image bounds later
    # pts_rect is Nx4, P2 is 3x4. P2.T is 4x3.
    # Result: Nx3 (homogeneous 2D: u, v, z)
    pts_2d_hom = np.dot(pts_rect, P2.T)
    
    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]
    return pts_2d, pts_rect[:, 2] # x,y and depth

def paint_dataset(args):
    lidar_dir = Path(args.lidar_dir)
    calib_dir = Path(args.calib_dir)
    map_dir = Path(args.map_dir)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all maps (since we only paint frames where we have maps)
    map_files = sorted(list(map_dir.glob('*.png')))
    print(f"Found {len(map_files)} difficulty maps")
    
    for map_file in tqdm(map_files, desc="Painting LiDAR"):
        frame_id = map_file.stem
        
        # Load LiDAR
        lidar_file = lidar_dir / f"{frame_id}.bin"
        if not lidar_file.exists(): continue
        
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        
        # Load Calib
        calib_file = calib_dir / f"{frame_id}.txt"
        if not calib_file.exists(): continue
        P2, R0_rect, Tr_velo = load_kitti_calib(calib_file)
        
        # Load Map
        diff_map = cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE) # (H, W) values 0-255
        H, W = diff_map.shape
        
        # Project
        pts_3d = points[:, :3]
        pts_2d, depth = project_velo_to_image(pts_3d, P2, R0_rect, Tr_velo)
        
        # Filter (in image and front of camera)
        u = np.round(pts_2d[:, 0]).astype(int)
        v = np.round(pts_2d[:, 1]).astype(int)
        
        mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth > 0)
        
        # Initialize Difficulty Channel (Feature 5)
        difficulty_feature = np.zeros((points.shape[0], 1), dtype=np.float32)
        
        # Correctly sample map values
        # u is column (width), v is row (height)
        # diff_map is (H, W), indexed by [row, col] -> [v, u]
        
        valid_u = u[mask]
        valid_v = v[mask]
        
        # Sample values 0-255
        sampled_vals = diff_map[valid_v, valid_u]
        
        # Normalize to 0-1
        difficulty_feature[mask, 0] = sampled_vals.astype(np.float32) / 255.0
        
        # Append to points
        painted_points = np.hstack((points, difficulty_feature))
        
        # Save
        if args.save_format == 'npy':
            np.save(output_dir / f"{frame_id}.npy", painted_points)
        else:
            painted_points.tofile(output_dir / f"{frame_id}.bin")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lidar_dir', type=str, required=True)
    parser.add_argument('--calib_dir', type=str, required=True)
    parser.add_argument('--map_dir', type=str, required=True, help='Directory containing .png difficulty maps')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--save_format', type=str, default='bin', choices=['bin', 'npy'])
    args = parser.parse_args()
    
    paint_dataset(args)

if __name__ == '__main__':
    main()
