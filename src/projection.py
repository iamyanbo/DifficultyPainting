import numpy as np
import os

class Calibration:
    def __init__(self, calib_file):
        """
        Parses the KITTI calibration file.
        """
        self.calib = self.read_calib_file(calib_file)
        
        # Projection matrix from rect camera coord to image2 coord
        self.P2 = self.calib['P2'].reshape(3, 4)
        
        # Rigid transform from Velodyne coord to reference camera coord
        if 'Tr_velo_to_cam' in self.calib:
            self.Tr_velo_to_cam = self.calib['Tr_velo_to_cam'].reshape(3, 4)
        else:
            self.Tr_velo_to_cam = self.calib['Tr_velo_cam'].reshape(3, 4)
        
        # Rotation from reference camera coord to rect camera coord
        if 'R0_rect' in self.calib:
            self.R0_rect = self.calib['R0_rect'].reshape(3, 3)
        elif 'R_rect' in self.calib:
             self.R0_rect = self.calib['R_rect'].reshape(3, 3)
        else:
            raise KeyError(f"Neither R0_rect nor R_rect found. Keys: {list(self.calib.keys())}")

    def read_calib_file(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line: continue
                
                if ':' in line:
                    key, value = line.split(':', 1)
                else:
                    # Handle Tr_velo_cam which might imply space separation
                    parts = line.split(None, 1) # Split on first whitespace
                    if len(parts) == 2:
                        key, value = parts
                    else:
                        continue
                        
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Output: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def project_velo_to_image(self, pts_3d_velo):
        """
        Input: nx3 points in Velodyne coordinates
        Output: nx2 points in Image2 coordinates
        """
        # 1. Velodyne to Reference Camera
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        pts_3d_ref = np.dot(pts_3d_velo, self.Tr_velo_to_cam.T) # nx3

        # 2. Reference Camera to Rectified Camera
        pts_3d_rect = np.dot(pts_3d_ref, self.R0_rect.T) # nx3

        # 3. Rectified Camera to Image Plane
        pts_3d_rect = self.cart2hom(pts_3d_rect) # nx4
        pts_2d_hom = np.dot(pts_3d_rect, self.P2.T) # nx3

        # 4. Normalize (u/w, v/w)
        pts_2d_hom[:, 0] /= pts_2d_hom[:, 2]
        pts_2d_hom[:, 1] /= pts_2d_hom[:, 2]
        
        return pts_2d_hom[:, 0:2], pts_3d_rect[:, 2] # Return (u, v) and depth

def filter_points_in_image(pts_2d, image_shape):
    """
    Filters points that are outside the image boundaries.
    image_shape: (H, W)
    """
    H, W = image_shape
    mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) & \
           (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
    return mask
