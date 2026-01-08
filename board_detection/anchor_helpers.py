import cv2
import numpy as np
def get_pts_contour(mask2):
    #return points of the biggest contour
    if mask2.dtype == bool:
        mask = (mask2* 255).astype(np.uint8)
    else:
        mask = mask2
    contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours2, key=cv2.contourArea)
    pts = cnt[:, 0, :]
    return pts

def get_radial_coords(points,cx,cy):
    dx = points[:, 0].astype(np.float32) - cx
    dy = points[:, 1].astype(np.float32) - cy
    angles = np.arctan2(dy, dx) 
    angles = (angles + 2*np.pi) % (2*np.pi)
    radii  = np.sqrt(dx*dx + dy*dy)
    order = np.argsort(angles)
    angles_sorted = angles[order]
    radii_sorted  = radii[order]
    #points_sorted    = points[order]
    return angles_sorted,radii_sorted

def extend_angles_and_radii(angles_sorted,radii_sorted,k=30):
    k = 30  # how many samples to copy from each end (tune if needed)
    angles_sorted_extended = np.concatenate([
        angles_sorted,
        angles_sorted[:k] + 2*np.pi     # head shifted right
    ])
    radii_sorted_extended = np.concatenate([
        radii_sorted,
        radii_sorted[:k]
    ])
    return angles_sorted_extended,radii_sorted_extended

def get_normalized_profile(profile):  
    scale_factor = np.median(profile)
    normalized_profile = profile / scale_factor
    return normalized_profile

def get_max_min_angle_radii(angles_sorted_extended,radii_sorted_extended,peaks_idx,valleys_idx):
    max_angles = angles_sorted_extended[peaks_idx]
    max_radii  = radii_sorted_extended[peaks_idx]
    min_angles = angles_sorted_extended[valleys_idx]
    min_radii  = radii_sorted_extended[valleys_idx]
    return max_angles,max_radii,min_angles,min_radii
    
def radial_to_cartesian(theta,radius,cx,cy):
    x =  cx + radius * np.cos(theta)
    y = cy +radius * np.sin(theta)
    return int(x),int(y)

