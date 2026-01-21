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



def rotationMatrixToEulerXYZ(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])   # roll  about X
        y = np.arctan2(-R[2, 0], sy)       # pitch about Y
        z = np.arctan2(R[1, 0], R[0, 0])   # yaw   about Z
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0

    return np.degrees(np.array([x, y, z]))


def correct_for_height(point_warped_naive, h, R, t, K_ref, scale_factor=1.0):
    """
    Given a point that was warped assuming z=0, correct it for actual height h.
    
    The correction depends on the viewing angle (from R) and height h.
    
    Args:
        point_warped_naive: (x, y) in warped image (computed with H assuming z=0)
        h: actual height of the point
        R: rotation matrix from decomposition
        t: translation vector from decomposition  
        K_ref: intrinsic matrix for the reference/warped image
        scale_factor: converts h to the same units as t (you'll need to calibrate this)
    
    Returns:
        corrected_point: (x, y) corrected position in warped image
    """
    # Extract the "tilt" from the rotation matrix
    # R transforms from original camera frame to rectified frame
    
    # The translation t (normalized) gives the viewing direction change
    # For a top-down rectified view, t indicates how "off-axis" the original view was
    
    t_normalized = t.ravel() / (np.linalg.norm(t.ravel()) + 1e-9)
    
    # The shift due to height h is proportional to h and the horizontal components of t
    # In the rectified (top-down) view, the shift is:
    #   delta_x ≈ h * t[0] / t[2] * scale
    #   delta_y ≈ h * t[1] / t[2] * scale
    
    # But we need to be more careful with the geometry...
    
    # Alternative: use the rotation to find the viewing angle
    # The camera's optical axis in the board frame
    optical_axis_in_board = R.T @ np.array([0, 0, 1])
    
    # The tilt angles
    tilt_x = np.arctan2(optical_axis_in_board[0], optical_axis_in_board[2])
    tilt_y = np.arctan2(optical_axis_in_board[1], optical_axis_in_board[2])
    
    # Shift in board coordinates (mm or whatever units h is in)
    shift_board_x = h * np.tan(tilt_x)
    shift_board_y = h * np.tan(tilt_y)
    
    # Convert to pixels in warped image
    # You need the scale: pixels per unit length in the warped image
    # This depends on your setup - you might know this from your reference board
    
    shift_px_x = shift_board_x * scale_factor
    shift_px_y = shift_board_y * scale_factor
    
    corrected = np.array([
        point_warped_naive[0] - shift_px_x,
        point_warped_naive[1] - shift_px_y
    ])
    
    return corrected