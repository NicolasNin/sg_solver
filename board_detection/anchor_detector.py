from scipy.signal import find_peaks
import cv2
import numpy as np
from board_detection.anchor_helpers import *
from board_detection.mask_helpers import get_center
from board_detection.reference_data import cropped_image_ref_SHAPE

def get_peaks_and_valley(r,profile,d=20,p=0.03):
    #r = radii_sorted_extended
    #profile = radii_sorted_extended
    # --- maxima (star tips) ---

    profile = get_normalized_profile(profile)
    peaks_idx, peak_props = find_peaks(
        profile,
        distance=d,      # min index separation between peaks (tune)
        prominence=p      # how much a peak must stand out (tune)
    )
    
    # --- minima (star valleys) ---
    valleys_idx, valley_props = find_peaks(
        -profile,
        distance=d,
        prominence=p
    )
    return peaks_idx,valleys_idx

def get_min_pts(cropped_mask,cropped_image):
    # take a mask, compute its radial profile, compute peaks and valley return points 
    cx,cy =  get_center(cropped_image)
    pts = get_pts_contour(cropped_mask)
    angles_sorted,radii_sorted = get_radial_coords(pts,cx,cy)
    angles_sorted_extended,radii_sorted_extended = extend_angles_and_radii(angles_sorted,radii_sorted,k=30)
    peaks_idx,valleys_idx = get_peaks_and_valley(angles_sorted_extended,radii_sorted_extended)

    max_angles, max_radii, min_angles, min_radii = get_max_min_angle_radii(angles_sorted_extended,radii_sorted_extended,peaks_idx,valleys_idx)
    min_pts = [radial_to_cartesian(theta,r ,cx,cy) for theta,r in zip(min_angles,min_radii)]
    max_pts = [radial_to_cartesian(theta,r ,cx,cy) for theta,r in zip(max_angles,max_radii)]
    return min_pts,max_pts

def _warp_images(min_pts,cropped_image,min_pts_ref,method=cv2.RANSAC,threshold=5.0,return_homography=False):
    assert(len(min_pts)==len(min_pts_ref))
    min_pts_reshape =np.array(min_pts).astype(np.float32).reshape(-1, 2).astype(np.float32)
    min_pt_refs_reshape =np.array(min_pts_ref).astype(np.float32).reshape(-1, 2).astype(np.float32)
    # cv2.LMEDS 
    H, status = cv2.findHomography(min_pts_reshape, min_pt_refs_reshape, method , threshold)
    warped_img = cv2.warpPerspective(cropped_image, H,(cropped_image_ref_SHAPE[1],cropped_image_ref_SHAPE[0]))
    if return_homography:
        return warped_img,( H, status)
    return warped_img


def warp_image_from_mask(cropped_mask, cropped_image, min_pts_ref):
    min_pts,max_pts = get_min_pts(cropped_mask,cropped_image)
    if len(min_pts) != len(min_pts_ref):
        print(f" min pts found different than ref")
        return None
    return  _warp_images(min_pts,cropped_image,min_pts_ref)