"""
Helper functions for processing YOLO pose detection results.
"""
import numpy as np
import cv2
from board_detection.reference_data import YOLO_KEY_PTS_ID_MIN


def extract_cropped_board(result, padding=10, expansion_pixels=5, keypoint_indices=None):
    """
    Extract cropped board image and translated keypoints from YOLO pose result.
    Creates a mask from all keypoints on the cropped image.
    
    Args:
        result: Ultralytics YOLO result object with:
            - result.orig_img: original image (numpy array, BGR format)
            - result.boxes: detected bounding boxes
            - result.keypoints: detected keypoints
        padding: pixels to add around bbox when cropping
        expansion_pixels: pixels to expand the mask (morphological dilation) to avoid cropping rounded edges
        keypoint_indices: list of keypoint indices to extract for return (default: YOLO_KEY_PTS_ID_MIN)
                         Note: mask is built from ALL keypoints, but only these are returned
    
    Returns:
        tuple: (cropped_image, translated_keypoints, mask)
            - cropped_image: numpy array in BGR format (OpenCV compatible)
            - translated_keypoints: numpy array of shape (N, 2) with x,y coords relative to crop
                                   (only the keypoints specified by keypoint_indices)
            - mask: binary mask same shape as cropped_image, built from ALL keypoints
    """
    if keypoint_indices is None:
        keypoint_indices = YOLO_KEY_PTS_ID_MIN
    
    # Get original image
    orig_img = result.orig_img
    img_height, img_width = orig_img.shape[:2]
    
    # Get bounding box in xyxy format
    bbox_xyxy = result.boxes.xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox_xyxy.astype(int)
    
    # Add padding and clamp to image bounds
    x1_padded = max(0, x1 - padding)
    y1_padded = max(0, y1 - padding)
    x2_padded = min(img_width, x2 + padding)
    y2_padded = min(img_height, y2 + padding)
    
    # Crop the image
    cropped_image = orig_img[y1_padded:y2_padded, x1_padded:x2_padded].copy()
    crop_height, crop_width = cropped_image.shape[:2]
    
    # Extract ALL keypoints (x, y, confidence) for mask creation
    keypoints_data = result.keypoints.data.cpu().numpy()[0]  # Shape: (num_keypoints, 3)
    all_keypoints_xy = keypoints_data[:, :2]  # Shape: (N, 2)
    
    # Translate ALL keypoints to cropped image coordinates
    all_keypoints_translated = all_keypoints_xy - np.array([x1_padded, y1_padded])
    
    # Create mask from ALL keypoints on the cropped image
    mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    
    # Convert keypoints to integer coordinates for polygon drawing
    polygon_points = all_keypoints_translated.astype(np.int32).reshape((-1, 1, 2))
    
    # Fill the polygon formed by the ordered keypoints
    cv2.fillPoly(mask, [polygon_points], 255)
    
    # Optionally expand the mask to avoid cropping rounded edges
    if expansion_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (expansion_pixels * 2 + 1, expansion_pixels * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Now extract only the SPECIFIC keypoints requested for return
    selected_keypoints = keypoints_data[keypoint_indices, :2]  # Only x, y (drop confidence)
    translated_keypoints = selected_keypoints - np.array([x1_padded, y1_padded])
    
    # Clamp the selected keypoints to be inside the crop (safety check)
    translated_keypoints[:, 0] = np.clip(translated_keypoints[:, 0], 0, crop_width - 1)
    translated_keypoints[:, 1] = np.clip(translated_keypoints[:, 1], 0, crop_height - 1)
    
    return cropped_image, translated_keypoints, mask,all_keypoints_translated


def extract_cropped_board_with_expansion(result, initial_padding=10, keypoint_indices=None):
    """
    Extract cropped board with automatic bbox expansion if keypoints fall outside.
    
    This version checks if keypoints are outside the initial crop and expands
    the bounding box accordingly rather than clamping.
    
    Args:
        result: Ultralytics YOLO result object (orig_img in BGR format)
        initial_padding: initial pixels to add around bbox
        keypoint_indices: list of keypoint indices to extract (default: YOLO_KEY_PTS_ID_MIN)
    
    Returns:
        tuple: (cropped_image, translated_keypoints, mask)
            - cropped_image: numpy array in BGR format (OpenCV compatible)
            - translated_keypoints: numpy array of shape (N, 2) with x,y coords relative to crop
            - mask: binary mask same shape as cropped_image (all 255, for pipeline compatibility)
    """
    if keypoint_indices is None:
        keypoint_indices = YOLO_KEY_PTS_ID_MIN
    
    # Get original image
    orig_img = result.orig_img
    img_height, img_width = orig_img.shape[:2]
    
    # Get bounding box
    bbox_xyxy = result.boxes.xyxy.cpu().numpy()[0]
    x1, y1, x2, y2 = bbox_xyxy.astype(int)
    
    # Extract keypoints
    keypoints_data = result.keypoints.data.cpu().numpy()[0]
    selected_keypoints = keypoints_data[keypoint_indices, :2]
    
    # Find min/max of keypoints to ensure they're all included
    kpt_x_min = selected_keypoints[:, 0].min()
    kpt_x_max = selected_keypoints[:, 0].max()
    kpt_y_min = selected_keypoints[:, 1].min()
    kpt_y_max = selected_keypoints[:, 1].max()
    
    # Expand bbox to include all keypoints plus padding
    x1_final = max(0, min(x1 - initial_padding, int(kpt_x_min) - initial_padding))
    y1_final = max(0, min(y1 - initial_padding, int(kpt_y_min) - initial_padding))
    x2_final = min(img_width, max(x2 + initial_padding, int(kpt_x_max) + initial_padding))
    y2_final = min(img_height, max(y2 + initial_padding, int(kpt_y_max) + initial_padding))
    
    # Crop the image
    cropped_image = orig_img[y1_final:y2_final, x1_final:x2_final].copy()
    
    # Translate keypoints
    translated_keypoints = selected_keypoints - np.array([x1_final, y1_final])
    
    # Create mask
    mask = np.full(cropped_image.shape[:2], 255, dtype=np.uint8)
    
    return cropped_image, translated_keypoints, mask


def create_mask_from_keypoints(result, expansion_pixels=10):
    """
    Create a binary mask from YOLO pose keypoints by filling the polygon formed by ordered keypoints.
    
    Since the keypoints are ordered around the perimeter of the star-shaped board, we can connect
    them to form a polygon and fill it. The mask is then optionally expanded to avoid cropping
    the rounded edges of the board.
    
    Args:
        result: Ultralytics YOLO result object with:
            - result.orig_img: original image (numpy array, BGR format)
            - result.keypoints: detected keypoints
        keypoint_indices: list of keypoint indices to use (default: YOLO_KEY_PTS_ID_MIN)
        expansion_pixels: number of pixels to expand the mask (using morphological dilation)
                         to avoid cropping rounded board edges. Set to 0 for no expansion.
    
    Returns:
        tuple: (mask, keypoints_xy)
            - mask: binary mask (uint8, 0 or 255) same size as orig_img
            - keypoints_xy: numpy array of shape (N, 2) with x,y coordinates of keypoints
    """
    
    orig_img = result.orig_img
    img_height, img_width = orig_img.shape[:2]
    
    # Extract keypoints (x, y, confidence)
    keypoints_data = result.keypoints.data.cpu().numpy()[0]  # Shape: (num_keypoints, 3)

    selected_keypoints = keypoints_data[:, :2]  # Shape: (N, 2)
    
    # Create empty mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Convert keypoints to integer coordinates for polygon drawing
    polygon_points = selected_keypoints.astype(np.int32).reshape((-1, 1, 2))
    
    # Fill the polygon formed by the ordered keypoints
    cv2.fillPoly(mask, [polygon_points], 255)
    
    # Optionally expand the mask to avoid cropping rounded edges
    if expansion_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (expansion_pixels * 2 + 1, expansion_pixels * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask, selected_keypoints
