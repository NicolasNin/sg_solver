import cv2
import numpy as np
from reference_data import SOLIDITY,MIN_AREA

def get_cnt_bbox(cnt):
    xs = cnt[:,0,0]
    ys = cnt[:,0,1]
    x_min,x_max = min(xs),max(xs)
    y_min,y_max = min(ys),max(ys)
    return x_min,y_min,x_max,y_max

def get_cnt_bbox_size(cnt):
    x_min,y_min,x_max,y_max = get_cnt_bbox(cnt)
    return x_max-x_min,y_max-y_min


def compute_invariant_contour2(cnt):
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    area = cv2.contourArea(cnt)
    solidity = area / hull_area if hull_area > 0 else 0
    total_area = np.prod(cnt.shape)
    percentage_area = area/total_area
    bbox_size_x,bbox_size_y    = get_cnt_bbox_size(cnt)
    return solidity,area,hull_area,bbox_size_x,bbox_size_y

def compute_invariant_contour(cnt):
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    area = cv2.contourArea(cnt)
    solidity = area / hull_area if hull_area > 0 else 0
    total_area = np.prod(cnt.shape)
    percentage_area = area/total_area
    return solidity,percentage_area,area

def compute_invariant_mask(mask):
    # compute invariant mask features to detect board
    mask_int = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    board_cnt = max(contours, key=cv2.contourArea)
    return compute_invariant_contour(board_cnt)

def invert_mask_and_get_large_connected(board_mask):
    board_mask = np.logical_not(board_mask)
    board_mask_int = (board_mask>0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        board_mask_int, connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_label = 1 + np.argmax(areas)
    biggest = (labels == max_label)
    return biggest

def get_possibles_mask(mask_list):
    possibles=[]
    for mask in mask_list:
        solidity,percentage_area,area = compute_invariant_mask(mask)
        #print(f"solidity: {solidity}, percentage_area: {percentage_area}, area: {area}")
        if abs(solidity-SOLIDITY)<0.01 and percentage_area>MIN_AREA and area>40000:
            possibles.append(mask)
    if len(possibles)>0:
        return possibles
    for mask in mask_list:
        invert_mask = invert_mask_and_get_large_connected(mask)
        solidity,percentage_area,area = compute_invariant_mask(invert_mask)  
        #print(f"INVERTED solidity: {solidity}, percentage_area: {percentage_area}, area: {area}")
        if abs(solidity-SOLIDITY)<0.01 and percentage_area>MIN_AREA and area>40000:
            possibles.append(invert_mask)
    return possibles


def pad_mask(mask, padding=10, value=False):
    """Pad a mask with specified value (default False)"""
    return np.pad(mask, 
                  pad_width=padding, 
                  mode='constant', 
                  constant_values=value)



def bbox_from_mask(mask, padding=10):
    """Get bounding box from binary mask with optional padding"""
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Add padding (clip to image boundaries)
    h, w = mask.shape
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    return (x_min, y_min, x_max, y_max)


def pad_mask(mask, padding=10, value=False):
    """Pad a mask with specified value (default False)"""
    return np.pad(mask, 
                  pad_width=padding, 
                  mode='constant', 
                  constant_values=value)

def bbox_inside(inner_bbox, outer_bbox, threshold=0.9):
    """
    Check if inner_bbox is inside outer_bbox
    threshold: fraction of inner bbox area that must be inside (default 90%)
    """
    x1_in, y1_in, x2_in, y2_in = inner_bbox
    x1_out, y1_out, x2_out, y2_out = outer_bbox
    
    # Calculate intersection
    x1_int = max(x1_in, x1_out)
    y1_int = max(y1_in, y1_out)
    x2_int = min(x2_in, x2_out)
    y2_int = min(y2_in, y2_out)
    
    # No intersection
    if x2_int <= x1_int or y2_int <= y1_int:
        return False
    
    # Calculate areas
    inner_area = (x2_in - x1_in) * (y2_in - y1_in)
    intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
    
    # Check if enough of inner is inside outer
    return (intersection_area / inner_area) >= threshold

def create_cropped(img,board_mask,padding=0):
    mask_uint8 = (board_mask * 255).astype(np.uint8)
    board_bbox = bbox_from_mask(board_mask,padding=padding)
    x_min, y_min, x_max, y_max =board_bbox
    cropped_mask = mask_uint8[y_min:y_max, x_min:x_max]
    cropped_mask = cropped_mask == 255
    cropped_image = img[y_min:y_max, x_min:x_max]
    return cropped_image,cropped_mask

# MORE GENERIC
def get_center(img):
    h,w,_ = img.shape
    center_point = [(w // 2)-1, h // 2]
    return center_point

def resize_keep_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resizes image maintaining aspect ratio.
    Specify EITHER width OR height.
    """
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

