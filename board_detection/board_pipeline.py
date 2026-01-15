import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from board_detection.reference_data import *
from board_detection.mask_helpers import *
from board_detection.color_helpers import *
from board_detection.anchor_detector import *
from board_detection.anchor_detector import _warp_images  # Explicit import for private function
from board_detection.anchor_helpers import *
from board_detection.white_triangles import *
from board_detection.graph_clustering import *
from board_detection.piece_identification import *
from board_detection.yolo_helpers import *
import time
from ultralytics import YOLO

_yolo_model = YOLO("/home/nicolas/code/star_genius_solver/notebooks/runs/pose/train4/weights/best.pt")  # adjust path

def get_yolo_model():
    return _yolo_model
@dataclass
class PieceDetection:
    """Represents a detected piece on the board."""
    name: str  # e.g., "T3", "3B", etc.
    cell_ids: List[int]  # List of cell IDs occupied by this piece
    orientation: str  # Orientation of the piece
    anchor: int  # Anchor cell ID


@dataclass
class WhiteTriangleDetection:
    """Represents a detected white triangle (blocker or small marker)."""
    cell_id: int  # Best matching cell ID
    triangle_type: str  # "big" or "small"
    center_of_mass: Tuple[float, float]  # (x, y) center position
    orientation: str  # "up" or "down"
    hull: Optional[np.ndarray] = None  # Convex hull points for drawing


@dataclass
class CameraParameters:
    """Camera intrinsic and extrinsic parameters from homography decomposition."""
    K: np.ndarray  # 3x3 camera intrinsic matrix
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector
    H: np.ndarray  # 3x3 homography matrix


@dataclass
class IntermediateImages:
    """Intermediate processing images for debugging/visualization."""
    original: np.ndarray  # Original input image
    cropped: np.ndarray  # Cropped board region
    cropped_mask: np.ndarray  # Mask of the cropped region
    normalized: np.ndarray  # L-normalized image
    warped: np.ndarray  # Perspective-corrected image
    warped_no_bg: np.ndarray  # Warped image with background removed


@dataclass
class BoardDetectionResult:
    """Complete result from board detection pipeline."""
    # Main results
    pieces: List[PieceDetection]  # Detected pieces
    white_triangles: List[int]  # Cell IDs with white blocker triangles
    
    # Geometric data
    min_pts: np.ndarray  # Detected keypoints (7 points)
    corrected_points: Dict[int, Tuple[int, int]]  # Cell centroids (cell_id -> (x, y))
    corrected_points_ori: Dict[int, Tuple[int, int]]  # Cell centroids (cell_id -> (x, y))
    corrections: Dict[int, Tuple[int, int]]  # Cell corrections (cell_id -> (dx, dy))
    white_corrections: Dict[int, Tuple[int, int]]  # Cell corrections (cell_id -> (dx, dy))
    # color data
    color_data: Dict[int, Dict[str, float]]
    # Camera parameters
    camera: Optional[CameraParameters] = None
    # Intermediate images (optional, for debugging)
    images: Optional[IntermediateImages] = None
    
    # White triangle details (optional, for visualization)
    white_triangle_detections: Optional[List[WhiteTriangleDetection]] = None
    small_triangles: Optional[List[WhiteTriangleDetection]] = None
    all_keypoints: Optional[List[Tuple[int, int]]] = None
    # Timing info
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pieces": [
                {
                    "name": p.name,
                    "cells": p.cell_ids,
                    "orientation": p.orientation,
                    "anchor": p.anchor
                }
                for p in self.pieces
            ],
            "white_triangles": self.white_triangles,
            "processing_time_ms": self.processing_time_ms,
        }


def visualize_detection(result: BoardDetectionResult, draw_piece_names: bool = True) -> np.ndarray:
    """
    Visualize detection results on the warped board image.
    
    Args:
        result: BoardDetectionResult from pipeline
        draw_piece_names: Whether to draw piece names on the board
    
    Returns:
        Debug image with visualizations drawn
    """
    # Get warped image and make a copy for drawing
    debug_img = result.images.warped_no_bg.copy()
    
    # Draw big white triangles (blockers) in green
    if result.white_triangle_detections:
        cnts_big = [wt.hull for wt in result.white_triangle_detections]
        cv2.drawContours(debug_img, cnts_big, -1, (0, 255, 0), 3)
    
    # Draw small white triangles in cyan
    if result.small_triangles:
        cnts_small = [st.hull for st in result.small_triangles]
        cv2.drawContours(debug_img, cnts_small, -1, (255, 255, 0), 3)
    
    # Draw piece names
    if draw_piece_names and result.pieces:
        for piece in result.pieces:
            # Get the anchor cell position
            if piece.anchor in result.corrected_points:
                x, y = result.corrected_points[piece.anchor]
                
                # Draw piece name (2-letter short name)
                cv2.putText(
                    debug_img,
                    piece.name,
                    (int(x) - 20, int(y) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 255),  # Magenta
                    2,
                    cv2.LINE_AA
                )
    
    return debug_img


def board_pipeline(image_path):
    print("Running full pipeline")
    t0=time.time()
    image = cv2.imread(image_path)
    possibles = get_possibles_mask_from_image_path(image_path)
    print(f"Mask found {len(possibles)} in {time.time()-t0:.2f}s")
    if len(possibles)==0:
        print(f"No mask found for image {image_path}")
        return None

    mask = possibles[0]
    data_image = {}
    t1=time.time()
    image = cv2.imread(image_path)
    cropped_image, cropped_mask = create_cropped(image,mask)
    normalized_bgr,cur = normalize_L_mean(cropped_image, cropped_mask, target_L_mean=100)
    min_pts,max_pts = get_min_pts(cropped_mask,cropped_image)
    print(f"Min pts found {len(min_pts)} in {time.time()-t1:.2f}s")

    if len(min_pts)!=7:
        print(f"Number of min pts not equal to seven")
        return None
    t2=time.time()
    warped = warp_image_from_mask(cropped_mask, normalized_bgr, MIN_PTS_REF)
    cropped_mask_int = cropped_mask.astype(np.uint8) * 255
    image_no_bg = cv2.bitwise_and(normalized_bgr, normalized_bgr, mask=cropped_mask_int)
    warped_no_bg = warp_image_from_mask(cropped_mask, image_no_bg, MIN_PTS_REF)
    data_image={"image":image,
              "cropped_image":cropped_image,
              "cropped_mask_int":cropped_mask_int,
              "image_no_bg":image_no_bg,
              "warped_no_bg":warped_no_bg,
              "normalized_bgr":normalized_bgr,
              "cropped_mask":cropped_mask,
             "min_pts":min_pts}

    #recompute homography to inverse it not optimal at all
    min_pts_reshape =np.array(min_pts).astype(np.float32).reshape(-1, 2).astype(np.float32)
    min_pt_refs_reshape =np.array(MIN_PTS_REF).astype(np.float32).reshape(-1, 2).astype(np.float32)
    H, status = cv2.findHomography(min_pts_reshape, min_pt_refs_reshape, cv2.RANSAC  , 5.0)
    image_no_bg = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask_int)
    warped_img = cv2.warpPerspective(image_no_bg, H,(cropped_image_ref_SHAPE[1],cropped_image_ref_SHAPE[0]))

    img_height,img_width = normalized_bgr.shape[:2]
    focal_length_px = img_width *1  # Rough guess for ~50-60° FOV
    print(img_width / 2,img_height / 2)
    K = np.array([[focal_length_px, 0, img_width / 2],
                  [0, focal_length_px, img_height / 2],
                  [0, 0, 1]])
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    pts1_norm = cv2.undistortPoints(min_pts_reshape, K, None)  # shape (N, 1, 2)
    pts2_norm = cv2.undistortPoints(min_pt_refs_reshape, K, None)
    possible = cv2.filterHomographyDecompByVisibleRefpoints(
        rotations, normals, pts1_norm, pts2_norm, status
    )
    if possible is not None:
        possible = np.array(possible).ravel()  # flatten to [0, 2, ...] etc.
        n_expected = np.array([0, 0, 1.0])   # example expected normal
        best_i = None
        best_dot = -1e9
        for idx in possible:
            idx = int(idx)                   # make sure it's a scalar int
            n = normals[idx].ravel()         # (3,)
            n = n / np.linalg.norm(n)
            dot = np.dot(n, n_expected)
            if dot > best_dot:
                best_dot = dot
                best_i = idx

        
    R_best = rotations[best_i]
    t_best = translations[best_i]
    data_image.update({"R":R_best,"t":t_best,"K":K,"warped_img2":warped_img})  
    print(f"Homography found in {time.time()-t2:.2f}s")
    #compute corrected centroid
    centroid_ref = get_centroid_ref(padding=19)[0]
    point_warped_naive = MIN_PTS_REF[0]
    h=100
    corrected_points = {}
    for id in centroid_ref:
        point_warped_naive = centroid_ref[id]
        corrected_pt = correct_for_height(point_warped_naive, h, R_best, t_best, K, scale_factor=1)
        x,y = int(corrected_pt[0]),int(corrected_pt[1])
        corrected_points[id]=x,y
        #corrected_pt=              correct_for_height_perspective(point_warped_naive, h, R_best, t_best, K, scale_factor=1.0)
        dx,dy=(x-point_warped_naive[0],y-point_warped_naive[1])
    data_image.update({"corrected_points":corrected_points})  
    t3=time.time()
    white_triangles = find_white_triangles(warped_no_bg,corrected_points) #need to work on clearer warped but need to make sure correction is the same
    print(f"White triangles found in {time.time()-t3:.2f}s")
    white_triangles_id = [x["best_post"]+1 for x in white_triangles if x["result"]=="big"] # id start at 0
    small_triangles = [x for x in white_triangles if x["result"]=="small"]
    #we can correct the corrected points based on small triangles:
    for small_tri in small_triangles:
        id = small_tri["best_post"]+1
        center = small_tri["center_of_mass"]
        x,y = int(center[0]), int(center[1])
        xc,yc = corrected_points[id]
        #print(id,x-xc,y-yc)
        corrected_points[id] = x,y
    t4=time.time()
    color_data = compute_color_values(warped_img,corrected_points) #on warped img 2 might be the same
    print(f"Color data computed in {time.time()-t4:.2f}s")
    t5=time.time()
    color_data_theta = {id:color_data[id]["theta_med"] for id in range(1,49)}
    clusters = cluster_graph_values(color_data_theta,5.5,distance=angle_diff,to_prune=white_triangles_id)
    print(f"clusters found #{len(clusters)} in {time.time()-t5:.2f}s")
    pieces = []
    used_names = set()
    for c in clusters:
        piece_name, orientation, anchor = identify_piece_from_cells(c)
        if piece_name is not None:
            # Handle duplicate names - T3 and 3B have the same shape
            if piece_name in used_names:
                if piece_name == "T3":
                    piece_name = "3B"  # Use alternate name for second T3
            used_names.add(piece_name)
            pieces.append((piece_name, list(c), orientation, anchor))  # (name, cell_ids, orientation, anchor)
    t6=time.time()
    print(f"Pieces identified {pieces}")
    print(f"white triangles {white_triangles_id}")
    print(f"Pieces identified in {time.time()-t6:.2f}s")
    print(f"Total time {time.time()-t0:.2f}s")

    return pieces, white_triangles_id


def board_pipelineYolo(image_path,do_correction=True):
    print("Running full pipeline")
    t0=time.time()
    print("Run custom yolo pose")
    results = _yolo_model(image_path)
    if len(results)==0:
        print("No board detected")
        return None
    
    result = results[0]
    
    # Extract cropped board with mask built from keypoints
    cropped_image, min_pts, cropped_mask,all_keypoints = extract_cropped_board(result, padding=10, expansion_pixels=5)


    print(f"Min pts found {len(min_pts)}")
    normalized_bgr,cur = normalize_L_mean(cropped_image, cropped_mask, target_L_mean=100)
    #min_pts,max_pts = get_min_pts(cropped_mask,cropped_image)
    #print(f"Min pts found {len(min_pts)} in {time.time()-t1:.2f}s")
    # we could check confidence or some other criteria for now that will do

    t2=time.time()
    warped = _warp_images(min_pts,cropped_image,MIN_PTS_REF)
    #warped = warp_image_from_mask(cropped_mask, normalized_bgr, MIN_PTS_REF)
    cropped_mask_int = cropped_mask.astype(np.uint8) * 255
    image_no_bg = cv2.bitwise_and(normalized_bgr, normalized_bgr, mask=cropped_mask_int)
    #warped_no_bg = warp_image_from_mask(cropped_mask, image_no_bg, MIN_PTS_REF)
    warp_method = cv2.RANSAC
    warp_threshold = 5.0
    warped_no_bg = _warp_images(min_pts,image_no_bg,MIN_PTS_REF,warp_method,warp_threshold)

    data_image={"image":result[0].orig_img,
              "cropped_image":cropped_image,
              "cropped_mask_int":cropped_mask_int,
              "image_no_bg":image_no_bg,
              "warped_no_bg":warped_no_bg,
              "normalized_bgr":normalized_bgr,
              "cropped_mask":cropped_mask,
             "min_pts":min_pts}

    #recompute homography to inverse it not optimal at all
    min_pts_reshape =np.array(min_pts).astype(np.float32).reshape(-1, 2).astype(np.float32)
    min_pt_refs_reshape =np.array(MIN_PTS_REF).astype(np.float32).reshape(-1, 2).astype(np.float32)
    H, status = cv2.findHomography(min_pts_reshape, min_pt_refs_reshape, warp_method  , warp_threshold)
    image_no_bg = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask_int)
    warped_img = cv2.warpPerspective(image_no_bg, H,(cropped_image_ref_SHAPE[1],cropped_image_ref_SHAPE[0]))

    img_height,img_width = normalized_bgr.shape[:2]
    focal_length_px = img_width *1  # Rough guess for ~50-60° FOV
    print(img_width / 2,img_height / 2)
    K = np.array([[focal_length_px, 0, img_width / 2],
                  [0, focal_length_px, img_height / 2],
                  [0, 0, 1]])
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    pts1_norm = cv2.undistortPoints(min_pts_reshape, K, None)  # shape (N, 1, 2)
    pts2_norm = cv2.undistortPoints(min_pt_refs_reshape, K, None)
    possible = cv2.filterHomographyDecompByVisibleRefpoints(
        rotations, normals, pts1_norm, pts2_norm, status
    )
    if possible is not None:
        possible = np.array(possible).ravel()  # flatten to [0, 2, ...] etc.
        n_expected = np.array([0, 0, 1.0])   # example expected normal
        best_i = None
        best_dot = -1e9
        for idx in possible:
            idx = int(idx)                   # make sure it's a scalar int
            n = normals[idx].ravel()         # (3,)
            n = n / np.linalg.norm(n)
            dot = np.dot(n, n_expected)
            if dot > best_dot:
                best_dot = dot
                best_i = idx

        
    R_best = rotations[best_i]
    t_best = translations[best_i]
    data_image.update({"R":R_best,"t":t_best,"K":K,"warped_img2":warped_img})  
    print(f"Homography found in {time.time()-t2:.2f}s")
    #compute corrected centroid
    centroid_ref = get_centroid_ref(padding=19)[0]
    point_warped_naive = MIN_PTS_REF[0]
    h=100
    corrected_points = {}
    corrections ={}
    for id in centroid_ref:
        point_warped_naive = centroid_ref[id]
        corrected_pt = correct_for_height(point_warped_naive, h, R_best, t_best, K, scale_factor=1)
        x,y = round(corrected_pt[0]),round(corrected_pt[1])
        if do_correction:
            corrected_points[id]= x,y 
            dx,dy=(x-point_warped_naive[0],y-point_warped_naive[1])
            corrections[id] = dx,dy
        else:
            corrected_points[id]= point_warped_naive

    
    data_image.update({"corrected_points":corrected_points})  
    t3=time.time()
    white_triangles = find_white_triangles(warped_no_bg,corrected_points) #need to work on clearer warped but need to make sure correction is the same
    print(f"White triangles found in {time.time()-t3:.2f}s")
    white_triangles_id = [x["best_post"]+1 for x in white_triangles if x["result"]=="big"] # id start at 0
    small_triangles = [x for x in white_triangles if x["result"]=="small" or x["result"]=="small_large"]
    big_triangles = [x for x in white_triangles if x["result"]=="big"]

    #we can correct the corrected points based on small triangles:
    white_corrections = {}
    corrected_points_ori = corrected_points.copy() 
    for tri in big_triangles:
        id = tri["best_post"]+1
        center = tri["center_of_mass"]
        x,y = round(center[0]), round(center[1])
        xc,yc = corrected_points[id]
        #print("correction based on big triangles",id,x-xc,y-yc,int(x),int(y))
        white_corrections[id] = x-xc,y-yc
        corrected_points[id] = x,y

    for small_tri in small_triangles:
        id = small_tri["best_post"]+1
        center = small_tri["center_of_mass"]
        x,y = round(center[0]), round(center[1])
        xc,yc = corrected_points[id]
        #print("correction based on small triangles",id,x-xc,y-yc,int(x),int(y))
        white_corrections[id] = x-xc,y-yc
        corrected_points[id] = x,y

    dx = np.median([white_corrections[x][0] for x in white_corrections])
    dy = np.median([white_corrections[x][1] for x in white_corrections])
    print(f"Warp correction {corrections[1]}")
    print(f"median white corrections {dx},{dy}")
    for idx in corrected_points:
        if idx not in white_corrections:
            corrected_points[idx] = round(corrected_points[idx][0]+dx),round(corrected_points[idx][1]+dy)
    
    data_image.update({"white_corrections":white_corrections})
    t4=time.time()
    color_data = compute_color_values(warped_img,corrected_points) #on warped img 2 might be the same
    print(f"Color data computed in {time.time()-t4:.2f}s")
    t5=time.time()
    color_data_theta = {id:color_data[id]["theta_med"] for id in range(1,49)}
    clusters = cluster_graph_values(color_data_theta,5.5,distance=angle_diff,to_prune=white_triangles_id)
    print(f"clusters found #{len(clusters)} in {time.time()-t5:.2f}s")
    pieces = []
    used_names = set()
    for c in clusters:
        piece_name, orientation, anchor = identify_piece_from_cells(c)
        if piece_name is not None:
            # Handle duplicate names - T3 and 3B have the same shape
            if piece_name in used_names:
                if piece_name == "T3":
                    piece_name = "3B"  # Use alternate name for second T3
            used_names.add(piece_name)
            pieces.append((piece_name, list(c), orientation, anchor))  # (name, cell_ids, orientation, anchor)
    t6=time.time()
    print(f"Pieces identified {pieces}")
    print(f"white triangles {white_triangles_id}")
    print(f"Pieces identified in {time.time()-t6:.2f}s")
    total_time = time.time()-t0
    print(f"Total time {total_time:.2f}s")

    # Convert to dataclasses
    piece_detections = [
        PieceDetection(name=name, cell_ids=cells, orientation=orient, anchor=anc)
        for name, cells, orient, anc in pieces
    ]
    
    # Convert white triangle detections
    big_triangles = [
        WhiteTriangleDetection(
            cell_id=wt["best_post"] + 1,
            triangle_type="big",
            center_of_mass=tuple(wt["center_of_mass"]),
            orientation=wt["orientation"],
            hull=wt["hull"]
        )
        for wt in white_triangles if wt["result"] == "big"
    ]
    
    small_triangle_detections = [
        WhiteTriangleDetection(
            cell_id=st["best_post"] + 1,
            triangle_type="small",
            center_of_mass=tuple(st["center_of_mass"]),
            orientation=st["orientation"],
            hull=st["hull"]
        )
        for st in small_triangles
    ]
    
    # Create camera parameters
    camera_params = CameraParameters(K=K, R=R_best, t=t_best, H=H)
    
    # Create intermediate images
    intermediate_imgs = IntermediateImages(
        original=result[0].orig_img,
        cropped=cropped_image,
        cropped_mask=cropped_mask,
        normalized=normalized_bgr,
        warped=warped,
        warped_no_bg=warped_no_bg
    )
    
    # Create and return result
    return BoardDetectionResult(
        pieces=piece_detections,
        white_triangles=white_triangles_id,
        min_pts=min_pts,
        corrected_points=corrected_points,
        corrected_points_ori=corrected_points_ori,
        corrections=corrections,
        white_corrections=white_corrections,
        camera=camera_params,
        images=intermediate_imgs,
        white_triangle_detections=big_triangles,
        small_triangles=small_triangle_detections,
        color_data=color_data,
        all_keypoints=all_keypoints,
        processing_time_ms=total_time * 1000
    )