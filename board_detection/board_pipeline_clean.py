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
from board_detection.yolo_helpers import get_yolo_model,extract_cropped_board
from board_detection.board_solver import *
import time
from ultralytics import YOLO

def get_yolo_results(image_path):
    print("Run custom yolo pose")
    yolo_model = get_yolo_model()
    results = yolo_model(image_path)
    print(len(results))
    if len(results)==0:
        print("No board detected")
        return None
    return results[0]


def find_rotations_from_H(H,status,image,min_pts):
    min_pts_reshape =np.array(min_pts).astype(np.float32).reshape(-1, 2).astype(np.float32)
    min_pt_refs_reshape =np.array(MIN_PTS_REF).astype(np.float32).reshape(-1, 2).astype(np.float32)
    img_height,img_width = image.shape[:2]
    focal_length_px = img_width *1  # Rough guess for ~50-60° FOV
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
    return R_best,t_best,K

def warp_and_correct(image_bgr,min_pts,cropped_mask_int,do_correction=True):


    warp_method = cv2.RANSAC
    warp_threshold = 5.0
    warped_no_bg,( H, status) = _warp_images(min_pts,image_bgr,MIN_PTS_REF,warp_method,warp_threshold,return_homography=True)
    R_best,t_best, K = find_rotations_from_H(H,status,image_bgr,min_pts)
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
    return warped_no_bg, corrected_points, corrections, H, R_best, t_best, K

def white_triangles_and_correct(warped,corrected_points):
    white_triangles = find_white_triangles(warped,corrected_points)
    white_triangles_id = [x["best_post"]+1 for x in white_triangles if x["result"]=="big"] # id start at 0
    small_triangles = [x for x in white_triangles if x["result"]=="small" or x["result"]=="small_large"]
    big_triangles = [x for x in white_triangles if x["result"]=="big"]
    white_corrections = {}
    corrected_points = corrected_points.copy() 
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
    return corrected_points,white_corrections,white_triangles,white_triangles_id

def get_patches_oriented(img,corrected_points,size=80):
    centers = [corrected_points[x] for x in corrected_points]
    patches_oriented = []
    patches = extract_patches(img,centers,size)
    for i,p in enumerate(patches):
        orientation = TRIANGLES_ORIENTATION[i+1]
        patches_oriented.append(patches[i] if  orientation == "up" else cv2.rotate(patches[i], cv2.ROTATE_180))
    return patches_oriented


from board_detection.pipeline_types import (
    YoloResult, ImageData, WarpingData, WhiteTrianglesData, 
    ClassifierData, BoardResult
)


def get_results(y_hat,color_predict):
    results =[]
    for i in range(48):
        if y_hat[i]==0:
            results.append("e")
        elif y_hat[i]==1:
            results.append("w")
        else:
            results.append(color_predict[i])
    return results

def board_pipelineYolo2(image_path, do_correction=False) -> Optional[BoardResult]:
    """
    Full board detection pipeline returning structured intermediate results.
    
    Returns:
        BoardResult with all intermediate data for debugging/visualization,
        or None if YOLO detection fails.
    """
    print("Running full pipeline")
    result = get_yolo_results(image_path)
    if result is None:
        return None
    print(result.boxes)
    # YOLO extraction
    (cropped_image, min_pts, cropped_mask_int, all_keypoints, 
     orig_img, bbox_xyxy, crop_offset) = extract_cropped_board(
        result, padding=10, expansion_pixels=5
    )
    
    # Image normalization
    cropped_image_normalized, _ = normalize_L_mean(
        cropped_image, cropped_mask_int, target_L_mean=100
    )
    image_no_bg = cv2.bitwise_and(
        cropped_image_normalized, cropped_image_normalized, mask=cropped_mask_int
    )
    
    # Warping and perspective correction
    warped, corrected_points_ori, corrections, H, R, t, K = warp_and_correct(
        image_no_bg, min_pts, cropped_mask_int, do_correction
    )
    
    # White triangle detection
    corrected_points, white_corrections, white_triangles, white_triangles_id = (
        white_triangles_and_correct(warped, corrected_points_ori)
    )
    
    # Feature extraction
    patches_oriented = get_patches_oriented(warped, corrected_points, size=80)
    color_data = compute_color_values(warped, corrected_points)
    
    # Classification
    edges_values = compute_all_compare(color_data)
    empty_patches, probs_empty = classify_patches_empty(patches_oriented)
    probs_dict = classify_patches_multi_from_oriented(patches_oriented)
    
    # clusters from edges classication on graph might not be useful but useful for vis
    clusters_edges = cluster_graph_edges_mat(CELLS_GRAPH,edges_values)
    n_components, clusters_connected = connected_components(edges_values>0.8)
    clusters_connected = convert_clusters_label_list(clusters_connected)
    # Merge classifiers
    pW = rescale_pw(probs_dict["white"])
    Q, y_hat = compute_empty_white_colored(probs_empty, pW, edges_values)
    
    # Color prediction
    white_ids = np.where(y_hat == 1)[0]
    normalized_lab = compute_normalized_lab(color_data, white_ids)
    color_predict = color_model.predict(normalized_lab)
    
    # Final results
    final_results = get_results(y_hat, color_predict)
    
    # Identify pieces from color groups
    identified_pieces = identify_pieces_from_results(final_results)
    
    # Build dataclasses
    yolo_data = YoloResult(
        all_keypoints=all_keypoints,
        min_pts=min_pts,
        original_image=orig_img,
        bbox_xyxy=bbox_xyxy,
        crop_offset=crop_offset
    )
    
    image_data = ImageData(
        filename=image_path if isinstance(image_path, str) else str(image_path),
        cropped=cropped_image,
        cropped_mask=cropped_mask_int,
        normalized=cropped_image_normalized,
        no_bg=image_no_bg,
        warped=warped
    )
    
    warping_data = WarpingData(
        H=H,
        R=R,
        t=t,
        K=K,
        corrected_points=corrected_points_ori,
        corrections=corrections
    )
    
    white_triangles_data = WhiteTrianglesData(
        white_triangles=white_triangles,
        white_triangles_id=white_triangles_id,
        white_corrections=white_corrections,
        final_corrected_points=corrected_points
    )
    
    classifier_data = ClassifierData(
        patches_oriented=patches_oriented,
        color_data=color_data,
        edges_values=edges_values,
        clusters=clusters_edges,
        clusters_connected=clusters_connected,
        probs_empty=probs_empty,
        probs_white=pW,
        Q=Q,
        y_hat=y_hat,
        color_predict=color_predict
    )
    
    return BoardResult(
        yolo=yolo_data,
        images=image_data,
        warping=warping_data,
        white_triangles=white_triangles_data,
        classification=classifier_data,
        final_results=final_results,
        identified_pieces=identified_pieces
    )


def identify_pieces_from_results(final_results: list[str]) -> list[tuple]:
    """
    Identify pieces from final_results by grouping cells of the same color.
    
    Args:
        final_results: 48-element list where each element is 'e', 'w', or a color code
        
    Returns:
        List of (piece_name, orientation, anchor_cell_id, cell_ids, color) tuples.
        piece_name is None if the shape couldn't be identified.
    """
    # Group cells by color (excluding 'e' empty and 'w' white)
    color_groups = {}
    for i, res in enumerate(final_results):
        if res not in ('e', 'w'):
            cell_id = i + 1  # Cell IDs are 1-indexed
            if res not in color_groups:
                color_groups[res] = []
            color_groups[res].append(cell_id)
    
    # Identify each color group
    identified = []
    for color, cell_ids in color_groups.items():
        piece_name, orientation, anchor = identify_piece_from_cells(cell_ids)
        identified.append((piece_name, orientation, anchor, cell_ids, color))
        print(f"[DEBUG] Color '{color}' cells {cell_ids} -> piece={piece_name}, orient={orientation}, anchor={anchor}")
    
    return identified