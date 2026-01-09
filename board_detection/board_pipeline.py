import cv2
import numpy as np
from board_detection.reference_data import *
from board_detection.mask_helpers import *
from board_detection.color_helpers import *
from board_detection.anchor_detector import *
from board_detection.anchor_helpers import *
from board_detection.white_triangles import *
from board_detection.graph_clustering import *
from board_detection.piece_identification import *

import time
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