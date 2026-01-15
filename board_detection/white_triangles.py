import numpy as np
import cv2
import math
from board_detection.reference_data import mask_logo
def get_white_mask(image_bgr,clean=True):
    image_hsv = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)
    lower = (0, 0, 150)
    upper = (180, 60, 255)
    mask_white = cv2.inRange(image_hsv, lower, upper)
    # clean the mask a bit
    if clean:
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    return mask_white

def equilateralness_from_sides(pts, tol_ratio=0.15):
    pts = np.asarray(pts, dtype=np.float32).reshape(3, 2)
    d = np.array([
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[1] - pts[2]),
        np.linalg.norm(pts[2] - pts[0]),
    ])
    mean_d = d.mean()
    if mean_d == 0:
        return 0.0
    rel_std = d.std() / mean_d
    if rel_std >= tol_ratio:
        return 0.0
    return float(1.0 - rel_std / tol_ratio)


def triangularity(cnt,return_inv=False):
    #T :equilateral==1 other triangle less than 1
    A = cv2.contourArea(cnt)
    P = cv2.arcLength(cnt, True)
    if P == 0:
        return 0.0,0,0
    T = 12 * math.sqrt(3) * A / (P * P)
    #print(A,P,T)
    # clip to [0, 1] in case of noise / numerical issues
    if return_inv:
        return float(1-np.abs(1-T)),A,P
    return float(1-np.abs(1-T)),A,P

def vertex_count_score(n, target=3, max_extra_vertices=5):
    if n <= target:
        return 1.0 if n == target else 0.0
    # linearly decrease from 1 to 0 as vertices go from 3 to 3+max_extra_vertices
    extra = n - target
    if extra >= max_extra_vertices:
        return 0.0
    return max(0.0, 1.0 - extra / max_extra_vertices)

def triangle_confidence(cnt):
    T,A,P = triangularity(cnt,return_inv=True)
    approx = cv2.approxPolyDP(cnt, 0.06 * P, True)
    n = len(approx)
    if n==3:
        eq = equilateralness_from_sides(approx)
    else:
        eq = 0
    v_score = vertex_count_score(n)
    conf = math.sqrt(T * v_score)
    
    return float(np.clip(conf, 0.0, 1.0)),P,A,n,eq


#helper
def filter_triangles_big_or_small(c,P,A,n,eq):
    # big triangle P~330 A~6000
    # small triangl P~220 A~2400
    #print(f"c {c:.2f},P {P:.2f},A {A:.2f},n {n},eq {eq:.2f}")
    #print(c,P,A,n,eq,"A-6000",abs(A-6000),"A-2550",abs(A-2550),(P>300 and abs(A-6000)<2500) or(P>180 and abs(A-2550)<800))
    if c>0.75 and eq>0.5 and abs(100*A/(P*P)-5.5)<0.5:
        if P>250:
            return "big" 
        else:
            return "small"         
    if c>0.75 and eq>0.5:
        if P>270 and A<9500:
            return "big"
        elif P>180 and abs(A-2550)<800:
            return "small" 
        elif P>180 and P<270 and abs(A-3000)<1000:
            return "small_large" 
    return "None"    

def count_triangles_big_small(hulls):
    results = []
    for k,cnt in enumerate(hulls):
        c,P,A,n,eq = triangle_confidence(cnt)
        results.append(filter_triangles_big_or_small(c,P,A,n,eq))
        print(k,c,"P",P,"A",A,"A/P^2",round(100*A/(P*P),2),n,"eq",eq,"A-6000",abs(A-6000),"A-2550",abs(A-2550),(P>300 and abs(A-6000)<2500) or(P>180 and abs(A-2550)<800),
        results[-1])
    return results

def get_center_of_mass_and_orientation(hull):
    M = cv2.moments(hull)
    x, y, w, h = cv2.boundingRect(hull)
    bbox_cx = x + w / 2.0
    bbox_cy = y + h / 2.0
    
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        center = (cx, cy)
        dy = cy - bbox_cy
        orientation = "up" if dy>0 else "down"
        return center,orientation
    return None


def find_best_matching_centers(centers_of_mass,centers):
    #centers is REF
    if len(centers_of_mass)==0:
        return -1
    diff = centers[:, np.newaxis, :] - centers_of_mass[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    #np.argmin(np.linalg.norm(np.array(centers)-centers_of_mass[0][0],axis=1))
    return np.argmin(distances,axis=0)


def find_white_triangles(warped_no_bg,corrected_points,blur_level=1):
    #need mask_logo
    if isinstance(corrected_points, dict):
        corrected_points = np.array([corrected_points[i] for i in corrected_points])
    else:
        corrected_points = np.array(corrected_points)
    warped_no_bg_no_logo = cv2.bitwise_and(warped_no_bg, warped_no_bg, mask=~mask_logo)

    #find white and create gray
    mask_white =  get_white_mask(warped_no_bg_no_logo)
    roi = cv2.bitwise_and(warped_no_bg_no_logo, warped_no_bg_no_logo, mask=mask_white)
    roi_gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    
    #find contours and hulls
    
    roi_gray = cv2.GaussianBlur(roi_gray, (blur_level, blur_level), 0)
    contours,_ = cv2.findContours(roi_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hulls = [cv2.convexHull(cnt) for cnt in contours]

    #rough filter to remove extra small or large contours
    hulls_area = [cv2.contourArea(cnt) for cnt in hulls]    
    idx_filter = [i for i,h_area in enumerate(hulls_area) if abs(h_area-6000)<5000]
    hulls_filter = [hulls[i] for i in idx_filter]
    if len(hulls_filter)==0:
        return {}
        
    results_big_small = count_triangles_big_small(hulls_filter)
    #find best position for the piece
    centers_of_mass = [get_center_of_mass_and_orientation(h) for h in hulls_filter]
    centers_of_mass_pts = np.array([x[0] for x in centers_of_mass])
    best_pos = find_best_matching_centers(centers_of_mass_pts,corrected_points)
    result_dic = [{
        "result":results_big_small[i],
        "orientation":centers_of_mass[i][1],
        "center_of_mass":centers_of_mass_pts[i],
        "best_post":best_pos[i],
        "hull":hulls_filter[i]
        } for i in range(len(hulls_filter))]
    return result_dic