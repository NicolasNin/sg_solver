import numpy as np
import cv2
from collections import deque
from board_detection.patch_classification import extract_patches
from board_detection.reference_data import CELLS_GRAPH


def cluster_graph(adj, values, threshold,distance):
    visited = set()
    clusters = []
    
    def too_far(u, v):
        return distance(values[u] , values[v]) > threshold
    for start in adj.keys():
        if start in visited:
            continue

        cluster = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            u = queue.popleft()
            cluster.add(u)

            for v in adj.get(u, []):
                if v in visited:
                    continue
                # only traverse edge if values are "close enough"
                if not too_far(u, v):
                    visited.add(v)
                    queue.append(v)
        clusters.append(cluster)

    return clusters   

def prune_graph(graph,node_to_remove):
    new_graph = {}
    for node in graph:
        if node in node_to_remove:
            continue
        l=[x for x in graph[node] if x not in node_to_remove]
        new_graph[node]=l
    return new_graph

def angle_diff(a, b, period=360):
    """
    Smallest distance between two angles a, b on a circle of given period.
    Works for radians (default, period = 2π) or degrees (period = 360).
    """
    d = abs(a - b) % period
    return min(d, period - d)
    
def distance_abs(a,b):
    return abs(a-b)


 #COLOR STUFF
def normalized_chroma(lab_img, reference_L=180):
    """Scale chroma as if L were at reference_L
    
    Intuition: darker pixels have 'compressed' chroma
    A dark red could be vivid red if brighter
    """
    L = lab_img[:,:,0].astype(np.float32)
    A = lab_img[:,:,1].astype(np.float32)
    B = lab_img[:,:,2].astype(np.float32)
    
    chroma = np.sqrt((A - 128)**2 + (B - 128)**2)
    
    # Scale factor: how much "room" does this L have for chroma?
    # At L=0 or L=255, max possible chroma is 0
    # At L~128, max possible chroma is highest
    # Simple model: scale by reference_L / L
    
    # Avoid div by zero, and don't over-boost very dark pixels
    L_safe = np.clip(L, 20, 255)
    
    scale = reference_L / L_safe
    
    # Cap the boost for very dark pixels
    scale = np.clip(scale, 0.5, 3.0)
    
    return chroma * scale

def compute_chroma(lab_img):
    """Chroma from AB channels"""
    A = lab_img[:,:,1].astype(np.float32)
    B = lab_img[:,:,2].astype(np.float32)
    return np.sqrt((A - 128)**2 + (B - 128)**2)

def circular_median(angles, period=2 * np.pi):
    """
    Compute the circular median of angles (radians by default).

    angles: iterable of numbers
    period: e.g. 2*np.pi for radians, 360.0 for degrees

    Returns:
        median angle in [0, period)
    """
    angles = np.asarray(angles, dtype=float).ravel()
    # normalize to [0, period)
    a = np.mod(angles, period)
    if len(a) == 0:
        raise ValueError("circular_median: empty input")

    # sort angles
    a_sorted = np.sort(a)

    # gaps between consecutive angles, including the wrap-around gap
    gaps = np.diff(np.concatenate([a_sorted, [a_sorted[0] + period]]))

    # find the largest gap: we will "cut" the circle there
    k = np.argmax(gaps)
    cut_angle = a_sorted[k] + gaps[k] / 2.0

    # rotate so that cut is at 0, then take regular median
    rotated = np.mod(a - cut_angle, period)
    med_rot = np.median(rotated)

    # rotate back
    med = np.mod(med_rot + cut_angle, period)
    return med

def compute_color_values(warped,corrected_points,patch_size=80):
    warped_blur = cv2.GaussianBlur(warped, (3, 3), 0)

    centers = [corrected_points[x] for x in corrected_points]
    warped_blur_lab  = cv2.cvtColor(warped_blur, cv2.COLOR_BGR2LAB)
    warped_blur_fakeHSV  = cv2.cvtColor(warped_blur_lab, cv2.COLOR_BGR2HSV)
    warped_blur_hsv  = cv2.cvtColor(warped_blur, cv2.COLOR_BGR2HSV)
    patches_lab = extract_patches(warped_blur_lab,centers,80) #a list of  (80, 80, 3) lab images
    patches_hsv = extract_patches(warped_blur_hsv,centers,80)
    patches_fake_hsv = extract_patches(warped_blur_fakeHSV,centers,patch_size)

    data_color={}
    for i in range(len(patches_lab)):
        patch_lab = patches_lab[i]
        L = patch_lab[:,:,0].astype(np.float32)
        A = patch_lab[:,:,1].astype(np.float32)-128
        B = patch_lab[:,:,2].astype(np.float32)-128
        theta = np.arctan2(B,A)/np.pi*180+180 #0-360
        
        patch_hsv =  patches_hsv[i]
        patch_fakehsv =  patches_hsv[i]
        patch_hue = patch_hsv[:,:,0]
        patch_fakehue = patch_fakehsv[:,:,0]

        patch_chroma = compute_chroma(patch_lab)
        patch_chroma_normed = normalized_chroma(patch_lab)
        data_color[i+1]={
            "L_med":np.median(L),
            "A_med":np.median(A),
            "B_med":np.median(B),
            "theta":theta,
            "theta_med":circular_median(theta,period=360.0),
            "chroma_med":np.median(patch_chroma),
            "chroma_normed_med":np.median(patch_chroma_normed),
            "hue_med":np.median(patch_hue),
            "fakehue_med":np.median(patch_fakehue)
        }
    return data_color
def distance_color_data(cd1,cd2):

    hue1 = cd1["hue_med"]
    theta1 = cd1["theta_med"]
    chroma1 =  cd1["chroma_med"]
    L1 =  cd1["L_med"]
    
    hue2 = cd2["hue_med"]
    theta2 = cd2["theta_med"]
    chroma2 =  cd2["chroma_med"]
    L2 =  cd2["L_med"]    
    
    delta_hue = abs(hue1 - hue2)
    delta_hue = min(delta_hue, 360 - delta_hue)

    delta_theta = abs(theta1 - theta2)
    delta_theta = min(delta_theta, 360 - delta_theta)

    min_chroma = min(chroma1,chroma2)
    delta_chroma = abs(chroma1 - chroma2)
    delta_L = abs(L1 - L2)

    feat = (delta_theta,delta_chroma,delta_L,delta_hue,min_chroma)
    #label = int(get_code_no_triangle(c1)==get_code_no_triangle(c2))
    
    return feat

def cluster_graph_values(values,threshold,distance,to_prune=None):
    #typically to prune are white triangles ids
    graph = CELLS_GRAPH
    graph_pruned = prune_graph(graph,to_prune)
    clusters = cluster_graph(graph_pruned, values, threshold,distance=distance)
    return clusters

def random_color(seed):
    import random
    random.seed(seed)
    return tuple(int(c) for c in random.choices(range(50, 255), k=3))

def visualize_clusters(debug_img, corrected_points,values, clusters, radius=5):
    """
    debug_img: base BGR image (will be modified in-place)
    corrected_points: list/array of (x, y) for each node index
    clusters: list[set[int]] or list[list[int]] from clustering
    radius: circle radius
    """
    img = debug_img  # or img = debug_img.copy() if you want to keep original

    # Generate one distinct color per cluster (BGR)
    rng = np.random.default_rng(42)  # fixed seed for reproducible colors
    colors = []
    for _ in range(len(clusters)):
        bgr = rng.integers(0, 255, size=3, dtype=np.uint8)
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))

    for cid, cluster in enumerate(clusters):
        color = colors[cid]
        for node in cluster:
            x, y = corrected_points[node]
            val = f"{int(values[node])}"
            # Draw the point
            cv2.circle(img, (int(x), int(y)), radius, color, thickness=-1)

            # Optional: label with cluster id
            t = val
            cv2.putText(img, t, (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    return img


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx

def cluster_graph_edges(adjacency, edge_values, threshold=0.5):
    uf = UnionFind()

    for u, neighbors in adjacency.items():
        uf.find(u)  # ensure node exists
        for v in neighbors:
            p = edge_values.get((u, v), 0.0)
            if p >= threshold:
                uf.union(u, v)

    # collect components
    clusters = {}
    for u in adjacency:
        root = uf.find(u)
        clusters.setdefault(root, []).append(u)

    return list(clusters.values())

def draw_cluster_mst(image, clusters, corrected_points):
    vis = image.copy()

    for k, cluster in enumerate(clusters):
        color = random_color(k)

        pts = [
            (n, tuple(map(int, corrected_points[n])))
            for n in cluster
            if n in corrected_points
        ]

        if not pts:
            continue

        pos = dict(pts)

        # ✅ ALWAYS draw points
        for p in pos.values():
            cv2.circle(vis, p, 6, color, -1)

        # ✅ Only draw edges if cluster size > 1
        if len(pos) == 1:
            continue

        # Prim's MST
        nodes = list(pos.keys())
        used = {nodes[0]}
        unused = set(nodes[1:])

        while unused:
            best = None
            best_d = 1e9
            for u in used:
                for v in unused:
                    d = (pos[u][0] - pos[v][0])**2 + (pos[u][1] - pos[v][1])**2
                    if d < best_d:
                        best = (u, v)
                        best_d = d

            u, v = best
            cv2.line(vis, pos[u], pos[v], color, 2)
            used.add(v)
            unused.remove(v)

    return vis
