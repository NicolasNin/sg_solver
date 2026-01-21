"""
Dataclasses for storing intermediate results from the board detection pipeline.
Enables debugging and visualization of each processing step.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np


@dataclass
class YoloResult:
    """YOLO pose detection outputs."""
    all_keypoints: np.ndarray       # Full 12-point keypoints (in cropped coords)
    min_pts: np.ndarray             # 6 star corner points (in cropped coords)
    original_image: np.ndarray      # Original image before cropping
    bbox_xyxy: np.ndarray           # Bounding box [x1, y1, x2, y2] in original coords
    crop_offset: Tuple[int, int]    # (x_offset, y_offset) used to translate keypoints
    
    @property
    def all_keypoints_original(self) -> np.ndarray:
        """Get all keypoints in original image coordinates."""
        return self.all_keypoints + np.array(self.crop_offset)
    
    @property
    def min_pts_original(self) -> np.ndarray:
        """Get min_pts in original image coordinates."""
        return self.min_pts + np.array(self.crop_offset)


@dataclass
class ImageData:
    """All intermediate image representations."""
    filename: str
    cropped: np.ndarray             # Raw crop from YOLO bbox
    cropped_mask: np.ndarray        # Binary mask from keypoints
    normalized: np.ndarray          # L-channel normalized
    no_bg: np.ndarray               # Background removed
    warped: np.ndarray              # Perspective-corrected


@dataclass
class WarpingData:
    """Homography and pose estimation results."""
    H: np.ndarray                   # Homography matrix
    R: np.ndarray                   # Rotation matrix
    t: np.ndarray                   # Translation vector
    K: np.ndarray                   # Camera intrinsics
    corrected_points: Dict[int, Tuple[int, int]]  # Triangle centroids after height correction
    corrections: Dict[int, Tuple[int, int]]       # Applied corrections per point


@dataclass
class WhiteTrianglesData:
    """White blocker triangle detection results."""
    white_triangles: List[dict]     # Full detection info per triangle
    white_triangles_id: List[int]   # IDs of detected blockers (1-48)
    white_corrections: Dict[int, Tuple[int, int]]
    final_corrected_points: Dict[int, Tuple[int, int]]  # Points after white triangle refinement


@dataclass
class ClassifierData:
    """All classifier outputs for debugging."""
    patches_oriented: List[np.ndarray]  # 48 oriented patches
    color_data: np.ndarray          # Color features per cell
    edges_values: np.ndarray        # Pairwise color comparisons
    clusters: list[list]            # a list of clusters from edges values alone
    clusters_connected:list[list]     # an array of labels from 0 to N N numbers of clusters
    probs_empty: np.ndarray         # P(empty) per cell
    probs_white: np.ndarray         # P(white blocker) per cell
    Q: np.ndarray                   # Combined probability matrix
    y_hat: np.ndarray               # Final classification (0=empty, 1=white, 2=colored)
    color_predict: np.ndarray       # Color predictions for colored cells


@dataclass
class BoardResult:
    """Complete pipeline output with all intermediate results."""
    yolo: YoloResult
    images: ImageData
    warping: WarpingData
    white_triangles: WhiteTrianglesData
    classification: ClassifierData
    final_results: List[str]        # 48-element board state: 'e', 'w', or color
    identified_pieces: List[Tuple[str, int, int, List[int], str]] = None  # [(piece_name, orientation, anchor, cell_ids, color), ...]
    
    # Color mapping for visualization (BGR format)
    _RESULT_COLORS = {
        'e': (0, 0, 0),           # Black for empty
        'w': (255, 255, 255),     # White for blocker
        'bl': (180, 50, 0),       # Dark blue
        'c': (255, 255, 0),       # Cyan
        'c2': (255, 255, 0),      # Cyan with pattern
        'o': (0, 165, 255),       # Orange
        'o2': (0, 165, 255),      # Orange with pattern
        'l': (0, 255, 128),       # Lime (light lime)
        'l2': (0, 255, 128),      # Lime with pattern
        'g': (0, 180, 0),         # Green
        'g2': (0, 180, 0),        # Green with pattern
        'p': (180, 0, 180),       # Purple
        'p2': (180, 0, 180),      # Purple with pattern
        'br': (40, 80, 140),      # Brown
        'br2': (40, 80, 140),     # Brown with pattern
        'r': (0, 0, 255),         # Red
        'r2': (0, 0, 255),        # Red with pattern
        'ro': (180, 105, 255),    # Rose
        'ro2': (180, 105, 255),   # Rose with pattern
        'y': (0, 255, 255),       # Yellow
        'y2': (0, 255, 255),      # Yellow with pattern
    }
    
    def visualize_final(self, radius: int = 15, thickness: int = -1) -> np.ndarray:
        """
        Visualize final_results on the warped image at final_corrected_points.
        
        Args:
            radius: Circle radius for each cell marker
            thickness: Circle thickness (-1 for filled)
            
        Returns:
            BGR image with classification results overlaid
        """
        import cv2
        
        img = self.images.warped.copy()
        points = self.white_triangles.final_corrected_points
        
        for cell_id, (x, y) in points.items():
            result = self.final_results[cell_id - 1]  # IDs are 1-indexed
            color = self._RESULT_COLORS.get(result, (128, 128, 128))
            
            # Draw filled circle with border
            cv2.circle(img, (x, y), radius, color, thickness)
            cv2.circle(img, (x, y), radius, (0, 0, 0), 1)  # Black border
            
            # Add cell ID text
            cv2.putText(img, str(result), (x - 8, y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img
    
    def visualize_clusters(self,kind="edges") -> np.ndarray:
        """
        Visualize edge-based clusters on the warped image using MST.
        
        Returns:
            BGR image with clusters drawn as connected components
        """
        from board_detection.graph_clustering import draw_cluster_mst
        if kind =="edges":
            clusters = self.classification.clusters
        else:
            clusters = self.classification.clusters_connected
        return draw_cluster_mst(
            self.images.warped,
            clusters,
            self.white_triangles.final_corrected_points
        )
    
    def visualize_original_bbox(self) -> np.ndarray:
        """
        Visualize YOLO detection on original image: bbox and all keypoints.
        
        Returns:
            BGR image with bounding box and keypoints drawn
        """
        import cv2
        
        img = self.yolo.original_image.copy()
        
        # Draw bounding box
        x1, y1, x2, y2 = self.yolo.bbox_xyxy.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw all keypoints in original coords
        keypoints = self.yolo.all_keypoints_original
        for i, (x, y) in enumerate(keypoints):
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(img, str(i), (int(x) + 5, int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return img
    
    def visualize_cropped_keypoints(self) -> np.ndarray:
        """
        Visualize keypoints on the cropped image.
        
        Returns:
            BGR image with all keypoints drawn
        """
        import cv2
        
        img = self.images.cropped.copy()
        
        # Draw all keypoints
        for i, (x, y) in enumerate(self.yolo.all_keypoints):
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(img, str(i), (int(x) + 5, int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Highlight min_pts (star corners) in different color
        for i, (x, y) in enumerate(self.yolo.min_pts):
            cv2.circle(img, (int(x), int(y)), 8, (255, 0, 0), 2)
        
        return img
    
    def visualize_warped_keypoints(self) -> np.ndarray:
        """
        Visualize keypoints transformed to warped image coordinates.
        
        Returns:
            BGR image with warped keypoints drawn
        """
        import cv2
        
        img = self.images.warped.copy()
        
        # Transform min_pts using homography
        pts = self.yolo.min_pts.reshape(-1, 1, 2).astype(np.float32)
        warped_pts = cv2.perspectiveTransform(pts, self.warping.H)
        warped_pts = warped_pts.reshape(-1, 2)
        
        # Draw warped keypoints
        for i, (x, y) in enumerate(warped_pts):
            cv2.circle(img, (int(x), int(y)), 8, (0, 255, 0), -1)
            cv2.putText(img, str(i), (int(x) + 5, int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
