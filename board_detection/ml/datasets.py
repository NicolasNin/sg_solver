"""
Unified dataset loading for ML training.

This module provides a consistent interface for loading board patches,
computing features, and preparing data for training various models.

TODO: Refactor to use board_pipeline_clean instead of the old board_pipeline
      for cleaner data flow (only run YOLO + warping, skip classification).
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Literal
import numpy as np
import cv2
import torch
import hashlib
from dataclasses import dataclass, field

from board_detection.link_roboflow_to_json import scan_images_with_json
from board_detection.yolo_helpers import get_yolo_model, extract_cropped_board
from board_detection.board_pipeline import board_pipelineYolo  # TODO: switch to board_pipeline_clean
from board_detection.graph_clustering import compute_color_values
from board_detection.patch_classification import extract_patches
from board_detection.reference_data import TRIANGLES_ORIENTATION
from board_detection.ml import config


# Color code mapping
COLOR_CODES = {
    "e": "empty", "br": "brown", "bl": "blue", "y": "yellow",
    "c": "cyan", "o": "orange", "g": "green", "l": "lime",
    "r": "red", "ro": "rose", "p": "purple", "w": "white"
}


def get_color_name(code: str) -> str:
    """Convert short code to full color name."""
    if code.endswith("2"):
        return COLOR_CODES.get(code[:-1], code) + "_triangle"
    return COLOR_CODES.get(code, code)


def get_color_name_no_triangle(code: str) -> str:
    """Convert short code to color name, ignoring triangle marker."""
    if code.endswith("2"):
        return COLOR_CODES.get(code[:-1], code[:-1])
    return COLOR_CODES.get(code, code)


# Known bad images to exclude
BAD_IMAGES = [
    "IMG_20260110_111510.jpg",
    "IMG_20260109_162811.jpg",
    "IMG_20260110_110415.jpg",
    "IMG_20260110_111019.jpg",
    "IMG_20260110_110202.jpg"
]


def is_bad_image(path: Path) -> bool:
    """Check if an image is in the known bad list."""
    if isinstance(path, str):
        path = Path(path)
    return path.name in BAD_IMAGES


@dataclass
class PatchData:
    """
    Data for a single patch (one cell on the board).
    
    Contains multiple representations for different model types.
    """
    file_path: str
    cell_id: int  # 0-indexed (0-47)
    
    # Raw patches
    patch_bgr: np.ndarray  # 80x80 BGR patch (original orientation)
    patch_oriented_bgr: np.ndarray  # Oriented for classification (up-pointing)
    
    # Different colorspaces for different models
    patch_oriented_lab: np.ndarray  # LAB version
    patch_oriented_gray: np.ndarray  # Grayscale version
    
    # Color statistics (for sklearn models)
    # Features: hue_med, L_med, A_med, B_med, chroma_med, theta_med
    color_stats: Dict[str, float]
    
    # Labels
    label_raw: str  # Original annotation like "e", "br", "bl2"
    label_color: str  # Color name without triangle: "empty", "brown", "blue"
    
    # Spatial info
    center: Tuple[int, int]  # Position in warped image
    orientation: str  # "up" or "down"


def compute_file_hash(file_path: str, algorithm: str = "md5") -> str:
    """Compute hash of a file for tracking purposes."""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ImageData:
    """Data for a single board image."""
    file_path: str
    file_hash: str  # MD5 hash of the original image file
    config: List[str]  # 48 labels
    patches: List["PatchData"]
    warped_image: np.ndarray
    corrected_points: Dict[int, Tuple[int, int]]


class TorchPatchDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for CNN training.
    
    Wraps PatchData objects and provides augmentation, jitter, and tensor conversion.
    Reusable across different CNN training scripts.
    
    Example:
        train_ds = TorchPatchDataset(
            patches=train_patches,
            board_dataset=dataset,
            colorspace="lab",
            label_type="white",
            transform=T.ColorJitter(...),
            patch_jitter=True
        )
    """
    
    def __init__(
        self,
        patches: List[PatchData],
        board_dataset: "BoardPatchDataset",
        colorspace: Literal["lab", "gray", "rgb"] = "lab",
        label_type: Literal["white", "empty", "pattern"] = "white",
        transform = None,
        patch_jitter: bool = False,
        max_jitter: int = 15
    ):
        """
        Args:
            patches: List of PatchData objects
            board_dataset: Parent dataset (for accessing warped images for jitter)
            colorspace: Output colorspace - "lab", "gray", or "rgb"
            label_type: What to predict - "white", "empty", or "pattern"
            transform: torchvision transforms for augmentation
            patch_jitter: Whether to randomly offset patch extraction
            max_jitter: Max pixels to jitter in each direction
        """
        self.patches = patches
        self.board_dataset = board_dataset
        self.colorspace = colorspace
        self.label_type = label_type
        self.transform = transform
        self.patch_jitter = patch_jitter
        self.max_jitter = max_jitter
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        import random
        patch_data = self.patches[idx]
        
        # Get patch (with optional jitter during training)
        if self.patch_jitter:
            img_bgr = self._get_jittered_patch(patch_data)
        else:
            img_bgr = patch_data.patch_oriented_bgr.copy()
        
        # Convert to RGB for transforms
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # To tensor (HWC -> CHW, normalize to [0,1])
        img_t = torch.from_numpy(img_rgb).float() / 255.0
        img_t = img_t.permute(2, 0, 1)  # CHW
        
        # Apply augmentation transforms
        if self.transform is not None:
            img_t = self.transform(img_t)
        
        # Convert to final colorspace
        if self.colorspace == "rgb":
            out = img_t
        elif self.colorspace == "gray":
            img_rgb_aug = (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            img_gray = cv2.cvtColor(img_rgb_aug, cv2.COLOR_RGB2GRAY)
            out = torch.from_numpy(img_gray).unsqueeze(0).float() / 255.0
        elif self.colorspace == "lab":
            img_rgb_aug = (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            img_lab = cv2.cvtColor(img_rgb_aug, cv2.COLOR_RGB2LAB)
            out = torch.from_numpy(img_lab).permute(2, 0, 1).float() / 255.0
        else:
            raise ValueError(f"Unknown colorspace: {self.colorspace}")
        
        # Get label
        raw_label = patch_data.label_raw
        if self.label_type == "white":
            label = torch.tensor(raw_label == "w", dtype=torch.float32)
        elif self.label_type == "empty":
            label = torch.tensor(raw_label == "e", dtype=torch.float32)
        elif self.label_type == "pattern":
            label = torch.tensor(raw_label.endswith("2"), dtype=torch.float32)
        else:
            raise ValueError(f"Unknown label_type: {self.label_type}")
        
        return out, label
    
    def _get_jittered_patch(self, patch_data: PatchData) -> np.ndarray:
        """Extract patch with random offset from center."""
        import random
        from board_detection.patch_classification import extract_patches
        
        image_data = self.board_dataset.images[patch_data.file_path]
        full_image = image_data.warped_image
        
        cx, cy = patch_data.center
        jitter_x = random.randint(-self.max_jitter, self.max_jitter)
        jitter_y = random.randint(-self.max_jitter, self.max_jitter)
        jittered_center = (cx + jitter_x, cy + jitter_y)
        
        patch = extract_patches(full_image, [jittered_center], config.PATCH_SIZE)[0]
        
        if patch_data.orientation == "down":
            patch = cv2.rotate(patch, cv2.ROTATE_180)
        
        return patch


class BoardPatchDataset:
    """
    Unified dataset for board patch classification.
    
    Handles loading images, running YOLO + warping pipeline, 
    extracting patches and computing features.
    
    Example:
        dataset = BoardPatchDataset()
        dataset.load_from_directory(Path("data/my_photos"))
        
        # For sklearn color comparison model
        X, y, images = dataset.get_color_distance_features()
        
        # For CNN empty detection
        X, y_empty, y_white = dataset.get_cnn_patches(colorspace="lab")
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize dataset.
        
        Args:
            cache_dir: Optional directory to cache processed data
        """
        self.cache_dir = cache_dir
        self.images: Dict[str, ImageData] = {}
        self._yolo_model = None
    
    @property
    def yolo_model(self):
        """Lazy load YOLO model."""
        if self._yolo_model is None:
            self._yolo_model = get_yolo_model()
        return self._yolo_model
    
    def load_from_directory(self, directory: Path, skip_bad: bool = True) -> int:
        """
        Load all images from a directory with JSON annotations.
        
        Args:
            directory: Path to directory with images and JSON files
            skip_bad: Whether to skip known bad images
            
        Returns:
            Number of images loaded
        """
        print(f"Scanning {directory}...")
        images_json = scan_images_with_json(directory)
        
        count = 0
        for image_path, json_info in images_json.items():
            if skip_bad and is_bad_image(image_path):
                print(f"  Skipping bad image: {Path(image_path).name}")
                continue
            
            try:
                self._load_single_image(image_path, json_info["json_data"])
                count += 1
                if count % 10 == 0:
                    print(f"  Loaded {count} images...")
            except Exception as e:
                print(f"  Error loading {image_path}: {e}")
        
        print(f"Loaded {count} images from {directory}")
        return count
    
    def load_unlabeled_directory(self, directory: Path, skip_bad: bool = True) -> int:
        """
        Load images from a directory WITHOUT requiring JSON annotations.
        
        Useful for inference on new/hard data where labels don't exist.
        Labels will be set to None for all patches.
        
        Args:
            directory: Path to directory with images
            skip_bad: Whether to skip known bad images
            
        Returns:
            Number of images loaded
        """
        print(f"Scanning unlabeled images in {directory}...")
        if isinstance(directory, str):
            directory = Path(directory)
        # Simple glob for image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(directory.glob(f"**/{ext}"))
        
        count = 0
        skipped_duplicates = 0
        
        for image_path in sorted(image_paths):
            image_path_str = str(image_path)
            
            # Skip duplicates (fast check)
            if image_path_str in self.images:
                skipped_duplicates += 1
                continue
            
            if skip_bad and is_bad_image(image_path):
                print(f"  Skipping bad image: {image_path.name}")
                continue
            
            try:
                self._load_single_image(image_path_str, config_data=None)
                count += 1
                if count % 10 == 0:
                    print(f"  Loaded {count} images...")
            except Exception as e:
                print(f"  Error loading {image_path}: {e}")
        
        if skipped_duplicates > 0:
            print(f"  Skipped {skipped_duplicates} duplicates (already loaded)")
        print(f"Loaded {count} unlabeled images from {directory}")
        return count
    
    def _load_single_image(self, image_path: str, config_data: Optional[List[str]] = None):
        """
        Process a single image and extract patch data.
        
        Args:
            image_path: Path to the image file
            config_data: List of 48 labels, or None for unlabeled data
        
        TODO: Refactor to use board_pipeline_clean for cleaner data flow.
              Currently uses the full pipeline which runs unnecessary classification.
        """
        # Run YOLO to detect board
        results = self.yolo_model(image_path)
        if len(results) == 0 or results[0].boxes is None:
            raise ValueError("No board detected")
        
        # Run pipeline to get warped image and corrected points
        # TODO: Use only the early steps from board_pipeline_clean:
        #   1. extract_cropped_board()
        #   2. normalize_L_mean()  
        #   3. warp_and_correct()
        #   4. white_triangles_and_correct() - optional
        board_result = board_pipelineYolo(image_path, do_correction=False)
        if board_result is None:
            raise ValueError("Pipeline failed")
        
        warped = board_result.images.warped_no_bg
        warped_blur = cv2.GaussianBlur(warped, (1, 1), 0)
        corrected_points = board_result.corrected_points
        
        # Extract patches
        centers = [corrected_points[i] for i in corrected_points]
        patches = extract_patches(warped_blur, centers, config.PATCH_SIZE)
        
        # Compute color features
        color_data = compute_color_values(warped, corrected_points, patch_size=config.PATCH_SIZE)
        
        # Build patch data
        patch_list = []
        for i in range(config.NUM_CELLS):
            # Get orientation - will KeyError if TRIANGLES_ORIENTATION is wrong
            # (intentionally not using .get() - we want it to fail loudly)
            orientation = TRIANGLES_ORIENTATION[i + 1]
            
            patch_bgr = patches[i]
            
            # Orient patch for classification
            if orientation == "up":
                patch_oriented = patch_bgr
            else:
                patch_oriented = cv2.rotate(patch_bgr, cv2.ROTATE_180)
            
            # Compute different colorspaces
            patch_lab = cv2.cvtColor(patch_oriented, cv2.COLOR_BGR2LAB)
            patch_gray = cv2.cvtColor(patch_oriented, cv2.COLOR_BGR2GRAY)
            
            cd = color_data[i + 1]
            
            # Handle labels (None for unlabeled data)
            if config_data is not None:
                label_raw = config_data[i]
                label_color = get_color_name_no_triangle(config_data[i])
            else:
                label_raw = None
                label_color = None
            
            patch_data = PatchData(
                file_path=image_path,
                cell_id=i,
                patch_bgr=patch_bgr,
                patch_oriented_bgr=patch_oriented,
                patch_oriented_lab=patch_lab,
                patch_oriented_gray=patch_gray,
                color_stats={
                    "hue_med": cd["hue_med"],
                    "L_med": cd["L_med"],
                    "A_med": cd["A_med"],
                    "B_med": cd["B_med"],
                    "chroma_med": cd["chroma_med"],
                    "theta_med": cd["theta_med"],
                },
                label_raw=label_raw,
                label_color=label_color,
                center=corrected_points[i + 1],
                orientation=orientation
            )
            patch_list.append(patch_data)
        
        # Compute file hash for tracking
        file_hash = compute_file_hash(image_path)
        
        self.images[image_path] = ImageData(
            file_path=image_path,
            file_hash=file_hash,
            config=config_data if config_data else [],
            patches=patch_list,
            warped_image=warped_blur,
            corrected_points=corrected_points
        )
    
    def get_all_patches(self) -> List[PatchData]:
        """Get all patches across all loaded images."""
        patches = []
        for img_data in self.images.values():
            patches.extend(img_data.patches)
        return patches
    
    def get_color_stats_features(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get color statistics features for patch classification (sklearn models).
        
        Features: [hue_med, L_med, A_med, B_med, chroma_med, theta_med]
        
        Returns:
            X: Feature matrix (N, 6)
            y: Labels (N,) - color names like "empty", "brown", "blue"
            file_paths: Source image for each patch
        """
        patches = self.get_all_patches()
        
        X = np.array([
            [
                p.color_stats["hue_med"],
                p.color_stats["L_med"],
                p.color_stats["A_med"],
                p.color_stats["B_med"],
                p.color_stats["chroma_med"],
                p.color_stats["theta_med"],
            ]
            for p in patches
        ])
        
        y = np.array([p.label_color for p in patches])
        file_paths = [p.file_path for p in patches]
        
        return X, y, file_paths
    
    def get_color_distance_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get features for patch pair comparison (same image only).
        
        Used by: RandomForest patch similarity model
        Features: [delta_theta, delta_chroma, delta_L, delta_hue, min_chroma]
        
        Returns:
            X: Feature matrix (N_pairs, 5)
            y: Labels (N_pairs,) - 1 if same color, 0 if different
            image_ids: Which image each pair came from
        """
        features = []
        labels = []
        image_ids = []
        
        for img_path, img_data in self.images.items():
            patches = img_data.patches
            
            # Compare all pairs within the same image
            for i in range(len(patches)):
                for j in range(i + 1, len(patches)):
                    p1, p2 = patches[i], patches[j]
                    
                    cd1 = p1.color_stats
                    cd2 = p2.color_stats
                    
                    # Compute distance features
                    delta_hue = abs(cd1["hue_med"] - cd2["hue_med"])
                    delta_hue = min(delta_hue, 360 - delta_hue)
                    
                    delta_theta = abs(cd1["theta_med"] - cd2["theta_med"])
                    delta_theta = min(delta_theta, 360 - delta_theta)
                    
                    delta_chroma = abs(cd1["chroma_med"] - cd2["chroma_med"])
                    delta_L = abs(cd1["L_med"] - cd2["L_med"])
                    min_chroma = min(cd1["chroma_med"], cd2["chroma_med"])
                    
                    feat = [delta_theta, delta_chroma, delta_L, delta_hue, min_chroma]
                    label = int(p1.label_raw == p2.label_raw)
                    
                    features.append(feat)
                    labels.append(label)
                    image_ids.append(img_path)
        
        return np.array(features), np.array(labels), np.array(image_ids)
    
    def get_cnn_patches(
        self, 
        colorspace: Literal["lab", "gray", "bgr"] = "lab"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get patches formatted for CNN training.
        
        Args:
            colorspace: Which colorspace to use
                - "lab": 3-channel LAB (default, good for color-invariant features)
                - "gray": 1-channel grayscale (for structure-only models)  
                - "bgr": 3-channel BGR (raw color)
        
        Returns:
            X: Tensor (N, C, 80, 80) - patches normalized to [0,1]
               C=3 for lab/bgr, C=1 for gray
            y_empty: Binary labels (N,) - 1 if empty
            y_white: Binary labels (N,) - 1 if white
        """
        patches = self.get_all_patches()
        
        if colorspace == "lab":
            X = np.array([p.patch_oriented_lab for p in patches])
            X = X.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        elif colorspace == "gray":
            X = np.array([p.patch_oriented_gray for p in patches])
            X = X[:, np.newaxis, :, :].astype(np.float32) / 255.0
        elif colorspace == "bgr":
            X = np.array([p.patch_oriented_bgr for p in patches])
            X = X.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unknown colorspace: {colorspace}. Use 'lab', 'gray', or 'bgr'")
        
        y_empty = np.array([int(p.label_color == "empty") for p in patches])
        y_white = np.array([int(p.label_color == "white") for p in patches])
        
        return X, y_empty, y_white
    
    def split_by_image(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[List[str], List[str]]:
        """
        Split images into train/test sets (no leakage across images).
        
        Returns:
            train_images: List of image paths for training
            test_images: List of image paths for testing
        """
        from sklearn.model_selection import train_test_split
        
        all_images = list(self.images.keys())
        train_imgs, test_imgs = train_test_split(
            all_images, 
            test_size=test_size, 
            random_state=random_state
        )
        return train_imgs, test_imgs
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __repr__(self) -> str:
        n_patches = sum(len(img.patches) for img in self.images.values())
        return f"BoardPatchDataset({len(self.images)} images, {n_patches} patches)"
    
    def get_image_hashes(self) -> Dict[str, str]:
        """
        Get mapping of image filenames to their hashes.
        
        Useful for logging to MLflow to track which exact images were used.
        
        Returns:
            Dict mapping filename (not full path) to MD5 hash
        """
        return {
            Path(img.file_path).name: img.file_hash
            for img in self.images.values()
        }
    
    def get_image_hash_summary(self) -> str:
        """
        Get a formatted string of image hashes for logging.
        
        Format: "filename: hash" per line, sorted by filename.
        """
        hashes = self.get_image_hashes()
        lines = [f"{name}: {hash_val}" for name, hash_val in sorted(hashes.items())]
        return "\n".join(lines)
    
    def save(self, path: Path):
        """
        Save processed dataset to disk for fast reloading.
        
        Note: This saves the processed patches and features, NOT the original images.
        The saved file includes image hashes so you can verify data integrity.
        
        Args:
            path: Path to save the dataset (will use .pkl extension)
        """
        import pickle
        
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pkl")
        
        # Create a serializable version (exclude YOLO model)
        save_data = {
            "images": self.images,
            "version": "1.0",
            "n_images": len(self.images),
            "n_patches": sum(len(img.patches) for img in self.images.values()),
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        
        print(f"Saved dataset to {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    @classmethod
    def load(cls, path: Path) -> "BoardPatchDataset":
        """
        Load a previously saved dataset.
        
        Args:
            path: Path to the saved dataset
            
        Returns:
            Loaded BoardPatchDataset
        """
        import pickle
        import io
        
        # Custom unpickler to fix module references
        # (handles cache saved from __main__ vs imported module)
        class FixedUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Remap classes to this module
                if name in ("ImageData", "PatchData"):
                    module = "board_detection.ml.datasets"
                return super().find_class(module, name)
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        with open(path, "rb") as f:
            save_data = FixedUnpickler(f).load()
        
        dataset = cls()
        dataset.images = save_data["images"]
        
        print(f"Loaded dataset from {path}: {dataset}")
        return dataset
    
    @classmethod
    def load_or_create(
        cls,
        cache_path: Path,
        data_dir: Path,
        skip_bad: bool = True,
        force_reload: bool = False
    ) -> "BoardPatchDataset":
        """
        Load from cache if available, otherwise create and cache.
        
        This is the recommended way to use the dataset for training:
        
            dataset = BoardPatchDataset.load_or_create(
                cache_path=Path("data/dataset_cache.pkl"),
                data_dir=Path("data/my_photos")
            )
        
        Args:
            cache_path: Path to cache file
            data_dir: Directory with annotated images (used if cache doesn't exist)
            skip_bad: Whether to skip known bad images
            force_reload: If True, recompute even if cache exists
            
        Returns:
            Loaded or newly created dataset
        """
        cache_path = Path(cache_path)
        
        if cache_path.exists() and not force_reload:
            print(f"Loading cached dataset from {cache_path}")
            return cls.load(cache_path)
        
        print(f"No cache found at {cache_path}, computing from scratch...")
        dataset = cls()
        dataset.load_from_directory(data_dir, skip_bad=skip_bad)
        dataset.save(cache_path)
        
        return dataset


def main():
    """
    Generate dataset cache from command line.
    
    Usage:
        python -m board_detection.ml.datasets
        python -m board_detection.ml.datasets --output data/my_cache.pkl
        python -m board_detection.ml.datasets --force  # overwrite existing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate dataset cache")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/nicolas/code/star_genius_solver/data/my_photos/"),
        help="Directory with annotated images"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/nicolas/code/star_genius_solver/data/dataset_cache.pkl"),
        help="Output cache file path"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cache file"
    )
    
    args = parser.parse_args()
    
    # Safety check: don't overwrite without --force
    if args.output.exists() and not args.force:
        print(f"❌ Cache file already exists: {args.output}")
        print("   Use --force to overwrite, or specify a different --output path")
        return
    
    print(f"Loading data from {args.data_dir}...")
    dataset = BoardPatchDataset()
    dataset.load_from_directory(args.data_dir, skip_bad=True)
    
    dataset.save(args.output)
    print(f"✅ Cache saved to {args.output}")


if __name__ == "__main__":
    main()
