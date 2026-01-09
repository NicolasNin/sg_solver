import cv2
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# Load SAM model from models/ at project root
_MODEL_PATH = Path(__file__).parent.parent / "models" / "sam_vit_b_01ec64.pth"
_sam = sam_model_registry["vit_b"](checkpoint=str(_MODEL_PATH))
_sam.to("cuda")
#predictor = SamPredictor(sam)

def get_sam():
    return _sam

def compute_sam(image_path,sam):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_generator = SamAutomaticMaskGenerator(
    model=sam, 
    points_per_side=16,  # Reduced from 32 (fewer sample points)
    pred_iou_thresh=0.92,  # Higher threshold (more confident masks only)
    stability_score_thresh=0.92,  # Higher (more stable masks)
    crop_n_layers=0,  # Don't crop (faster, finds larger objects)
    crop_n_points_downscale_factor=1,
    min_mask_region_area=5000,  # IMPORTANT: minimum pixels (filters out small stuff)
    )

    masks = mask_generator.generate(image)
    masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
    return masks_sorted

