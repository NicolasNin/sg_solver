"""
Test script to demonstrate creating a mask from YOLO pose keypoints.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from board_detection.yolo_helpers import create_mask_from_keypoints

# Load the YOLO model
model = YOLO("/home/nicolas/code/star_genius_solver/notebooks/runs/pose/train4/weights/best.pt")

# Test with an image
image_path = "/path/to/your/test/image.jpg"  # Replace with actual image path
results = model(image_path)

if len(results) > 0:
    result = results[0]
    
    # Create mask from keypoints with different expansion values
    for expansion in [0, 10, 20]:
        mask, keypoints = create_mask_from_keypoints(result, expansion_pixels=expansion)
        
        print(f"\nExpansion: {expansion} pixels")
        print(f"Mask shape: {mask.shape}")
        print(f"Keypoints shape: {keypoints.shape}")
        print(f"Number of keypoints: {len(keypoints)}")
        
        # Visualize the result
        # Create a copy of the original image
        vis_img = result.orig_img.copy()
        
        # Apply mask to show the masked region
        masked_img = cv2.bitwise_and(vis_img, vis_img, mask=mask)
        
        # Draw keypoints on the visualization
        for i, (x, y) in enumerate(keypoints):
            cv2.circle(vis_img, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(vis_img, str(i), (int(x) + 10, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw the polygon outline
        polygon_points = keypoints.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [polygon_points], True, (0, 0, 255), 2)
        
        # Save visualization
        cv2.imwrite(f"mask_visualization_exp{expansion}.jpg", vis_img)
        cv2.imwrite(f"masked_board_exp{expansion}.jpg", masked_img)
        cv2.imwrite(f"mask_only_exp{expansion}.jpg", mask)
        
        print(f"Saved: mask_visualization_exp{expansion}.jpg")
        print(f"Saved: masked_board_exp{expansion}.jpg")
        print(f"Saved: mask_only_exp{expansion}.jpg")
else:
    print("No board detected in the image!")
