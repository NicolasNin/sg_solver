import cv2
import numpy as np
import pandas as pd
from typing import Optional, Dict

def add_file_ids(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'file_id' column to the dataframe mapping unique file paths to integers.
    Also returns the mapping dicts for reference if needed (by printing or separate call).
    
    This helps simplify the analysis by using short IDs instead of long paths.
    """
    if "file" not in results_df.columns:
        return results_df
        
    unique_paths = sorted(results_df["file"].unique())
    file2id = {p: i for i, p in enumerate(unique_paths)}
    
    # Create copy to avoid SettingWithCopy warnings if slice
    df = results_df.copy()
    df["file_id"] = df["file"].map(file2id)
    
    print(f"Mapped {len(unique_paths)} files to IDs 0-{len(unique_paths)-1}")
    return df

def create_patch_grid(
    results_df: pd.DataFrame, 
    dataset, 
    max_images: int = 50,
    uncertainty_threshold: float = 0.1,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a grid visualization of model predictions.
    
    Layout: one row per image (48 patches).
    Visuals:
        - Red border: prediction error
        - Yellow border: uncertain prediction (0.5 +/- threshold)
        - Text: predicted probability
        
    Args:
        results_df: DataFrame containing 'file', 'cell_id', 'pred_prob', 'is_error'
        dataset: The BoardPatchDataset containing source images/patches
        max_images: Maximum number of rows to generate to avoid huge images
        uncertainty_threshold: Range around 0.5 to flag as uncertain (e.g. 0.1 means 0.4-0.6)
        output_path: If provided, save the image to this path
        
    Returns:
        The generated grid image (BGR format for cv2, or convert to RGB for matplotlib)
    """
    # Ensure file_ids exist
    if "file_id" not in results_df.columns:
        results_df = add_file_ids(results_df)
        
    # Select images to visualize
    unique_ids = sorted(results_df["file_id"].unique())
    if len(unique_ids) > max_images:
        print(f"Truncating visualization: showing first {max_images} out of {len(unique_ids)} images.")
        unique_ids = unique_ids[:max_images]
    
    rows = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    print(f"Generating grid for {len(unique_ids)} images...")
    
    for fid in unique_ids:
        # Get predictions for this file
        file_df = results_df[results_df.file_id == fid].sort_values("cell_id")
        
        # Get original image data
        # We assume the user has the dataset loaded and it matches the result_df
        if file_df.empty:
            continue
            
        file_path = file_df.iloc[0]["file"]
        
        # Check if file is in dataset
        if file_path not in dataset.images:
            print(f"Warning: File {file_path} not found in dataset. Skipping row.")
            continue
            
        # Get the 48 patches for this image
        # Note: We rely on the order in dataset.images[path].patches matching cell_id 0..47
        patches_data = dataset.images[file_path].patches
        
        row_imgs = []
        for i, patch_data in enumerate(patches_data):
            # Find the corresponding result row
            # (Assuming cell_id matches index, which it should)
            row_res = file_df[file_df.cell_id == i]
            
            if row_res.empty:
                # Should not happen if df is complete, but fill with black/empty if missing
                img = np.zeros_like(patches_data[0].patch_oriented_bgr)
            else:
                row_res = row_res.iloc[0]
                img = patch_data.patch_oriented_bgr.copy()
                
                # Visuals
                prob = row_res.get("pred_prob", 0.0)
                is_error = row_res.get("is_error", False)
                
                # 1. BORDERS
                border_color = None
                thickness = 1
                
                if is_error:
                    border_color = (0, 0, 255) # Red
                    thickness = 4
                elif abs(prob - 0.5) < uncertainty_threshold:
                    border_color = (0, 255, 255) # Yellow
                    thickness = 2
                
                if border_color:
                    # Draw inner rectangle to not lose pixels on edge of grid
                    h, w = img.shape[:2]
                    cv2.rectangle(img, (0, 0), (w-1, h-1), border_color, thickness)
                
                # 2. PROBABILITY TEXT
                text = f"{prob:.2f}"
                # Background strip for readability
                cv2.rectangle(img, (0,0), (32, 14), (0,0,0), -1) 
                cv2.putText(img, text, (1, 11), font, 0.35, (255, 255, 255), 1)
            
            row_imgs.append(img)
            
        # Concatenate this row
        if row_imgs:
            full_row = np.hstack(row_imgs)
            rows.append(full_row)
    
    if not rows:
        print("No rows generated.")
        return None
        
    # Stack all rows
    final_grid = np.vstack(rows)
    
    if output_path:
        cv2.imwrite(output_path, final_grid)
        print(f"Saved visualization to {output_path}")
        
    return final_grid


def visualize_attention_grid(
    model, 
    dataset, 
    results_df: pd.DataFrame,
    num_samples: int = 16, 
    filter_type: str = "error",
    threshold: Optional[float] = None,
    output_path: Optional[str] = None
):
    """
    Visualize attention masks for specific samples (e.g. errors).
    
    Args:
        model: PyTorch model (must return (output, attention) when return_attention=True)
        dataset: BoardPatchDataset
        results_df: DataFrame with predictions and 'file'/'cell_id' columns
        num_samples: Number of samples to show
        filter_type: "error", "fp" (false pos), "fn" (false neg), or "random"
        output_path: Path to save the plot
    """
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from board_detection.ml.datasets import TorchPatchDataset
    from torch.utils.data import DataLoader
    
    # 1. Select candidates based on filter
    if filter_type == "error":
        candidates = results_df[results_df.is_error]
    elif filter_type == "fp": # Predicted white (1), but False (0) -> is_error & pred_prob > 0.5
        candidates = results_df[results_df.is_error & (results_df.pred_prob > 0.5)]
    elif filter_type == "fn":
        candidates = results_df[results_df.is_error & (results_df.pred_prob <= 0.5)]
    elif filter_type == "threshold":
        if threshold is None:
            threshold = 0.45
        candidates = results_df[abs(results_df.pred_prob -0.5)< threshold]
    else:
        candidates = results_df
        
    if candidates.empty:
        print(f"No candidates found for filter '{filter_type}'")
        return

    # Sample randomly
    if len(candidates) > num_samples:
        candidates = candidates.sample(num_samples)
    else:
        num_samples = len(candidates)
        
    # 2. Prepare data for model
    # We need to construct a mini-dataset just for these candidates to use standard transforms
    # However, TorchPatchDataset takes a list of PatchData. We can extract them.
    selected_patches = []
    
    # Using a loop here is fine for small num_samples
    for _, row in candidates.iterrows():
        # Find patch in dataset
        if row["file"] in dataset.images:
             # cell_id matches index
            patch = dataset.images[row["file"]].patches[row["cell_id"]]
            selected_patches.append(patch)
            
    if not selected_patches:
        print("Could not retrieve patches from dataset.")
        return

    # Create temporary loader
    # Note: We need to know colorspace used by model. Assuming "lab" or passing it would be better.
    # For visualization, we'll try to guess or default to 'lab' (most common here)
    temp_ds = TorchPatchDataset(selected_patches, dataset, colorspace="lab", label_type="white")
    temp_loader = DataLoader(temp_ds, batch_size=num_samples, shuffle=False)
    
    # 3. Inference
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        images, labels = next(iter(temp_loader))
        images = images.to(device)
        
        # Expect model to return (logits, attention_map)
        _, attn_maps = model(images, return_attention=True)
        
    # 4. Plotting
    # Layout: Pairs of (Image, Attention)
    cols = 8 # 4 pairs per row
    rows = (num_samples * 2 + cols - 1) // cols 
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2.5 * rows))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # Original Image (Convert back from Tensor for display)
        # Note: images are LAB normalized, but for display we want the original BGR patch
        # or we can convert the tensor. Let's use the original BGR for clarity.
        orig_bgr = selected_patches[i].patch_oriented_bgr
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        
        # Attention Map
        attn = attn_maps[i] # Shape [1, H, W] or [H, W]
        if attn.dim() == 3: attn = attn[0]
        
        # Resize attention to match image size (80x80)
        attn_resized = F.interpolate(
            attn.unsqueeze(0).unsqueeze(0), 
            size=(80, 80), 
            mode='bilinear', 
            align_corners=False
        )[0, 0].cpu().numpy()
        
        # Info from dataframe
        row_info = candidates.iloc[i]
        title_color = "red" if row_info.is_error else "black"
        
        # Plot Image
        ax_img = axes[i * 2]
        ax_img.imshow(orig_rgb)
        ax_img.set_title(f"T:{int(row_info.true_white)} P:{row_info.pred_prob:.2f}", color=title_color, fontsize=9)
        ax_img.axis('off')
        
        # Plot Attention
        ax_attn = axes[i * 2 + 1]
        ax_attn.imshow(orig_rgb, alpha=0.4) # Overlay on faint image
        ax_attn.imshow(attn_resized, cmap='jet', alpha=0.6) # Heatmap
        ax_attn.set_title("Attention", fontsize=9)
        ax_attn.axis('off')
        
    # Hide unused
    for j in range(num_samples * 2, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved attention plot to {output_path}")
        plt.close(fig) # Close to avoid double display if saving
    else:
        plt.show()


def get_results_styler(df: pd.DataFrame):
    """
    Return a pandas Styler for the results dataframe.
    
    Highlights:
    - Hides 'file' column (too wide)
    - Gradient for 'pred_prob'
    - Red text for errors
    """
    styler = df.style
    
    if "file" in df.columns:
        styler = styler.hide(subset=["file"], axis="columns")
        
    styler = (styler
        .format({"pred_prob": "{:.4f}", "cell_id": "{:02d}"})
        .background_gradient(subset=["pred_prob"], cmap="RdYlGn", vmin=0, vmax=1)
        # Highlight errors in red (text color)
        .apply(lambda col: ["color: red; font-weight: bold" if v else "" for v in col], subset=["is_error"])
    )
    
    return styler

