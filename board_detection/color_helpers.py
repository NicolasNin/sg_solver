import cv2
import numpy as np
def rgb_to_chromaticity(rgb):
    """
    rgb: array of shape (..., 3), channels in any consistent scale
    returns: array of shape (..., 3) with (r, g, b_chroma)
    """
    rgb = rgb.astype(np.float32)
    S = rgb.sum(axis=-1, keepdims=True)  # (..., 1)
    eps = 1e-8
    S_safe = np.where(S == 0, eps, S)

    chroma = rgb / S_safe  # (..., 3), r+g+b ~= 1
    return chroma

def normalize_L_mean(img_BGR, board_mask, target_L_mean=100):
    lab_img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
    lab = lab_img.copy()
    L = lab[:, :, 0].astype(np.float32)

    # current mean L over the board
    board_pixels = L[board_mask > 0]
    cur_mean = float(board_pixels.mean())

    # additive offset to bring cur_mean → target_L_mean
    delta = target_L_mean - cur_mean
    L_new = L + delta
    L_new = np.clip(L_new, 0, 255).astype(np.uint8)

    lab[:, :, 0] = L_new
    img_BGR_new = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img_BGR_new, cur_mean

