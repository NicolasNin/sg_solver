import cv2
import numpy as np

colors_hex={
    "BlueBoard":"#7a7875",
    "Black":"#000000",
    "White":"#ffffff",
    "Blue": "#00325b",  # deep blue
    "Yellow": "#ffc100",  # Yellow
    "Cyan": "#00a0de",  # Cyan
    "Cyan": "#00a0de",  # Cyan
    "Rose": "#e74a61", #ee68a7",  # Rose
    "Red": "#de241b",  # Red
    "Purple": "#7a2d9e",  # Purple
    "Orange": "#ff8717",  # ORANGE
    "Teal": "#006c43",  # teal
    "Lime": "#8bd100",  # lime
    "Brown": "#594247",#8a5e3c",  # BRown
}

def hex_to_lab(hex_color: str):
    """
    hex_color: e.g. "#FF8800" or "FF8800"
    returns: (L, a, b) as OpenCV LAB (uint8)
    """
    h = hex_color.lstrip('#')
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)

    # OpenCV expects BGR, uint8
    bgr = np.uint8([[[b, g, r]]])  # shape (1,1,3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0, :]
    return lab

def extract_patches(img, centers, patch_size):
    """
    img      : HxWxC
    centers  : iterable of (x, y) in image coordinates
    patch_size : int or (w, h)
    returns: list of patches (each h x w x C)
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)  # (w, h)

    patches = []
    for (cx, cy) in centers:
        patch = cv2.getRectSubPix(img, patch_size, (float(cx), float(cy)))
        patches.append(patch)
    return patches