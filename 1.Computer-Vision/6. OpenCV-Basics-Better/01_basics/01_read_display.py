"""
PHASE 1 - LESSON 1: Read, Display, Write Images
================================================
Learn: imread, imshow, imwrite, image properties
Dataset: Photos/cat.jpg
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────
# 1. READ AN IMAGE
# ─────────────────────────────────────────────
# cv2.imread returns a NumPy array (BGR format, not RGB!)
# Flags: cv2.IMREAD_COLOR (default), cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED

img = cv2.imread("../Photos/cat.jpg")

# Always check if image loaded successfully
if img is None:
    print("ERROR: Image not found. Check the path.")
    exit()

# ─────────────────────────────────────────────
# 2. IMAGE PROPERTIES
# ─────────────────────────────────────────────
print(f"Shape  : {img.shape}")       # (height, width, channels)
print(f"Size   : {img.size}")        # total pixels * channels
print(f"Dtype  : {img.dtype}")       # uint8 (values 0–255)

height, width, channels = img.shape
print(f"Height : {height}px")
print(f"Width  : {width}px")
print(f"Channels: {channels} (B, G, R)")

# ─────────────────────────────────────────────
# 3. DISPLAY IMAGE
# ─────────────────────────────────────────────
# imshow(window_name, image)
cv2.imshow("Original Cat", img)

# waitKey(0) = wait forever until any key pressed
# waitKey(2000) = wait 2 seconds then continue
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 4. READ AS GRAYSCALE
# ─────────────────────────────────────────────
img_gray = cv2.imread("../Photos/cat.jpg", cv2.IMREAD_GRAYSCALE)
print(f"\nGrayscale shape: {img_gray.shape}")  # (H, W) — no channel dim

cv2.imshow("Grayscale Cat", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 5. SAVE / WRITE IMAGE
# ─────────────────────────────────────────────
# imwrite(filename, image) → returns True if success
success = cv2.imwrite("output_gray_cat.jpg", img_gray)
print(f"\nSaved grayscale: {success}")

# ─────────────────────────────────────────────
# 6. LOAD MULTIPLE IMAGES
# ─────────────────────────────────────────────
images = {
    "Cat":   cv2.imread("../Photos/cat.jpg"),
    "Group": cv2.imread("../Photos/group 1.jpg"),
    "Park":  cv2.imread("../Photos/park.jpg"),
    "Lady":  cv2.imread("../Photos/lady.jpg"),
}

for name, image in images.items():
    if image is not None:
        cv2.imshow(name, image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# EXERCISE
# ─────────────────────────────────────────────
# 1. Load "Photos/dog.jpg" (if exists) and print its shape
# 2. Load "Photos/park.jpg" as grayscale and save it as "park_gray.jpg"
# 3. Display both the color and grayscale versions side by side using imshow
