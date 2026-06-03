"""
PHASE 1 - LESSON 4: Resize, Crop, Flip, Rotate
================================================
Learn: resize, ROI cropping, flip, rotate, translate
Dataset: Photos/cat_large.jpg, Photos/cats.jpg
"""

import cv2
import numpy as np

img = cv2.imread("../Photos/cat_large.jpg")
if img is None:
    img = cv2.imread("../Photos/cat.jpg")   # fallback

print(f"Original shape: {img.shape}")

# ─────────────────────────────────────────────
# 1. RESIZE
# ─────────────────────────────────────────────
# Resize to exact pixel size
img_small = cv2.resize(img, (300, 200))           # (width, height) — NOT (H, W)!
img_large = cv2.resize(img, (800, 600))

# Resize by scale factor (preserve aspect ratio)
scale = 0.5
h, w = img.shape[:2]
img_half = cv2.resize(img, (int(w * scale), int(h * scale)))

# Interpolation methods:
# cv2.INTER_LINEAR  — default, good for enlarging
# cv2.INTER_AREA    — best for shrinking (less aliasing)
# cv2.INTER_CUBIC   — slower but sharper for enlarging
img_shrunk = cv2.resize(img, (200, 150), interpolation=cv2.INTER_AREA)

print(f"Resized shape: {img_small.shape}")
print(f"Half-scale shape: {img_half.shape}")

cv2.imshow("Original", img)
cv2.imshow("Small (300x200)", img_small)
cv2.imshow("Half Scale", img_half)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 2. CROP (Region of Interest — ROI)
# NumPy slicing: img[y1:y2, x1:x2]
# ─────────────────────────────────────────────
h, w = img.shape[:2]

# Crop top-left quarter
top_left = img[0:h//2, 0:w//2]

# Crop center
cx, cy = w // 2, h // 2
crop_size = 100
center_crop = img[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]

# Crop bottom-right
bottom_right = img[h//2:, w//2:]

cv2.imshow("Top Left", top_left)
cv2.imshow("Center Crop", center_crop)
cv2.imshow("Bottom Right", bottom_right)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 3. FLIP
# flipCode: 0=vertical, 1=horizontal, -1=both
# ─────────────────────────────────────────────
flip_h = cv2.flip(img, 1)     # mirror left-right
flip_v = cv2.flip(img, 0)     # mirror up-down
flip_both = cv2.flip(img, -1) # rotate 180°

cv2.imshow("Horizontal Flip", flip_h)
cv2.imshow("Vertical Flip", flip_v)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 4. ROTATE
# ─────────────────────────────────────────────
h, w = img.shape[:2]
center = (w // 2, h // 2)

# getRotationMatrix2D(center, angle, scale)
# angle: positive = counter-clockwise
M_90  = cv2.getRotationMatrix2D(center, 90, 1.0)
M_45  = cv2.getRotationMatrix2D(center, 45, 1.0)
M_neg = cv2.getRotationMatrix2D(center, -30, 0.8)  # 30° CW, slightly smaller

rotated_90  = cv2.warpAffine(img, M_90, (w, h))
rotated_45  = cv2.warpAffine(img, M_45, (w, h))
rotated_neg = cv2.warpAffine(img, M_neg, (w, h))

cv2.imshow("90° Rotation", rotated_90)
cv2.imshow("45° Rotation", rotated_45)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 5. TRANSLATE (SHIFT)
# ─────────────────────────────────────────────
tx, ty = 100, 50   # shift right 100px, down 50px
M_translate = np.float32([[1, 0, tx],
                           [0, 1, ty]])
translated = cv2.warpAffine(img, M_translate, (w, h))

cv2.imshow("Translated", translated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 6. PADDING (add border)
# ─────────────────────────────────────────────
padded = cv2.copyMakeBorder(img, 20, 20, 20, 20,
                             cv2.BORDER_CONSTANT,
                             value=(0, 0, 0))  # black border
# Other border types: BORDER_REFLECT, BORDER_REPLICATE

cv2.imshow("Padded", padded)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# EXERCISE
# ─────────────────────────────────────────────
# 1. Load "Photos/cats 2.jpg" and resize to exactly 512x512
# 2. Crop just the face/head area of one cat (estimate coordinates)
# 3. Save both the resized and cropped images
# 4. Rotate the image 180° and flip it horizontally — what does that equal?
