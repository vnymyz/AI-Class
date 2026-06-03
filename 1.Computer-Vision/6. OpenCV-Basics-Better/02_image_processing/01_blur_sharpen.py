"""
PHASE 2 - LESSON 1: Blurring and Sharpening
=============================================
Learn: averaging, Gaussian, median, bilateral blur + sharpening
Why: noise removal, pre-processing before edge/face detection
Dataset: Photos/cats.jpg
"""

import cv2
import numpy as np

img = cv2.imread("../Photos/cats.jpg")

# ─────────────────────────────────────────────
# 1. AVERAGE BLUR
# Simple mean of all pixels in kernel
# Fast but blurs edges too
# ─────────────────────────────────────────────
blur_avg = cv2.blur(img, (5, 5))       # kernel size (width, height), must be odd
blur_avg_large = cv2.blur(img, (15, 15))

cv2.imshow("Original", img)
cv2.imshow("Average Blur 5x5", blur_avg)
cv2.imshow("Average Blur 15x15", blur_avg_large)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 2. GAUSSIAN BLUR  ← most commonly used
# Weighted average — pixels closer to center get more weight
# Better than average: less edge destruction
# ─────────────────────────────────────────────
# GaussianBlur(img, ksize, sigmaX)
# sigmaX=0 lets OpenCV calculate from ksize
blur_gauss_3 = cv2.GaussianBlur(img, (3, 3), 0)
blur_gauss_7 = cv2.GaussianBlur(img, (7, 7), 0)
blur_gauss_21 = cv2.GaussianBlur(img, (21, 21), 0)

cv2.imshow("Gaussian 3x3", blur_gauss_3)
cv2.imshow("Gaussian 7x7", blur_gauss_7)
cv2.imshow("Gaussian 21x21", blur_gauss_21)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 3. MEDIAN BLUR
# Replaces pixel with MEDIAN of surrounding pixels
# Best for: removing salt-and-pepper noise
# Preserves edges better than Gaussian
# ─────────────────────────────────────────────

# First: add artificial noise to see the difference
noisy = img.copy()
noise_mask = np.random.randint(0, 2, (img.shape[0], img.shape[1]), dtype=np.uint8)
noisy[noise_mask == 0] = 0    # black dots
noisy[noise_mask == 1] = 255  # white dots (50% chance each)

# Actually create proper salt-pepper noise
noisy2 = img.copy()
salt = np.random.random(img.shape[:2]) < 0.02   # 2% salt
pepper = np.random.random(img.shape[:2]) < 0.02  # 2% pepper
noisy2[salt] = 255
noisy2[pepper] = 0

blur_median = cv2.medianBlur(noisy2, 5)   # ksize must be odd integer

cv2.imshow("Noisy", noisy2)
cv2.imshow("After Median Blur", blur_median)
cv2.imshow("After Gaussian Blur (compare)", cv2.GaussianBlur(noisy2, (5, 5), 0))
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 4. BILATERAL FILTER  ← best quality, slowest
# Blurs smooth areas but PRESERVES EDGES
# Great for: face smoothing, photo enhancement
# ─────────────────────────────────────────────
# bilateralFilter(img, d, sigmaColor, sigmaSpace)
# d: diameter of pixel neighborhood
# sigmaColor: how much color difference is allowed
# sigmaSpace: spatial distance allowed
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
bilateral_strong = cv2.bilateralFilter(img, 15, 100, 100)

cv2.imshow("Original", img)
cv2.imshow("Bilateral Filter", bilateral)
cv2.imshow("Bilateral Strong", bilateral_strong)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 5. SHARPENING (using kernel)
# Sharpening = Original + (Original - Blurred) → enhance edges
# ─────────────────────────────────────────────

# Method 1: Manual kernel
sharpen_kernel = np.array([[ 0, -1,  0],
                            [-1,  5, -1],
                            [ 0, -1,  0]])
sharp = cv2.filter2D(img, -1, sharpen_kernel)

# Method 2: Unsharp masking
blurred = cv2.GaussianBlur(img, (5, 5), 0)
unsharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

cv2.imshow("Original", img)
cv2.imshow("Sharpened (kernel)", sharp)
cv2.imshow("Unsharp Mask", unsharp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# SUMMARY: When to use which blur
# ─────────────────────────────────────────────
# Average Blur    → fast, rough noise removal
# Gaussian Blur   → general pre-processing (before Canny, face detect)
# Median Blur     → salt-and-pepper noise
# Bilateral       → keep edges, smooth face/skin
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# EXERCISE
# ─────────────────────────────────────────────
# 1. Load "Photos/lady.jpg"
# 2. Apply bilateral filter — does it smooth skin while keeping hair detail?
# 3. Compare Gaussian vs Bilateral on face area (crop face first)
# 4. Add salt-pepper noise manually, then compare median vs Gaussian removal
