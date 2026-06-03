"""
PHASE 2 - LESSON 3: Thresholding
==================================
Learn: simple, adaptive, Otsu thresholding
Why: separate foreground from background (binary segmentation)
Dataset: Photos/cat.jpg, Photos/group 1.jpg
"""

import cv2
import numpy as np

img = cv2.imread("../Photos/cat.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ─────────────────────────────────────────────
# 1. SIMPLE THRESHOLD
# If pixel > thresh → set to maxval, else → 0
# threshold(src, thresh, maxval, type)
# ─────────────────────────────────────────────
# Types:
# THRESH_BINARY     : pixel > T → maxval, else 0
# THRESH_BINARY_INV : pixel > T → 0, else maxval  (inverted)
# THRESH_TRUNC      : pixel > T → T, else unchanged
# THRESH_TOZERO     : pixel > T → unchanged, else 0
# THRESH_TOZERO_INV : pixel > T → 0, else unchanged

ret, thresh_binary     = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh_binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh_trunc      = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh_tozero     = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)

print(f"Threshold used: {ret}")  # returns the threshold value

cv2.imshow("Grayscale", gray)
cv2.imshow("Binary (>127=white)", thresh_binary)
cv2.imshow("Binary Inv (>127=black)", thresh_binary_inv)
cv2.imshow("Trunc", thresh_trunc)
cv2.imshow("To Zero", thresh_tozero)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# PROBLEM with simple threshold:
# One global threshold for entire image
# Fails when lighting is uneven
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 2. ADAPTIVE THRESHOLD
# Calculates threshold for small regions separately
# Works much better on images with varied lighting
# ─────────────────────────────────────────────
# adaptiveThreshold(src, maxval, adaptiveMethod, thresholdType, blockSize, C)
# adaptiveMethod:
#   ADAPTIVE_THRESH_MEAN_C    → threshold = mean of neighborhood - C
#   ADAPTIVE_THRESH_GAUSSIAN_C → threshold = gaussian-weighted mean - C
# blockSize: size of neighborhood (odd, >1)
# C: constant subtracted from mean

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

adaptive_mean = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY,
                                       blockSize=11, C=2)

adaptive_gauss = cv2.adaptiveThreshold(blurred, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        blockSize=11, C=2)

cv2.imshow("Simple Threshold", thresh_binary)
cv2.imshow("Adaptive Mean", adaptive_mean)
cv2.imshow("Adaptive Gaussian", adaptive_gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 3. OTSU'S BINARIZATION  ← automatic threshold selection
# Finds optimal threshold by minimizing intra-class variance
# Best for bimodal histograms (clear foreground/background)
# ─────────────────────────────────────────────
# Pass 0 as threshold value + THRESH_OTSU flag
ret_otsu, thresh_otsu = cv2.threshold(blurred, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu's optimal threshold: {ret_otsu}")

cv2.imshow("Otsu Threshold", thresh_otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 4. VISUALIZE HISTOGRAM (understand pixel distribution)
# ─────────────────────────────────────────────
# Histogram shows pixel intensity distribution
# Bimodal (2 peaks) = good candidate for Otsu
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Plot histogram as bar chart image
hist_img = np.zeros((200, 256, 3), dtype=np.uint8)
cv2.normalize(hist, hist, 0, 200, cv2.NORM_MINMAX)
for x, val in enumerate(hist):
    cv2.line(hist_img, (x, 200), (x, 200 - int(val[0])), (0, 255, 0), 1)

cv2.imshow("Pixel Histogram", hist_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 5. PRACTICAL: BACKGROUND REMOVAL (rough)
# ─────────────────────────────────────────────
group = cv2.imread("../Photos/group 1.jpg")
if group is not None:
    g_gray = cv2.cvtColor(group, cv2.COLOR_BGR2GRAY)
    g_blur = cv2.GaussianBlur(g_gray, (5, 5), 0)
    _, g_thresh = cv2.threshold(g_blur, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use threshold as mask
    result = cv2.bitwise_and(group, group, mask=g_thresh)

    cv2.imshow("Original Group", group)
    cv2.imshow("Thresholded", g_thresh)
    cv2.imshow("Masked Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
# Simple Threshold  → use when lighting is uniform
# Adaptive          → use when lighting varies across image
# Otsu              → use when you don't know the threshold value
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# EXERCISE
# ─────────────────────────────────────────────
# 1. Load "Photos/park.jpg"
# 2. Apply all 3 threshold methods and display side by side
# 3. What threshold value does Otsu choose? Is it a good split?
# 4. Try different blockSize values (7, 15, 21) in adaptive threshold
#    — how does it affect the result?
