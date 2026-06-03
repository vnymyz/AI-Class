"""
PHASE 2 - LESSON 2: Edge Detection
=====================================
Learn: Sobel, Laplacian, Canny — the 3 main edge detectors
Why: find object boundaries, shape analysis, pre-step for many algorithms
Dataset: Photos/park.jpg, Photos/cat.jpg
"""

import cv2
import numpy as np

img = cv2.imread("../Photos/park.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ALWAYS blur before edge detection to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ─────────────────────────────────────────────
# 1. SOBEL OPERATOR
# Detects edges in X direction (vertical edges)
# or Y direction (horizontal edges)
# ─────────────────────────────────────────────
# Sobel(img, depth, dx, dy, ksize)
# depth=-1 means same depth as source
# dx=1,dy=0 → vertical edges
# dx=0,dy=1 → horizontal edges
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Convert back to uint8 for display
sobel_x_abs = cv2.convertScaleAbs(sobel_x)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# Combine both directions — full edge magnitude
sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

cv2.imshow("Original", img)
cv2.imshow("Sobel X (vertical edges)", sobel_x_abs)
cv2.imshow("Sobel Y (horizontal edges)", sobel_y_abs)
cv2.imshow("Sobel Combined", sobel_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 2. LAPLACIAN
# Detects edges in ALL directions at once
# More sensitive to noise — always blur first
# ─────────────────────────────────────────────
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)

cv2.imshow("Laplacian", laplacian_abs)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 3. CANNY EDGE DETECTOR  ← most popular, best results
# Multi-step: noise reduction → gradient → NMS → hysteresis
# Two thresholds control what counts as an edge
# ─────────────────────────────────────────────
# Canny(img, threshold1, threshold2)
# threshold1: lower — edges below this are rejected
# threshold2: upper — edges above this are kept
# edges between → kept only if connected to strong edge

canny_tight  = cv2.Canny(blurred, 150, 200)  # fewer edges (strict)
canny_mid    = cv2.Canny(blurred, 100, 150)  # balanced
canny_loose  = cv2.Canny(blurred, 50, 100)   # more edges (permissive)

cv2.imshow("Canny (tight: 150-200)", canny_tight)
cv2.imshow("Canny (mid: 100-150)", canny_mid)
cv2.imshow("Canny (loose: 50-100)", canny_loose)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 4. AUTO CANNY (compute thresholds automatically)
# Smart approach: use median pixel value to set thresholds
# ─────────────────────────────────────────────
def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

canny_auto = auto_canny(blurred)
cv2.imshow("Auto Canny", canny_auto)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 5. EDGE DETECTION COMPARISON ON DIFFERENT IMAGES
# ─────────────────────────────────────────────
for filename in ["../Photos/cat.jpg", "../Photos/lady.jpg", "../Photos/group 1.jpg"]:
    test_img = cv2.imread(filename)
    if test_img is None:
        continue
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_blur = cv2.GaussianBlur(test_gray, (5, 5), 0)
    edges = auto_canny(test_blur)

    # Show original + edges side by side
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([test_img, edges_color])
    cv2.imshow(f"Edges: {filename.split('/')[-1]}", combined)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 6. OVERLAY EDGES ON ORIGINAL
# ─────────────────────────────────────────────
edges_final = auto_canny(blurred)
overlay = img.copy()
overlay[edges_final != 0] = [0, 255, 0]   # color edges green

cv2.imshow("Edge Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
# Sobel     → directional edges, good for gradient analysis
# Laplacian → all-direction, simple but noisy
# Canny     → best overall, use this by default
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# EXERCISE
# ─────────────────────────────────────────────
# 1. Load "Photos/cats 2.jpg"
# 2. Compare Canny results with and without Gaussian blur pre-processing
# 3. Try ksize=5 and ksize=7 in Sobel — does larger ksize give thicker edges?
# 4. Build a function: takes image path + threshold → returns edge image + saves it
