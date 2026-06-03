"""
PHASE 1 - LESSON 2: Color Spaces
=================================
Learn: BGR, RGB, Grayscale, HSV, LAB
Why it matters: different tasks need different color spaces
  - Face detection: grayscale
  - Color filtering (isolate red/green/blue): HSV
  - Perceptual similarity: LAB
Dataset: Photos/cat.jpg
"""

import cv2
import numpy as np

img = cv2.imread("../Photos/cat.jpg")

# ─────────────────────────────────────────────
# KEY FACT: OpenCV reads images as BGR, not RGB!
# Most other libraries (matplotlib, PIL) use RGB.
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 1. BGR → RGB
# ─────────────────────────────────────────────
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Visually same as original in imshow (imshow expects BGR)
# But needed when passing to matplotlib or ML models

# ─────────────────────────────────────────────
# 2. BGR → GRAYSCALE
# ─────────────────────────────────────────────
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Shape becomes (H, W) instead of (H, W, 3)
print(f"Gray shape: {img_gray.shape}")

# ─────────────────────────────────────────────
# 3. BGR → HSV  (Hue, Saturation, Value)
# ─────────────────────────────────────────────
# Best for color-based filtering and detection
# H: 0–179 (red=0/179, green=60, blue=120)
# S: 0–255 (0=gray, 255=pure color)
# V: 0–255 (0=black, 255=bright)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ─────────────────────────────────────────────
# 4. BGR → LAB  (Lightness, A, B)
# ─────────────────────────────────────────────
# Perceptually uniform — good for color similarity
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# ─────────────────────────────────────────────
# 5. DISPLAY ALL COLOR SPACES
# ─────────────────────────────────────────────
cv2.imshow("Original (BGR)", img)
cv2.imshow("Grayscale", img_gray)
cv2.imshow("HSV", img_hsv)
cv2.imshow("LAB", img_lab)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 6. SPLIT & MERGE CHANNELS
# ─────────────────────────────────────────────
b, g, r = cv2.split(img)   # each is a 2D grayscale array

print(f"Blue channel shape: {b.shape}")
print(f"Blue avg pixel: {b.mean():.1f}")
print(f"Green avg pixel: {g.mean():.1f}")
print(f"Red avg pixel: {r.mean():.1f}")

cv2.imshow("Blue Channel", b)
cv2.imshow("Green Channel", g)
cv2.imshow("Red Channel", r)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Merge back
img_merged = cv2.merge([b, g, r])

# ─────────────────────────────────────────────
# 7. COLOR FILTERING WITH HSV (Practical Use)
# ─────────────────────────────────────────────
# Example: isolate yellow tones in the image
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_yellow = np.array([20, 100, 100])   # lower HSV bound
upper_yellow = np.array([30, 255, 255])   # upper HSV bound

mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Yellow Mask", mask)
cv2.imshow("Yellow Filtered", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# EXERCISE
# ─────────────────────────────────────────────
# 1. Load "Photos/park.jpg" and split its HSV channels
#    Print min/max of H channel to understand hue range
# 2. Create a mask that isolates green colors in the park image
#    Hint: green HSV range is roughly [35, 50, 50] to [85, 255, 255]
# 3. Apply that mask and display result
