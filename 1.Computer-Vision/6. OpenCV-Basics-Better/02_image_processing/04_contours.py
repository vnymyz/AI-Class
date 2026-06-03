"""
PHASE 2 - LESSON 4: Contours
==============================
Learn: find contours, draw them, measure area/perimeter, bounding boxes
Why: shape detection, object counting, region analysis
Dataset: Photos/cat.jpg, Photos/group 1.jpg
"""

import cv2
import numpy as np

img = cv2.imread("../Photos/cat.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ─────────────────────────────────────────────
# 1. FIND CONTOURS
# Contours are curves joining continuous points of same intensity
# MUST use binary image as input (threshold or Canny edges)
# ─────────────────────────────────────────────
# Using Canny edges
edges = cv2.Canny(blurred, 50, 150)

# findContours(image, mode, method)
# mode:
#   RETR_EXTERNAL → only outermost contours
#   RETR_LIST     → all contours, no hierarchy
#   RETR_TREE     → all contours + full hierarchy
# method:
#   CHAIN_APPROX_NONE    → every contour point stored
#   CHAIN_APPROX_SIMPLE  → compress — only endpoints stored (smaller)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(contours)} contours")

# ─────────────────────────────────────────────
# 2. DRAW CONTOURS
# drawContours(img, contours, contourIdx, color, thickness)
# contourIdx=-1 draws ALL contours
# ─────────────────────────────────────────────
canvas = img.copy()
cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

cv2.imshow("All Contours", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 3. FILTER BY AREA
# Ignore tiny contours (noise), keep significant ones
# ─────────────────────────────────────────────
canvas2 = img.copy()
min_area = 500

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > min_area:
        cv2.drawContours(canvas2, [cnt], -1, (0, 255, 0), 2)

cv2.imshow(f"Contours with area > {min_area}", canvas2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 4. CONTOUR PROPERTIES
# ─────────────────────────────────────────────
for i, cnt in enumerate(contours[:5]):  # look at first 5
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, closed=True)

    # Bounding rectangle (axis-aligned)
    x, y, w, h = cv2.boundingRect(cnt)

    # Minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)

    # Aspect ratio
    aspect = float(w) / h if h > 0 else 0

    print(f"Contour {i}: area={area:.0f}, perimeter={perimeter:.0f}, "
          f"bbox=({x},{y},{w},{h}), aspect={aspect:.2f}")

# ─────────────────────────────────────────────
# 5. BOUNDING BOXES AROUND CONTOURS
# ─────────────────────────────────────────────
canvas3 = img.copy()

for cnt in contours:
    if cv2.contourArea(cnt) < min_area:
        continue

    # Straight bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(canvas3, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Rotated bounding box (tighter fit)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(canvas3, [box], 0, (255, 0, 0), 2)

cv2.imshow("Bounding Boxes (red=straight, blue=rotated)", canvas3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 6. CONVEX HULL
# Smallest convex shape enclosing the contour
# ─────────────────────────────────────────────
canvas4 = img.copy()

for cnt in contours:
    if cv2.contourArea(cnt) < min_area:
        continue

    hull = cv2.convexHull(cnt)
    cv2.drawContours(canvas4, [hull], -1, (255, 255, 0), 2)

cv2.imshow("Convex Hulls", canvas4)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 7. SHAPE APPROXIMATION
# Reduce contour points while keeping shape
# ─────────────────────────────────────────────
canvas5 = img.copy()

for cnt in contours:
    if cv2.contourArea(cnt) < min_area:
        continue

    # epsilon = accuracy — fraction of perimeter
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    cv2.drawContours(canvas5, [approx], -1, (0, 128, 255), 2)
    print(f"Original points: {len(cnt)}, Approximated: {len(approx)}")

cv2.imshow("Approximated Contours", canvas5)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 8. PRACTICAL: COUNT OBJECTS
# ─────────────────────────────────────────────
# Threshold-based approach (cleaner than Canny for counting)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Count only meaningful contours
significant = [c for c in cnts if cv2.contourArea(c) > 1000]
print(f"\nSignificant objects detected: {len(significant)}")

count_canvas = img.copy()
for i, cnt in enumerate(significant):
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(count_canvas, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(count_canvas, str(i+1), (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Object Count", count_canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# EXERCISE
# ─────────────────────────────────────────────
# 1. Load "Photos/cats 2.jpg" — how many cats can contours detect?
# 2. Filter contours by aspect ratio (w/h > 1.5) to find wide objects
# 3. Draw convex hull + bounding box on same image, compare shapes
# 4. Try different min_area values — what's the "right" cutoff?
