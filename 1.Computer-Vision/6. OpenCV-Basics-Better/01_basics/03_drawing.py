"""
PHASE 1 - LESSON 3: Drawing on Images
======================================
Learn: lines, rectangles, circles, polygons, text
Use case: visualizing bounding boxes, annotations, results
Dataset: Photos/park.jpg, Photos/group 1.jpg
"""

import cv2
import numpy as np

img = cv2.imread("../Photos/park.jpg")

# ─────────────────────────────────────────────
# IMPORTANT: drawing modifies image IN-PLACE
# Always work on a copy if you need the original
# ─────────────────────────────────────────────
canvas = img.copy()

# ─────────────────────────────────────────────
# 1. LINE
# cv2.line(img, start_pt, end_pt, color, thickness)
# color = (B, G, R) — note BGR order!
# ─────────────────────────────────────────────
cv2.line(canvas, (0, 0), (400, 400), (0, 255, 0), 2)        # green diagonal
cv2.line(canvas, (0, 100), (canvas.shape[1], 100), (255, 0, 0), 1)  # blue horizontal

# ─────────────────────────────────────────────
# 2. RECTANGLE
# cv2.rectangle(img, top_left, bottom_right, color, thickness)
# thickness=-1 fills the rectangle
# ─────────────────────────────────────────────
cv2.rectangle(canvas, (50, 50), (200, 150), (0, 0, 255), 3)       # red outline
cv2.rectangle(canvas, (220, 50), (350, 150), (0, 165, 255), -1)    # filled orange

# ─────────────────────────────────────────────
# 3. CIRCLE
# cv2.circle(img, center, radius, color, thickness)
# ─────────────────────────────────────────────
cv2.circle(canvas, (100, 250), 50, (255, 255, 0), 2)   # cyan outline
cv2.circle(canvas, (250, 250), 30, (255, 0, 255), -1)  # filled magenta

# ─────────────────────────────────────────────
# 4. ELLIPSE
# cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)
# ─────────────────────────────────────────────
cv2.ellipse(canvas, (400, 200), (80, 40), 45, 0, 360, (0, 255, 255), 2)

# ─────────────────────────────────────────────
# 5. POLYGON
# Must use np.array with int32 dtype
# ─────────────────────────────────────────────
pts = np.array([[100, 300], [200, 280], [300, 320], [250, 400], [150, 390]], dtype=np.int32)
pts = pts.reshape((-1, 1, 2))  # required shape for polylines

cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 128), thickness=2)
# isClosed=True connects last point back to first

# ─────────────────────────────────────────────
# 6. TEXT
# cv2.putText(img, text, origin, font, scale, color, thickness, lineType)
# origin = bottom-left corner of text
# ─────────────────────────────────────────────
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(canvas, "OpenCV Drawing", (10, 30), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(canvas, "Hello World", (10, 450), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

# cv2.LINE_AA = antialiased (smoother edges)

cv2.imshow("Drawing Demo", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 7. BLANK CANVAS
# ─────────────────────────────────────────────
# Create black image: zeros
black = np.zeros((400, 600, 3), dtype=np.uint8)

# Create white image: 255
white = np.ones((400, 600, 3), dtype=np.uint8) * 255

# Draw on blank canvas — useful for creating visualizations from scratch
cv2.rectangle(black, (50, 50), (550, 350), (0, 200, 255), 5)
cv2.putText(black, "Custom Canvas", (150, 220), font, 1.5, (0, 200, 255), 2)

cv2.imshow("Black Canvas", black)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 8. PRACTICAL: DRAW BOUNDING BOXES ON FACES
# ─────────────────────────────────────────────
# Load group photo and simulate annotation
group = cv2.imread("../Photos/group 1.jpg")
if group is not None:
    # These are fake boxes — in real use, detector gives you these coords
    fake_boxes = [(50, 30, 150, 130), (200, 40, 300, 140), (380, 20, 480, 120)]
    for (x1, y1, x2, y2) in fake_boxes:
        cv2.rectangle(group, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(group, "Face", (x1, y1 - 5), font, 0.5, (0, 255, 0), 1)

    cv2.imshow("Annotated Group", group)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# EXERCISE
# ─────────────────────────────────────────────
# 1. Create 600x600 black canvas
# 2. Draw a traffic light: 3 circles (red, yellow, green) stacked vertically
# 3. Add label "STOP", "SLOW", "GO" next to each circle
# 4. Save as "traffic_light.jpg"
