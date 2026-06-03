"""
PHASE 3 - LESSON 2: Background Subtraction & Motion Detection
==============================================================
Learn: MOG2, KNN background subtractors, motion detection
Why: detect moving objects in video (people, cars, animals)
Dataset: Videos/dog.mp4, Videos/kitten.mp4
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────
# CONCEPT: Background Subtraction
# Algorithm learns the "background" over time
# Any pixel that differs from background = foreground (moving object)
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 1. MOG2 (Mixture of Gaussians v2)  ← most popular
# Models each pixel as mix of Gaussian distributions
# Adapts to gradual lighting changes
# ─────────────────────────────────────────────
cap = cv2.VideoCapture("../Videos/dog.mp4")

# Create background subtractor
# history: number of frames used to build background model
# varThreshold: sensitivity (lower = more sensitive to change)
# detectShadows: mark shadows gray instead of white (slower)
bg_mog2 = cv2.createBackgroundSubtractorMOG2(history=500,
                                               varThreshold=50,
                                               detectShadows=True)

# Morphological kernels for cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_mog2.apply(frame)
    # fg_mask: 0=background, 128=shadow, 255=foreground

    # Remove shadows (set 128 → 0)
    _, fg_clean = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Morphological cleanup: remove noise
    fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_OPEN, kernel)   # remove small dots
    fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_CLOSE, kernel)  # fill holes

    # Find contours in foreground mask
    contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # ignore tiny noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output, "Moving", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display
    cv2.imshow("Original", frame)
    cv2.imshow("Foreground Mask", fg_clean)
    cv2.imshow("Detected Motion", output)

    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 2. KNN (K-Nearest Neighbors) Background Subtractor
# More robust to sudden lighting changes
# Better for complex backgrounds
# ─────────────────────────────────────────────
cap = cv2.VideoCapture("../Videos/kitten.mp4")

bg_knn = cv2.createBackgroundSubtractorKNN(history=500,
                                            dist2Threshold=400,
                                            detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = bg_knn.apply(frame)
    _, fg_clean = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("KNN Detection", output)
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 3. SIMPLE FRAME DIFFERENCE (no ML, pure math)
# Compare current frame to previous frame
# ─────────────────────────────────────────────
cap = cv2.VideoCapture("../Videos/dog.mp4")

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Absolute difference between current and previous frame
    diff = cv2.absdiff(prev_gray, gray)

    # Threshold the difference
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Dilate to fill gaps
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)

    # Count motion pixels
    motion_pixels = cv2.countNonZero(motion_mask)

    output = frame.copy()
    if motion_pixels > 1000:
        cv2.putText(output, "MOTION DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Frame Diff Motion", output)
    cv2.imshow("Motion Mask", motion_mask)

    prev_gray = gray.copy()

    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
# Frame Difference  → simplest, works when camera is static
# MOG2              → best general purpose background subtraction
# KNN               → better with complex/changing backgrounds
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# EXERCISE
# ─────────────────────────────────────────────
# 1. Run MOG2 on kitten.mp4 — can it detect the kitten's movement?
# 2. Compare detection quality: does MOG2 or KNN work better on dog.mp4?
# 3. Add a frame counter: count how many frames have motion detected
# 4. Save frames where motion_pixels > 5000 as JPEG files (motion snapshots)
