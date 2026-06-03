"""
PHASE 3 - LESSON 1: Video Reading and Writing
===============================================
Learn: VideoCapture, frame-by-frame processing, VideoWriter
Dataset: Videos/dog.mp4, Videos/kitten.mp4
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────
# 1. READ VIDEO FILE
# ─────────────────────────────────────────────
cap = cv2.VideoCapture("../Videos/dog.mp4")

# Always check if opened successfully
if not cap.isOpened():
    print("ERROR: Cannot open video file")
    exit()

# ─────────────────────────────────────────────
# 2. VIDEO PROPERTIES
# ─────────────────────────────────────────────
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS    : {fps}")
print(f"Size   : {width}x{height}")
print(f"Frames : {total}")
print(f"Duration: {total/fps:.1f} seconds")

# ─────────────────────────────────────────────
# 3. READ FRAMES IN A LOOP
# ─────────────────────────────────────────────
# cap.read() returns (success, frame)
# When video ends: success=False, frame=None

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print(f"Video ended at frame {frame_count}")
        break

    frame_count += 1

    # Display current frame number on video
    cv2.putText(frame, f"Frame: {frame_count}/{total}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Dog Video", frame)

    # waitKey(delay_ms) — controls playback speed
    # int(1000/fps) = real-time speed
    # 1 = as fast as possible
    # 0 = pause until key press
    key = cv2.waitKey(int(1000 / fps))
    if key == ord('q'):      # press Q to quit
        break
    elif key == ord('p'):    # press P to pause
        cv2.waitKey(0)       # wait until any key pressed

cap.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 4. PROCESS EACH FRAME (apply image processing)
# ─────────────────────────────────────────────
cap = cv2.VideoCapture("../Videos/kitten.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply grayscale to each frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edges
    edges = cv2.Canny(gray, 50, 150)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Stack original + processed side by side
    display = np.hstack([frame, edges_color])

    # Resize for display if too wide
    dh, dw = display.shape[:2]
    if dw > 1200:
        scale = 1200 / dw
        display = cv2.resize(display, (1200, int(dh * scale)))

    cv2.imshow("Original | Edges", display)
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 5. SAVE VIDEO
# ─────────────────────────────────────────────
cap = cv2.VideoCapture("../Videos/dog.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# FourCC = 4-character codec code
# mp4v → .mp4
# XVID → .avi
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_processed.mp4", fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame — apply blur
    processed = cv2.GaussianBlur(frame, (15, 15), 0)

    out.write(processed)   # write processed frame to output

cap.release()
out.release()
print("Saved: output_processed.mp4")

# ─────────────────────────────────────────────
# 6. JUMP TO SPECIFIC FRAME
# ─────────────────────────────────────────────
cap = cv2.VideoCapture("../Videos/dog.mp4")

target_frame = 30  # jump to frame 30
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

ret, frame = cap.read()
if ret:
    cv2.imshow(f"Frame {target_frame}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cap.release()

# ─────────────────────────────────────────────
# 7. WEBCAM CAPTURE (bonus)
# ─────────────────────────────────────────────
# To use webcam instead of file:
# cap = cv2.VideoCapture(0)   # 0 = default webcam
# cap = cv2.VideoCapture(1)   # external webcam

# Uncomment to try:
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret: break
#     cv2.imshow("Webcam", frame)
#     if cv2.waitKey(1) == ord('q'): break
# cap.release()
# cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# EXERCISE
# ─────────────────────────────────────────────
# 1. Load "Videos/kitten.mp4" and save every 5th frame as a JPEG
#    Name them: frame_000.jpg, frame_005.jpg, frame_010.jpg...
# 2. Create a grayscale version of dog.mp4 and save it
# 3. Extract the first 5 seconds of kitten.mp4 and save as new file
