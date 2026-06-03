"""
OpenCV Learning Path — Quick Reference
========================================
Run files in order. Each builds on the previous.

PHASE 1 — BASICS (start here)
    01_basics/01_read_display.py    ← imread, imshow, imwrite
    01_basics/02_color_spaces.py    ← BGR, RGB, Gray, HSV
    01_basics/03_drawing.py         ← lines, rectangles, text
    01_basics/04_resize_crop.py     ← resize, crop, flip, rotate

PHASE 2 — IMAGE PROCESSING
    02_image_processing/01_blur_sharpen.py  ← Gaussian, median, bilateral
    02_image_processing/02_edges.py         ← Sobel, Canny edge detection
    02_image_processing/03_thresholding.py  ← binary, Otsu, adaptive
    02_image_processing/04_contours.py      ← find shapes, bounding boxes

PHASE 3 — VIDEO
    03_video/01_read_video.py              ← VideoCapture, frame loop, save
    03_video/02_background_subtraction.py  ← MOG2, KNN, motion detection

PHASE 4 — FACES
    04_faces/01_face_detection.py    ← Haar Cascade, find faces
    04_faces/02_face_recognition.py  ← LBPH, train + predict who it is

DATASET
    Photos/          → 8 images for phases 1-2
    Videos/          → dog.mp4, kitten.mp4 for phase 3
    Faces/train/     → 4 celebrities for training
    Faces/val/       → same 4 people for validation

TIPS
    - Press Q in any imshow window to quit
    - Press P to pause video playback
    - Each file has EXERCISE section at bottom — try them!
    - All files run from the OpenCV-Basics/ root directory
"""

print("OpenCV Learning Path loaded.")
print("Start with: 01_basics/01_read_display.py")
print()
print("Install dependencies:")
print("  pip install opencv-python opencv-contrib-python numpy")
