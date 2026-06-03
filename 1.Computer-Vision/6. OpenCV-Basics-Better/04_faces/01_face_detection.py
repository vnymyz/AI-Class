"""
PHASE 4 - LESSON 1: Face Detection
=====================================
Learn: Haar Cascade detector, DNN-based detector
Why: detect WHERE faces are in an image
Dataset: Photos/group 1.jpg, Photos/lady.jpg, Faces/val/
"""

import cv2
import numpy as np
import os

# ─────────────────────────────────────────────
# METHOD 1: HAAR CASCADE (classic, fast, less accurate)
# Pre-trained XML files bundled with OpenCV
# ─────────────────────────────────────────────

# Find OpenCV's bundled Haar Cascade files
cv2_data = cv2.data.haarcascades
print(f"Haar Cascades location: {cv2_data}")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2_data + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2_data + "haarcascade_eye.xml")

def detect_faces_haar(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot load: {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # improve contrast — helps detection

    # detectMultiScale(image, scaleFactor, minNeighbors, minSize)
    # scaleFactor: how much to reduce image size each pass (1.1 = 10%)
    # minNeighbors: how many neighbors each candidate needs (higher = stricter)
    # minSize: minimum face size in pixels
    faces = face_cascade.detectMultiScale(gray,
                                           scaleFactor=1.1,
                                           minNeighbors=5,
                                           minSize=(30, 30))

    print(f"{img_path}: {len(faces)} face(s) detected")

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Detect eyes inside each face ROI
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(face_roi_color, (ex + ew//2, ey + eh//2), ew//2, (255, 0, 0), 2)

        cv2.putText(img, f"Face", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Haar Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

# Test on photos
detect_faces_haar("../Photos/group 1.jpg")
detect_faces_haar("../Photos/group 2.jpg")
detect_faces_haar("../Photos/lady.jpg")

# Test on validation faces
val_dir = "../Faces/val"
for person in os.listdir(val_dir):
    person_dir = os.path.join(val_dir, person)
    if os.path.isdir(person_dir):
        imgs = [f for f in os.listdir(person_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            detect_faces_haar(os.path.join(person_dir, imgs[0]))

# ─────────────────────────────────────────────
# METHOD 2: DNN-based detector (more accurate, modern)
# Uses deep learning — handles tilted/partial faces better
# Need to download model files (see below)
# ─────────────────────────────────────────────

# Download these two files from:
# https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
# - deploy.prototxt
# - res10_300x300_ssd_iter_140000.caffemodel

PROTOTXT = "deploy.prototxt"
MODEL    = "res10_300x300_ssd_iter_140000.caffemodel"

if os.path.exists(PROTOTXT) and os.path.exists(MODEL):
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

    def detect_faces_dnn(img_path, confidence_threshold=0.5):
        img = cv2.imread(img_path)
        if img is None:
            return

        h, w = img.shape[:2]

        # DNN requires image in blob format: 300x300, mean subtracted
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                      scalefactor=1.0,
                                      size=(300, 300),
                                      mean=(104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                # Scale bounding box back to original image size
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{confidence:.2f}",
                            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("DNN Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    detect_faces_dnn("../Photos/group 1.jpg")
else:
    print("\nDNN model files not found.")
    print("Download from OpenCV samples to use DNN detector.")
    print("For now, Haar Cascade works fine for learning!")

# ─────────────────────────────────────────────
# HAAR vs DNN COMPARISON
# ─────────────────────────────────────────────
# Haar Cascade:
#   + Fast, no extra files needed, built into OpenCV
#   - Misses tilted/side faces, more false positives
#   Use when: real-time, frontal faces only
#
# DNN:
#   + Much more accurate, handles angles and occlusion
#   - Slightly slower, needs model files
#   Use when: accuracy matters, faces at various angles
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# EXERCISE
# ─────────────────────────────────────────────
# 1. Try different minNeighbors values (3, 5, 7, 10) on group photo
#    — how does false positive rate change?
# 2. Try different scaleFactor values (1.05, 1.1, 1.3)
#    — what's the tradeoff between speed and accuracy?
# 3. Run detection on ALL val/ images for each person
#    Count: how many does Haar successfully detect?
# 4. Crop each detected face and save as face_1.jpg, face_2.jpg etc.
