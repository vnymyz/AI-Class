"""
PHASE 4 - LESSON 2: Face Recognition
========================================
Learn: LBPH face recognizer — train on known faces, predict unknown
Why: identify WHO a face belongs to (not just WHERE)
Dataset: Faces/train/ (4 people), Faces/val/ (same 4 people)
"""

import cv2
import numpy as np
import os

# ─────────────────────────────────────────────
# CONCEPT: Face Recognition Pipeline
# 1. Detect face in image (find WHERE the face is)
# 2. Crop + normalize the face region
# 3. Train recognizer on known faces with labels
# 4. Predict label for new face
# ─────────────────────────────────────────────

TRAIN_DIR = "../Faces/train"
VAL_DIR   = "../Faces/val"
IMG_SIZE  = (100, 100)   # normalize all faces to same size

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ─────────────────────────────────────────────
# 1. LOAD TRAINING DATA
# Assign integer label to each person
# ─────────────────────────────────────────────
def load_faces_from_dir(directory):
    """Returns: faces (list of grayscale arrays), labels (list of ints), label_map (int→name)"""
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    people = sorted(os.listdir(directory))

    for person_name in people:
        person_path = os.path.join(directory, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person_name
        print(f"  Loading: {person_name} (label={label_id})")

        img_files = [f for f in os.listdir(person_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        loaded = 0
        for img_file in img_files:
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Try to detect face in image
            face_rects = face_cascade.detectMultiScale(gray,
                                                        scaleFactor=1.1,
                                                        minNeighbors=5,
                                                        minSize=(30, 30))

            if len(face_rects) > 0:
                # Use largest detected face
                (x, y, w, h) = max(face_rects, key=lambda r: r[2]*r[3])
                face = gray[y:y+h, x:x+w]
            else:
                # No face detected — use entire image
                face = gray

            # Resize to uniform size
            face = cv2.resize(face, IMG_SIZE)

            faces.append(face)
            labels.append(label_id)
            loaded += 1

        print(f"    → {loaded} images loaded")
        label_id += 1

    return faces, labels, label_map


print("Loading training data...")
train_faces, train_labels, label_map = load_faces_from_dir(TRAIN_DIR)
print(f"\nTotal training samples: {len(train_faces)}")
print(f"People: {label_map}\n")

# ─────────────────────────────────────────────
# 2. CREATE AND TRAIN RECOGNIZER
# LBPH = Local Binary Pattern Histogram
# Describes face texture using local binary patterns
# Fast, works well with small datasets
# ─────────────────────────────────────────────
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,       # radius of circular LBP
    neighbors=8,    # number of sample points on circle
    grid_x=8,       # number of cells in X direction
    grid_y=8        # number of cells in Y direction
)

# Train the recognizer
recognizer.train(train_faces, np.array(train_labels))
print("Training complete.")

# Save model (so you don't need to retrain each time)
recognizer.save("face_recognizer.yml")
print("Model saved: face_recognizer.yml")

# ─────────────────────────────────────────────
# 3. PREDICT ON VALIDATION DATA
# ─────────────────────────────────────────────

# To load saved model:
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("face_recognizer.yml")

print("\n--- Validation Results ---")
correct = 0
total = 0
results = []

for person_name in sorted(os.listdir(VAL_DIR)):
    person_path = os.path.join(VAL_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    # Find the true label for this person
    true_label = None
    for lid, name in label_map.items():
        if name.lower().replace(" ", "_") == person_name.lower() or \
           name.lower() == person_name.lower():
            true_label = lid
            break

    img_files = [f for f in os.listdir(person_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in img_files:
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_rects = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(face_rects) > 0:
            (x, y, w, h) = max(face_rects, key=lambda r: r[2]*r[3])
            face = gray[y:y+h, x:x+w]
        else:
            face = gray

        face = cv2.resize(face, IMG_SIZE)

        # predict() returns (label, confidence)
        # Lower confidence = BETTER match (it's a distance metric)
        pred_label, confidence = recognizer.predict(face)
        pred_name = label_map.get(pred_label, "Unknown")

        is_correct = (true_label == pred_label) if true_label is not None else None
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "file": img_file,
            "true": person_name,
            "pred": pred_name,
            "confidence": confidence,
            "correct": is_correct
        })

        print(f"  {person_name}/{img_file}")
        print(f"    → Predicted: {pred_name} (confidence: {confidence:.1f})")
        print(f"    → {'CORRECT' if is_correct else 'WRONG'}")

accuracy = correct / total * 100 if total > 0 else 0
print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")

# ─────────────────────────────────────────────
# 4. VISUALIZE PREDICTIONS
# ─────────────────────────────────────────────
print("\nVisualizing predictions...")

for person_name in sorted(os.listdir(VAL_DIR)):
    person_path = os.path.join(VAL_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    img_files = [f for f in os.listdir(person_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        continue

    img_path = os.path.join(person_path, img_files[0])
    img = cv2.imread(img_path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    if len(face_rects) > 0:
        (x, y, w, h) = max(face_rects, key=lambda r: r[2]*r[3])
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, IMG_SIZE)
        pred_label, confidence = recognizer.predict(face_resized)
        pred_name = label_map.get(pred_label, "Unknown")

        color = (0, 255, 0) if pred_name.lower() in person_name.lower() or \
                               person_name.lower() in pred_name.lower() else (0, 0, 255)

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, f"{pred_name}", (x, y-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img, f"conf: {confidence:.0f}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow(f"True: {person_name}", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# OTHER RECOGNIZERS (also in cv2.face)
# ─────────────────────────────────────────────
# cv2.face.EigenFaceRecognizer_create()  — PCA-based
# cv2.face.FisherFaceRecognizer_create() — LDA-based
# All have same API: train(), predict(), save(), read()
# LBPH is generally most robust for small datasets

# ─────────────────────────────────────────────
# EXERCISE
# ─────────────────────────────────────────────
# 1. Try EigenFaceRecognizer — does it perform better or worse?
# 2. Change IMG_SIZE to (50,50) and (200,200) — how does accuracy change?
# 3. Add "Unknown" detection: if confidence > threshold, say "Unknown"
#    Find a good threshold value by experimenting
# 4. Run recognition on Videos/dog.mp4 — detect + recognize any human faces
