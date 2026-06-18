"""
Body/hand motion tracker + lightweight object detector.

Optimized for: i5-11th gen CPU, MX350 2GB VRAM, 16GB RAM.
- MediaPipe Hands: runs every frame (cheap, CPU-only).
- YOLOv8n: runs every DETECT_EVERY_N_FRAMES frames (heavier, CPU or GPU).

Controls: 'q' to quit.
"""
import time

import cv2

from src.hand_tracker import HandTracker
from src.motion_detector import MotionDetector
from src.object_detector import ObjectDetector

DETECT_EVERY_N_FRAMES = 2
CAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
YOLO_DEVICE = "cpu"  # set to "0" to try MX350 GPU (needs torch+cuda installed)
YOLO_CONF = 0.25  # lower = catches smaller/farther objects, more false positives
MOTION_MIN_AREA = 800  # px area threshold to count as motion (lower = more sensitive)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check CAM_INDEX or camera permissions.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Requested {FRAME_WIDTH}x{FRAME_HEIGHT}, camera gave {actual_w}x{actual_h}")

    hand_tracker = HandTracker(max_hands=2)
    detector = ObjectDetector(device=YOLO_DEVICE, conf=YOLO_CONF)
    motion_detector = MotionDetector(min_area=MOTION_MIN_AREA)

    frame_count = 0
    last_detections = []
    prev_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror for natural hand movement

            frame, motion_detected, motion_boxes = motion_detector.detect(frame)

            frame, num_hands = hand_tracker.process(frame)

            if frame_count % DETECT_EVERY_N_FRAMES == 0:
                last_detections = detector.detect(frame)
            frame = ObjectDetector.draw(frame, last_detections)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(
                frame,
                f"FPS:{fps:.0f} hands:{num_hands} objects:{len(last_detections)} motion:{len(motion_boxes)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            if motion_detected:
                cv2.putText(
                    frame, "MOTION DETECTED", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2,
                )

            cv2.imshow("Body Detection - Hands + Objects", frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        hand_tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
