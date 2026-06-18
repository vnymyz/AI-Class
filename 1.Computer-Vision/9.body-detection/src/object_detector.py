"""Lightweight object detection via YOLOv8n (ultralytics). Runs every N frames to save CPU/GPU."""
import cv2
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.45, device=None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device  # "cpu", "0" (cuda:0), or None for auto

    def detect(self, frame_bgr):
        """Run detection on a frame. Returns list of (label, conf, (x1,y1,x2,y2))."""
        results = self.model.predict(
            frame_bgr, conf=self.conf, device=self.device, verbose=False
        )
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((label, conf, (x1, y1, x2, y2)))
        return detections

    @staticmethod
    def draw(frame_bgr, detections):
        for label, conf, (x1, y1, x2, y2) in detections:
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame_bgr,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
        return frame_bgr
