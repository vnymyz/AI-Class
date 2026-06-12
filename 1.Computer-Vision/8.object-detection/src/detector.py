"""YOLO model loading + inference wrapper."""
from ultralytics import YOLO


def load_model(model_path: str = "models/yolov8n.pt") -> YOLO:
    """Load a YOLO model from disk (auto-downloads if not present)."""
    return YOLO(model_path)


def predict(model: YOLO, source, conf: float = 0.25):
    """Run inference on an image/frame/video. Returns ultralytics Results list."""
    return model.predict(source=source, conf=conf, verbose=False)
