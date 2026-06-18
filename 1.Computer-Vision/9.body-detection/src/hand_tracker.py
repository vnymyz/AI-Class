"""Hand + finger landmark tracking via MediaPipe Tasks API, with fingertip motion trail."""
import collections
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def _ensure_model(model_path):
    if os.path.exists(model_path):
        return
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, model_path)


class HandTracker:
    INDEX_FINGER_TIP = 8

    def __init__(self, max_hands=2, detection_conf=0.6, tracking_conf=0.6, trail_len=20,
                 model_path=_MODEL_PATH):
        _ensure_model(model_path)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.trails = collections.defaultdict(lambda: collections.deque(maxlen=trail_len))
        self._last_positions = {}
        self._last_time = time.time()
        self._start_time = time.time()

    def process(self, frame_bgr):
        """Detect hands, draw landmarks + fingertip trail + speed. Returns annotated frame + hand count."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        now = time.time()
        dt = max(now - self._last_time, 1e-6)
        self._last_time = now

        num_hands = 0
        if result.hand_landmarks:
            num_hands = len(result.hand_landmarks)
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                for connection in HAND_CONNECTIONS:
                    start = hand_landmarks[connection.start]
                    end = hand_landmarks[connection.end]
                    pt1 = (int(start.x * w), int(start.y * h))
                    pt2 = (int(end.x * w), int(end.y * h))
                    cv2.line(frame_bgr, pt1, pt2, (0, 200, 0), 2)
                for lm in hand_landmarks:
                    cv2.circle(frame_bgr, (int(lm.x * w), int(lm.y * h)), 3, (0, 0, 255), -1)

                tip = hand_landmarks[self.INDEX_FINGER_TIP]
                cx, cy = int(tip.x * w), int(tip.y * h)
                self.trails[idx].append((cx, cy))

                prev = self._last_positions.get(idx)
                speed = 0.0
                if prev is not None:
                    dist = ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5
                    speed = dist / dt
                self._last_positions[idx] = (cx, cy)

                cv2.putText(
                    frame_bgr,
                    f"hand{idx} speed:{speed:.0f}px/s",
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
        else:
            self.trails.clear()
            self._last_positions.clear()

        return frame_bgr, num_hands

    def close(self):
        self.landmarker.close()
