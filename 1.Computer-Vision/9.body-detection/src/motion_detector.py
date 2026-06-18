"""General motion detection via background subtraction (MOG2). Catches any movement
in frame (body, arm, waved object), not tied to hand landmarks or object classes."""
import cv2


class MotionDetector:
    def __init__(self, history=500, var_threshold=25, min_area=800):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.min_area = min_area

    def detect(self, frame_bgr):
        """Returns (frame_with_boxes, motion_detected: bool, motion_boxes: list)."""
        fg_mask = self.bg_subtractor.apply(frame_bgr)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            motion_boxes.append((x, y, w, h))
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 140, 255), 2)

        motion_detected = len(motion_boxes) > 0
        return frame_bgr, motion_detected, motion_boxes
