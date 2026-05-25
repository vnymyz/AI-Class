from ultralytics import YOLO
import cv2
import numpy as np
from utils import power_map

# load model once
model = YOLO("yolov8n.pt")

def detect(img):

    # resize for consistency
    img = cv2.resize(img, (640, 640))

    # optional sharpening
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    # run YOLO
    results = model(img, imgsz=1280, conf=0.25)

    # store counts
    detected_counts = {}

    # loop detections
    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            # count only relevant objects
            if label in power_map:
                if label in detected_counts:
                    detected_counts[label] += 1
                else:
                    detected_counts[label] = 1

            # draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            watt = power_map.get(label, 0)
            display_label = f"{label} - {watt}W"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, display_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # calculate total power
    total_power = 0
    for item, count in detected_counts.items():
        total_power += power_map[item] * count

    return img, total_power, detected_counts

cv2.putText(img,
            f"Cost/hour: Rp {int(cost)}",
            (20,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,(0,255,255),2)

cv2.putText(img,
            f"Power: {total_power}W",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,(0,0,255),2)

cv2.imshow("Electricity Estimator", img)
cv2.waitKey(0)
cv2.destroyAllWindows()