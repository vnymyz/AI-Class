# NORMAL DETECTION FOR TESTING PURPOSES
# from ultralytics import YOLO
# import cv2

# model = YOLO("yolov8n.pt")

# img = cv2.imread("images/livingroom.jpg")

# results = model(img)

# for r in results:
#     for box in r.boxes:
#         cls = int(box.cls[0])
#         label = model.names[cls]

#         x1,y1,x2,y2 = map(int, box.xyxy[0])

#         cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
#         cv2.putText(img,label,(x1,y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

# cv2.imshow("Result", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# COMBINE DETECTION AND CALCULATE THE ELECTRICITY USAGE
from ultralytics import YOLO
import cv2
import numpy as np
from utils import calculate_power, calculate_cost, power_map

model = YOLO("yolov8n.pt")

img = cv2.imread("images/kitchen.jpg")

# RESIZE IMAGE FOR BETTER PERFORMANCE
img = cv2.resize(img, (640, 640))

# SHARPENING
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
img = cv2.filter2D(img, -1, kernel)

# Run YOLO
results = model(img, imgsz=1280, conf=0.25)

detected_objects = []

for r in results:
    for box in r.boxes:
        
        cls = int(box.cls[0])
        label = model.names[cls]

        detected_objects.append(label)

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        # modify label to match power_map keys
        watt = power_map.get(label, 0)
        
        # create label with power info
        display_label = f"{label} - {watt}W"
        
        # put the label on the image
        cv2.putText(img, display_label, (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

total_power = calculate_power(detected_objects)

cost = calculate_cost(total_power)

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