from ultralytics import YOLO
import cv2

# load pretrained model
model = YOLO("yolov8n.pt")

# load image
# try using gruop 2
img = cv2.imread("Resources/Photos/group-1.jpg")

# run detection
results = model(img)

people_count = 0

for result in results:
    
    boxes = result.boxes
    
    for box in boxes:
        
        cls = int(box.cls[0])
        
        if cls == 0:  # class 0 = person
            
            people_count += 1
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

print("People detected:", people_count)

cv2.putText(img,
            f"People: {people_count}",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2)

cv2.imshow("People Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()