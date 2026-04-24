import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from utils import calculate_power, calculate_cost, power_map

st.title("⚡ Electricity Estimator AI")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

model = YOLO("yolov8n.pt")

if uploaded_file is not None:

    # read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img = cv2.resize(img, (640, 640))

    results = model(img, imgsz=1280, conf=0.25)

    detected_objects = []

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            detected_objects.append(label)

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            watt = power_map.get(label, 0)
            display_label = f"{label} - {watt}W"

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img, display_label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    total_power = calculate_power(detected_objects)
    cost = calculate_cost(total_power)

    st.image(img, channels="BGR")

    st.subheader(f"🔌 Total Power: {total_power} W")
    st.subheader(f"💰 Cost/hour: Rp {int(cost)}")