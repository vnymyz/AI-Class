# website
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from utils import calculate_cost, power_map

st.title("⚡ Electricity Estimator AI")

# upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

# load model
model = YOLO("yolov8n.pt")

if uploaded_file is not None:

    # read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # resize
    img = cv2.resize(img, (640, 640))

    # run YOLO
    results = model(img, imgsz=1280, conf=0.25)

    # 🔥 NEW: use dictionary instead of list
    detected_counts = {}

    # detection loop
    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            # only count objects in power_map
            if label in power_map:
                if label in detected_counts:
                    detected_counts[label] += 1
                else:
                    detected_counts[label] = 1

            # bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # get watt
            watt = power_map.get(label, 0)

            # label text
            display_label = f"{label} - {watt}W"

            # draw
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, display_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 🔥 NEW: calculate total power using counts
    total_power = 0
    for item, count in detected_counts.items():
        total_power += power_map[item] * count

    cost = calculate_cost(total_power)

    # show image
    st.image(img, channels="BGR")

    # show total
    st.subheader(f"🔌 Total Power: {total_power} W")
    st.subheader(f"💰 Cost/hour: Rp {int(cost)}")

    # 🔥 NEW: show detected devices (table)
    st.subheader("🔍 Detected Devices")

    if len(detected_counts) == 0:
        st.write("No electrical devices detected.")
    else:
        data = []

        for item, count in detected_counts.items():
            watt = power_map.get(item, 0)

            data.append({
                "Device": item,
                "Count": count,
                "Power per Item (W)": watt,
                "Total Power (W)": watt * count
            })

        df = pd.DataFrame(data)

        st.table(df)