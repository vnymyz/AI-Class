"""Local Streamlit app for YOLOv8 object detection.

Run with: streamlit run app/streamlit_app.py
"""
import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st

# allow `from src...` imports when running via `streamlit run app/streamlit_app.py`
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.detector import load_model, predict
from src.utils import draw_detections

MODEL_PATH = "models/yolov8n.pt"


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title("YOLOv8 Object Detection")

model = get_model()
conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

tab_upload, tab_webcam = st.tabs(["Upload Image/Video", "Webcam"])

# ---------------------------------------------------------------- upload
with tab_upload:
    file = st.file_uploader(
        "Upload an image or video",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    )

    if file is not None:
        suffix = Path(file.name).suffix.lower()
        tmp_path = Path(tempfile.gettempdir()) / file.name
        tmp_path.write_bytes(file.read())

        if suffix in (".jpg", ".jpeg", ".png"):
            frame = cv2.imread(str(tmp_path))
            results = predict(model, frame, conf=conf)
            annotated = draw_detections(frame, results[0])
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Detections", use_container_width=True)

        else:  # video
            cap = cv2.VideoCapture(str(tmp_path))
            stframe = st.empty()
            stop = st.button("Stop")

            while cap.isOpened():
                ok, frame = cap.read()
                if not ok or stop:
                    break
                results = predict(model, frame, conf=conf)
                annotated = draw_detections(frame, results[0])
                stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)

            cap.release()

# ---------------------------------------------------------------- webcam
with tab_webcam:
    st.write("Uses your local webcam (index 0). Check the box to start, uncheck to stop.")
    run = st.checkbox("Start webcam")
    stframe = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ok, frame = cap.read()
            if not ok:
                st.error("Could not read from webcam.")
                break
            results = predict(model, frame, conf=conf)
            annotated = draw_detections(frame, results[0])
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                           channels="RGB", use_container_width=True)
        cap.release()
