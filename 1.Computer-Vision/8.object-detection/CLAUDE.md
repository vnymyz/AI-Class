i already finished an image classification project (animals, MobileNetV2 transfer learning + Streamlit app). now i want to learn object detection.

i've learned: python fundamentals, numpy, pandas, matplotlib, seaborn, scikit-learn, tensorflow/keras, basic CNN + transfer learning, OpenCV basics (read/display, color spaces, drawing, resize/crop, blur/edges/thresholding/contours, background subtraction), and a bit of YOLO already (motion + YOLO folder in OpenCV basics).

## Task

help me build an object detection project using YOLO (pretrained, ultralytics), step by step, since im still learning. this is a learning project, runs locally only (no cloud deploy).

## Role

act as a data science / computer vision expert, guide me step by step — explain concepts simply (im beginner-intermediate, prefer analogies over heavy jargon first).

## Python Libraries

tell me what libraries we need (ultralytics, opencv-python, streamlit) and why. keep it minimal since local-only.

## Data

recommend a good beginner-friendly approach for object detection — pretrained COCO classes (via ultralytics YOLOv8) should be fine for v1 learning, no custom dataset needed unless useful later as stretch goal.

## App requirement

local Streamlit app (run via `streamlit run`), two input modes:
1. upload an image/video file for object detection
2. use local webcam (cv2.VideoCapture, local-only, no streamlit-webrtc needed) for real-time detection

show detected boxes + labels + confidence on output.

## Structure code

keep it clean and maintainable:
- jupyter notebook (.ipynb) for exploration/testing the model
- separate .py files for the streamlit app, model loading, inference logic
- clear folder structure (data/, src/, app/, models/, notebooks/)
