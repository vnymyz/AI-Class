"""Streamlit demo: upload an animal image, get a class prediction."""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Allow `from src...` imports when running `streamlit run app/app.py` from project root.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model import load_model
from src.utils import preprocess_image

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "animal_classifier.keras"

# Update this list to match the class_names order printed during training.
CLASS_NAMES = [
    "butterfly", "cat", "chicken", "cow", "dog",
    "elephant", "horse", "sheep", "spider", "squirrel",
]

CLASS_EMOJI = {
    "butterfly": "\U0001F98B", "cat": "\U0001F431", "chicken": "\U0001F414",
    "cow": "\U0001F404", "dog": "\U0001F436", "elephant": "\U0001F418",
    "horse": "\U0001F40E", "sheep": "\U0001F411", "spider": "\U0001F577",
    "squirrel": "\U0001F43F",
}

st.set_page_config(
    page_title="Animal Classifier",
    page_icon="\U0001F43E",
    layout="centered",
    initial_sidebar_state="auto",
)


@st.cache_resource
def get_model():
    return load_model(str(MODEL_PATH))


with st.sidebar:
    st.header("About")
    st.write(
        "This model was fine-tuned from MobileNetV2 (transfer learning) "
        "on the Animals-10 dataset to recognize 10 animal classes."
    )
    st.subheader("Classes")
    st.write(", ".join(f"{CLASS_EMOJI.get(c, '')} {c}" for c in CLASS_NAMES))

st.title(f"{CLASS_EMOJI['dog']} Animal Image Classifier")
st.caption("Upload a photo and the model will guess which animal it is.")

uploaded_file = st.file_uploader(
    "Drag and drop or browse an image",
    type=["jpg", "jpeg", "png"],
    help="JPG or PNG, any size — it'll be resized automatically.",
)

if uploaded_file is None:
    st.info("Upload an image to get started.")
else:
    image = Image.open(uploaded_file)

    col_image, col_result = st.columns([1, 1], gap="large")

    with col_image:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col_result:
        with st.spinner("Classifying..."):
            model = get_model()
            batch = preprocess_image(image)
            predictions = model.predict(batch, verbose=0)[0]

        top_idx = int(np.argmax(predictions))
        top_label = CLASS_NAMES[top_idx]
        top_confidence = float(predictions[top_idx])

        st.metric(
            label="Prediction",
            value=f"{CLASS_EMOJI.get(top_label, '')} {top_label.capitalize()}",
            delta=f"{top_confidence:.1%} confidence",
        )

        st.write("**All class probabilities**")
        ranked = sorted(zip(CLASS_NAMES, predictions), key=lambda x: x[1], reverse=True)
        for label, score in ranked:
            st.progress(float(score), text=f"{CLASS_EMOJI.get(label, '')} {label} — {score:.1%}")
