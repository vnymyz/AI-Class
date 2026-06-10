"""Shared constants and preprocessing helpers used by notebooks and the app."""

import numpy as np
from PIL import Image

IMG_SIZE = (160, 160)

# Animals-10 dataset folder names are in Italian -> map to English labels.
LABEL_TRANSLATE = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize a PIL image and convert it to a model-ready batch of shape (1, H, W, 3)."""
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)
