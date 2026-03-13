# 1. load 5 images and print their shapes
# 2. convert images to grayscale
# 3. resize images
import cv2

# list of images to load
image_paths = [
    "Resources/Photos/cat.jpg",
    "Resources/Photos/cat_large.jpg",
    "Resources/Photos/cats-2.jpg",
    "Resources/Photos/group-1.jpg",
    "Resources/Photos/park.jpg"
]

# Looping for printing the shape of each images
for path in image_paths:
    img = cv2.imread(path)

    print("Image:", path)
    print("Shape:", img.shape)
    print("----------------------")

# Output should be
# Image: Resources/Photos/cat.jpg
# Shape: (427, 640, 3)
# ----------------------
# Image: Resources/Photos/cat_large.jpg
# Shape: (1600, 2400, 3)
# ----------------------
# Image: Resources/Photos/cats-2.jpg
# Shape: (427, 640, 3)
# ----------------------
# Image: Resources/Photos/group-1.jpg
# Shape: (405, 640, 3)
# ----------------------
# Image: Resources/Photos/park.jpg
# Shape: (427, 640, 3)
# ----------------------