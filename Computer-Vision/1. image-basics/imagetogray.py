# for turning a bunch of images to grayscale
import cv2

# load the images
image_paths = [
    "Resources/Photos/cat.jpg",
    "Resources/Photos/cat_large.jpg",
    "Resources/Photos/cats-2.jpg",
    "Resources/Photos/group-1.jpg",
    "Resources/Photos/park.jpg"
]

# looping to convert each image to grayscale 
# and print the shape of original and grayscale images
for path in image_paths:
    img = cv2.imread(path)

    # turning all the images to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Image:", path)
    print("Original Shape:", img.shape)
    print("Grayscale Shape:", gray.shape)
    print("----------------------")

    # display the grayscale image
    cv2.imshow("Grayscale", gray)
    
    # wait for a key press to move to the next image
    cv2.waitKey(0)

# close all windows after displaying the images
cv2.destroyAllWindows()