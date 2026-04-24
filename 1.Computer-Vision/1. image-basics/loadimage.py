# Load and display an image using OpenCV and Matplotlib
import cv2
import matplotlib.pyplot as plt

# Load the image from the specified path
img = cv2.imread("Resources/Photos/cats-2.jpg")

# Check if the image was loaded successfully
# the size or shape of the image
print(img.shape)

# Display the image using Matplotlib
# cvt itu convert
# open cv atau computer vision dia itu default color nya
# BGR atau blue green red
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# Hide the axes for better visualization
plt.axis("off")
# Show the image
plt.show()