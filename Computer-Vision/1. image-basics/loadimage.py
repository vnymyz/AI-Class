# Load and display an image using OpenCV and Matplotlib
import cv2
import matplotlib.pyplot as plt

# Load the image from the specified path
img = cv2.imread("Resources/Photos/cat.jpg")

# Check if the image was loaded successfully
print(img.shape)

# Display the image using Matplotlib
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# Hide the axes for better visualization
plt.axis("off")
# Show the image
plt.show()