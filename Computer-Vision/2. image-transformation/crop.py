import cv2

img = cv2.imread("Resources/Photos/group-1.jpg")

# Crop the image using array slicing
# The format is img[y1:y2, x1:x2]
# 100:400 → height
# 200:500 → width
# images are numpy arrays (height , width , channels)
crop = img[100:400, 200:500]

cv2.imshow("Original", img)
cv2.imshow("Cropped", crop)

cv2.waitKey(0)
cv2.destroyAllWindows()