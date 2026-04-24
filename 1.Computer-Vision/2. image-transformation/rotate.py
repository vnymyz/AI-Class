import cv2

img = cv2.imread("Resources/Photos/park.jpg")

# Get the dimensions of the image
(h, w) = img.shape[:2]

# Calculate the center of the image
center = (w//2, h//2)

# Get the rotation matrix for a 45 degree rotation around the center of the image
# 1.0 is for scaling the image to make it smaller or bigger
matrix = cv2.getRotationMatrix2D(center, 90, 0.5)

# Perform the rotation using the warpAffine function
rotated = cv2.warpAffine(img, matrix, (w, h))

cv2.imshow("Original", img)
cv2.imshow("Rotated", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()