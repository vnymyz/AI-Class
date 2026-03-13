import cv2

# i wanna try using the group 1 image
# so i can see the difference between crop and resize images
img = cv2.imread("Resources/Photos/group-1.jpg")

# Resize the image to a specific size (width, height)
resized = cv2.resize(img, (224,224))

cv2.imshow("Original", img)
cv2.imshow("Resized", resized)

print(img.shape)
print(resized.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()