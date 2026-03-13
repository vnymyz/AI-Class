import cv2

img = cv2.imread("Resources/Photos/park.jpg")

# OpenCV Usually Loads image as BGR
# convert color spaces
# HSV usually used for color detection like
# detecting red object, traffic loghts or fruits
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("Original BGR", img)
cv2.imshow("Grayscale", gray)
cv2.imshow("HSV", hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()