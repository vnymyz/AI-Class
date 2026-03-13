import cv2

img = cv2.imread("Resources/Photos/group-1.jpg")

flip_horizontal = cv2.flip(img,1)
flip_vertical = cv2.flip(img,0)

cv2.imshow("Original", img)
cv2.imshow("Horizontal Flip", flip_horizontal)
cv2.imshow("Vertical Flip", flip_vertical)

cv2.waitKey(0)
cv2.destroyAllWindows()