# threshold is for converting image into binary format
# example
# 0 → black
# 255 → white
# it is used for :
# object segmentation, document scanning like camscanner, shape detection.
import cv2

img = cv2.imread("Resources/Photos/group-1.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

cv2.imshow("Original", img)
cv2.imshow("Threshold", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()

