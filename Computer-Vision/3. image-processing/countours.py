# countour is for detecting object outlines
# it used for :
# shape detection, object counting, size measurement
# thats why we need threshold since countour works best with binary format images
import cv2

img = cv2.imread("Resources/Photos/group-1.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

contours,_ = cv2.findContours(
    thresh,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

# this is for drawing the countours
# (0,255,0) → green color
# 2 → thickness
cv2.drawContours(img, contours, -1, (0,255,0), 2)

cv2.imshow("Contours", img)

cv2.waitKey(0)
cv2.destroyAllWindows()