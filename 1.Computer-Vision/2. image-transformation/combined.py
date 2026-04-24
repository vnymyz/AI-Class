import cv2

img = cv2.imread("Resources/Photos/cat.jpg")

# resize
img = cv2.resize(img,(224,224))

# flip
img = cv2.flip(img,1)

# rotate
(h,w) = img.shape[:2]
center = (w//2, h//2)

matrix = cv2.getRotationMatrix2D(center,30,1.0)
img = cv2.warpAffine(img,matrix,(w,h))

cv2.imshow("Result", img)

cv2.waitKey(0)
cv2.destroyAllWindows()