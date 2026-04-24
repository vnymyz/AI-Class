# this is just a computer vision but without the deep learning
import cv2

img = cv2.imread("Resources/Photos/cat.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray,(5,5),0)

edges = cv2.Canny(blur,100,200)

contours,_ = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,contours,-1,(0,255,0),2)

cv2.imshow("Edges", edges)
cv2.imshow("Contours", img)

cv2.waitKey(0)
cv2.destroyAllWindows()