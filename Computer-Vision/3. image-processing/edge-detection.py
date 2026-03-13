import cv2

# try using different images
img = cv2.imread("Resources/Photos/group-1.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# for making an outline 
# to find object boundaries
# example like finding a :
# cat outline, car edges, building borders
# theres also an algorithm for it and its called
# Canny Edge Detection
edges = cv2.Canny(gray,100,200)

# the gray,100,200 is for adjusting the threshold
# 100 → lower threshold
# 200 → upper threshold

cv2.imshow("Original", img)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()