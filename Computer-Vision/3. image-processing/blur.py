import cv2

img = cv2.imread("Resources/Photos/park.jpg")

# we can set the kernel to adjust how much we want to blur it
# kernel size
# (5,5)
# (7,7)
# (15,15)
# the bigger the kernel the stronger it blurs
blur_small = cv2.GaussianBlur(img,(5,5),0)
blur_large = cv2.GaussianBlur(img,(15,15),0)

cv2.imshow("Original", img)
cv2.imshow("Gaussian Blur 5x5", blur_small)
cv2.imshow("Gaussian Blur 15x15", blur_large)

cv2.waitKey(0)
cv2.destroyAllWindows()

# blurring smooths the image and reduces noise
# we usually do this before edge detection
# there are different kind of blur methods such as :
# Gaussian Blur, Median Blur, Average Blur.