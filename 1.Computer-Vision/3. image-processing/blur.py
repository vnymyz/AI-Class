import cv2

img = cv2.imread("Resources/Photos/group-1.jpg")

# Average blur
blur_avg_small = cv2.blur(img,(5,5))
blur_avg_large = cv2.blur(img,(15,15))

# Gaussian blur
blur_gaussian = cv2.GaussianBlur(img,(15,15),0)

# Median blur
blur_median = cv2.medianBlur(img,25)

cv2.imshow("Original", img)
cv2.imshow("Average Blur 5x5", blur_avg_small)
cv2.imshow("Average Blur 15x15", blur_avg_large)
cv2.imshow("Gaussian Blur", blur_gaussian)
cv2.imshow("Median Blur", blur_median)

cv2.waitKey(0)
cv2.destroyAllWindows()