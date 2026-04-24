import cv2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    # mirror the frame
    # if its 1 then flip horizontal
    # 0 its vertical
    # -1 both direction
    mirror = cv2.flip(frame, 1)

    cv2.imshow("Mirrored Webcam", mirror)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()