import cv2

cap = cv2.VideoCapture(0)

# read first frame
ret, prev_frame = cap.read()
prev_frame = cv2.flip(prev_frame, 1)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21,21), 0)

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    # compute difference
    diff = cv2.absdiff(prev_gray, gray)

    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        if cv2.contourArea(contour) < 2000:
            continue

        x,y,w,h = cv2.boundingRect(contour)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(
            frame,
            "Motion Detected",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

    cv2.imshow("Motion Detector", frame)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()