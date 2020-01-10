import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    original = frame

    frame = cv2.GaussianBlur(frame,(3,3), 0)
    graysc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    balllow = np.array([0,132,162])
    ballhigh = np.array([26,255,255])

    # Colour Mask Implementation:
    mask = cv2.inRange(hsv, balllow, ballhigh)

    # Laplacian filter Implementation:
    dst = cv2.Laplacian(graysc, cv2.CV_16S, ksize=3)
    abs_dst = cv2.convertScaleAbs(dst)

    # Display the resulting frame
    cv2.imshow('original', original)    
    cv2.imshow('frame', frame)
    cv2.imshow('Colour mask',mask)
    cv2.imshow('Laplacian', abs_dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()