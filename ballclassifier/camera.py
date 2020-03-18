import cv2
from imutils.video import VideoStream

vs = cv2.VideoCapture(0)#.start()

while 1:
    ret, frame = vs.read()
    if frame is None:
        break
    cv2.imshow('img', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break