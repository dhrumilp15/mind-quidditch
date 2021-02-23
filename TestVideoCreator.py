import cv2

vid = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('output.avi', -1, 20.0, (640, 360))

start = False

while (vid.isOpened()):
    ret, frame = vid.read()
    cv2.resize(frame, (640, 360))
    if ret:
        cv2.imshow('video', frame)
        if cv2.waitKey(0) & 0xFF == ord('s'):
            start = True
        if start:
            videoWriter.write(frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
vid.release()
videoWriter.release()
cv2.destroyAllWindows()
