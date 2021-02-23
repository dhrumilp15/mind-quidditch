import cv2


def screenDebug(frame, *messages):
    '''
    Displays a given string to the window supplied by frame
    :param frame: The window on which the message will be displayed
    :param message: The string to show on the given frame
    '''
    height, width, channels = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    defwidth = 10
    defheight = height - 20
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 1
    for index, message in enumerate(messages):
        cv2.putText(frame, message, (defwidth, defheight - index * 30),
                    font, fontScale, fontColor, thickness, cv2.LINE_AA)
