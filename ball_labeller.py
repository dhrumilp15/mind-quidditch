import cv2
import numpy as np
import os
import imutils
import json

class Ball_Labeller:
    def __init__(self):
        self.outfile = "ball_images/images.json"
        open(self.outfile, 'w').close()
        filenames = [name[:name.index(".jpg")] for name in os.listdir("ball_images/") if name.endswith(".jpg")]
        filenames.sort(key = lambda x: int(x))
        images = [cv2.imread("ball_images/" + img + ".jpg") for img in filenames]
        # print(images)
        self.imgiter = iter(images)
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_radius)
        self.record = dict()
        self.center = None
        self.radius = None
    
    def click_radius(self, event, x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Click at {(x,y)}")
            if self.center:
                self.radius = np.sqrt((y - self.center[1])**2 + (x - self.center[0])**2)
            else:
                self.center = (x,y)

    def main(self):
        counter = 2
        while True:
            try:
                frame = next(self.imgiter)
            except StopIteration:
                break
            frame = imutils.resize(frame, width=600)
            self.center = None
            self.radius = None
            while True:
                cv2.imshow("image", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    self.record[counter] = [self.center, self.radius]
                    break
            counter += 1
        with open(self.outfile, 'w') as f:
            f.write(json.dumps(self.record, indent=4))
        cv2.destroyAllWindows()
if __name__ == "__main__":
    BL = Ball_Labeller()
    BL.main()
