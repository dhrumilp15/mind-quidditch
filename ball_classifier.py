import numpy as np
import cv2
from collections import deque
import argparse
import time
import os
import json

# plushie
# balllow = np.array([32, 62, 67])
# ballhigh = np.array([51, 255, 255])

# Green
# balllow = np.array([28,98,63])
# ballhigh = np.array([65,255,255])

# # Another
# balllow = np.array([35,43,117])
# ballhigh = np.array([57,113,152])

# Apple
# balllow = np.array([18, 119, 81])
# ballhigh = np.array([39, 255, 255])

# Orange
balllow = np.array([14,166,145])
ballhigh = np.array([21,255,255])

class BallClassifier:
    def __init__(self, args):        
        self.pts = deque(maxlen=args["buffer"])
        self.args = args
        self.vs = None
        self.testfile = "ball_images/images.json"
        self.outfile = "ball_images/report.json"
        self.record = []
        self.iourecord = dict()
        self.depthrecord = dict()
        try:
            open(self.outfile, 'r').close()
        except FileNotFoundError:
            print("Outfile doesn't exist")
        print(f"Get test arg: {args.get('test', False)}")
        if args.get("video", False):
            self.vs = cv2.VideoCapture(args["video"])
        elif args.get("test", False):
            filenames = [name for name in os.listdir("ball_images/") if name.endswith(".jpg")]
            filenames.sort(key = lambda x: int(x[:x.index(".jpg")]))
            # print(filenames)
            images = [cv2.imread("ball_images/" + img) for img in filenames]
            # print(images)
            self.imgiter = iter(images)
        else:
            self.vs = cv2.VideoCapture(0)
        
        if self.vs is not None:
            self.vs.set(cv2.CAP_PROP_FPS, 30)
            self.vs.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
            self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    
    def screenDebug(self, frame, *messages):
        '''
        Prints a given string to the window supplied by frame
        :param frame: The window on which the message will be displayed
        :param message: The string to show on the given frame
        '''
        height, width, channels = frame.shape
        font                    = cv2.FONT_HERSHEY_SIMPLEX
        defwidth                = 10
        defheight               = height - 20
        fontScale               = 1
        fontColor               = (255,255,255)
        thickness               = 1
        for index, message in enumerate(messages):
            cv2.putText(frame, message, (defwidth, defheight - index * 30), font, fontScale, fontColor, thickness, cv2.LINE_AA)
    
    def get_dist(self, radius, ref_dist = (101, 5)) -> float:
        '''
            Gets the distance in inches from the object to the camera
            :param radius: The radius of the ball (in px)
            :param ref_dist: Conditions that are true, calculated with the camera's focal length
            :return: The distance from the object to the camera
        '''
        return np.prod(ref_dist) / radius

    def get_hough_frame(self, frame, x,y, radius, multiplier = 1.5) -> list:
        '''
        Gets a smaller window of the frame for the Hough transform
        :param frame: The original frame
        :param x: The x value of the center of the current best circle
        :param y: The y value of the center of the current base circle
        :param multiplier: The value to multiply the raidus by since the colour mask underestimates the true area of the ball
        :return: A smaller window as an ndarray
        '''
        if x and y:
            # x, y = x, y
            ymin = int(max(0, y - multiplier * radius))
            ymax = int(min(frame.shape[0], y + multiplier * radius))
            xmax = int(max(0, x - multiplier * radius))
            xmin = int(min(frame.shape[1], x + multiplier * radius))
        # Simply returning frame[ymin:ymax, xmax: xmin] would be great, but then we get weird potential problems in 
        # Reduce frame to a circle?
        # givenframe = frame[ymin:ymax, xmax:xmin]
        
        # print(f"ymin: {y1min}, ymax: {ymax}, xmax: {xmax}, xmin: {xmin}")
        return frame[ymin:ymax, xmax:xmin]
    
    def hough(self, frame, center, radius):
        # Canny edge detection
        high = 75
        graysc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dp = 1.2
        minDist = 10
        # accthresh = 10
        rad_mult = 1.7
        smallerframe = self.get_hough_frame(frame=graysc, x=center[0], y=center[1], radius=radius, multiplier=rad_mult)
        # edges = cv2.Canny(smallerframe, high // 2, high)
        # cv2.imshow('canny', edges)
        circles = cv2.HoughCircles(smallerframe, cv2.HOUGH_GRADIENT, dp, minDist, param1=high, param2=50,minRadius=0,maxRadius=200)
        print(circles)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            xadj, yadj, radius = circles[0,:][0]
            cv2.circle(smallerframe, (xadj, yadj), radius, (255, 0, 0), thickness=2)
            cv2.circle(smallerframe, (xadj, yadj), 2, (0, 255, 0), thickness=2)
            return ((center[0] + int(xadj - rad_mult * radius), center[1] + int(yadj - rad_mult * radius)), radius)
    
    def colour_mask(self, frame):
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # Colour Mask Implementation:
        mask = cv2.inRange(hsv, balllow, ballhigh)
        mask = cv2.erode(mask, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=3)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)
        # cv2.imshow('mask', mask)
        cnts, hier = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        radius = None
        y, x = None, None
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x,y), radius) = cv2.minEnclosingCircle(c)
            if radius < 10:
                return
            M = cv2.moments(c)
            return ((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])), radius)

    def predict(self):
        # Quadratic fit approach
            # Ideally, we'd like to give a higher weight to more recent values, since they'll (hopefully) be more accurate
            # The accuracy of the model at certain distances has yet to be established, but I think it's a little like a
            # gaussian distribution - so we'll have to play around with the weights

            # if len(set(pts)) > 5: # Only if there are more than three distinct values
                # coeffs, res, rank, singular_values, rcond = np.polyfit(x= xtrace, y= ytrace, deg= 2,full= True) # For testing purposes
                # print(coeffs)
                # coeffs = np.polyfit(x = xtrace,y =  ytrace,deg =  2) # For real use
        pass
    
    def iou(self, c1, c2, r, R):
        dist = np.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
        if dist <= abs(R-r):
            return np.pi * (min(R,r) / max(R,r))**2
        elif dist >= r + R:
            return 0
        r2, R2, dist2 = r**2, R**2, dist**2
        alpha = np.arccos((dist2 + r2 - R2) / (2*dist*r))
        beta = np.arccos((dist2 + R2 - r2) / (2*dist*R))
        intersection = r2 * alpha + R2 * beta - 0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta))
        union = np.pi * (r2 + R2) - intersection
        return intersection / union

    def generateReport(self):
        print("-----------------")
        with open(self.testfile, 'r') as f:
            data = json.loads(f.read())
        for label, info in data.items():
            label = int(label)
            c1, r1 = info[0], float(info[1])
            c1 = [int(c1[0]), int(c1[1])]
            c2, r2 = self.record[label - 2]
            iou = self.iou(c1, c2, r1, r2)
            self.iourecord[label] = iou
            self.depthrecord[label] = (label - self.get_dist(r2)) / label
        print(f"Depth Delta Std. Dev: {np.std(list(self.depthrecord.values())):.4f}")
        print(f"Average IOU: {np.sum(list(self.iourecord.values())) / len(self.iourecord) * 100: .4f}%")
    
    def main(self):
        while True:
            # Capture frame-by-frame
            if self.args.get("test", False):
                try:
                    frame = next(self.imgiter)
                except StopIteration:
                    self.generateReport()
                    with open(self.outfile, 'w') as f:
                        f.write(json.dumps(self.iourecord, indent = 4))
                    break
            else:
                ret, frame = self.vs.read()
                # print(frame)
                if not ret:
                    break

            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            frame = cv2.bilateralFilter(frame, 5, 100, 100)

            cres = self.colour_mask(frame)
            center, radius = None, None
            if cres is not None:
                center, radius = cres
                res = self.hough(frame, center, radius)
                if res is not None:
                    center, radius = res
            
            if self.args.get("test", False):
                self.record.append((center, radius))
                print(f"Estimated Distance: {self.get_dist(radius)}")
                # while True:
                #     cv2.imshow('frame', frame)
                #     key = cv2.waitKey(1) & 0xFF
                #     if key == ord('o'):
                #         cv2.destroyAllWindows()
                #         break
            else:
                if center is not None and radius is not None:
                    self.pts.appendleft(center)
                    self.screenDebug(frame, f"radius(px): {radius:.4f}", f"Distance(in):{self.get_dist(radius):.4f}")
                    cv2.circle(img=frame,center=center, radius= int(radius), color= (0,255,0), thickness=2)
                    cv2.circle(img=frame,center=center, radius=2, color= (255,0,0), thickness=2)

                cv2.imshow('frame', frame)
                # cv2.imshow('Colour mask',mask)
                # cv2.imshow('Laplacian', abs_dst)s

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.vs.release()
        # When everything done, release the capture
        cv2.destroyAllWindows()

def configure_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
    ap.add_argument("-t", "--test", type=bool, default=False, help="Whether to conduct tests on the `ball_images` folder")
    print(vars(ap.parse_args()))
    return vars(ap.parse_args())

if __name__ == "__main__":
    args = configure_args()
    # if not args.get("test", False):
    #     arg = input("Are you sure that test should be negative? ")
    #     if arg:
    #         args["test"] = True
    bc = BallClassifier(args)
    bc.main()
