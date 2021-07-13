from BallClassifier import BallClassifier
import numpy as np
import cv2
import argparse
import os
import json
import logging


class ClassifierMetrics:
    def __init__(self, args):
        img_iter = self.get_img_iter(args["folder"])
        self.BC = BallClassifier({'video': None, 'buffer': 64}, img_iter)
        self.testfile = args["label"]
        self.outfile = args["output"]
        self.report = dict()
        self.iourecord = dict()

    def get_img_iter(self, folder):
        logging.info(f"Looking for image files in {folder}")
        filenames = [name for name in os.listdir(
            folder) if name.endswith(".jpg")]
        if not filenames:
            logging.error("Couldn't find any images!")
            return iter([])
        logging.info(f"Found images in {folder}")
        filenames.sort(key=lambda x: int(x[:x.index(".jpg")]))
        # print(filenames)
        images = [cv2.imread(folder + img) for img in filenames]
        # print(images)
        return iter(images)

    def main(self):
        self.BC.main()
        self.predicted_record = self.BC.record
        self.generateReport()

    def iou(self, c1, c2, r, R):
        dist = np.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
        if dist <= abs(R-r):
            return (min(R, r) / max(R, r))**2
        elif dist >= r + R:
            return 0
        r2, R2, dist2 = r**2, R**2, dist**2
        alpha = np.arccos((dist2 + r2 - R2) / (2*dist*r))
        beta = np.arccos((dist2 + R2 - r2) / (2*dist*R))
        intersection = r2 * alpha + R2 * beta - 0.5 * \
            (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta))
        union = np.pi * (r2 + R2) - intersection
        return intersection / union

    def generateReport(self):
        try:
            f = open(self.testfile, 'r')
        except EnvironmentError:
            logging.error(f"{self.testfile} can't be found!")
        else:
            data = json.loads(f.read())
            logging.info(f"Loaded image labels from {self.testfile}")

        for label, info in data.items():
            label = int(label)
            c1, r1 = info[0], float(info[1])
            c1 = [int(c1[0]), int(c1[1])]
            c2, r2 = self.predicted_record[label - 2]
            iou = self.iou(c1, c2, r1, r2)
            print(f"For label {label}, iou : {iou}")
            self.report[label] = {"iou": iou}
            self.iourecord[label] = iou
        self.report["Averages"] = {"IOU": np.sum(
            list(self.iourecord.values())) / len(self.iourecord)}

        try:
            f = open(self.outfile, 'w')
        except EnvironmentError:
            logging.error(f"{self.outfile} can't be created!")
        else:
            f.write(json.dumps(self.report, indent=4))
            logging.info(f"Wrote report to {self.outfile}")
        print(
            f"Average IOU: {np.sum(list(self.iourecord.values())) / len(self.iourecord) * 100: .4f}%")


def configure_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-f", "--folder", help="path to (optional) folder file", default="ball_images/")
    ap.add_argument("-o", "--output", help="path to (optional) report file",
                    default="ball_images/report.json")
    ap.add_argument("-l", "--label", help="path to (optional) file of labels for images",
                    default="ball_images/images.json")

    print(vars(ap.parse_args()))
    return vars(ap.parse_args())


if __name__ == "__main__":
    args = configure_args()
    CM = ClassifierMetrics(args)
    CM.main()
