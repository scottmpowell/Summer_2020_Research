from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2 as cv
import numpy as np
import sys

#tracker = cv.Tracker_create("boosting")
#tracker = cv.TrackerCSRT_create

cap = cv.VideoCapture(sys.argv[1])
while(True):
    ret, frame = cap.read()
    cv.imshow("name",frame)
    key = cv.waitKey(10)
    if key == ord('q'):
        break

