from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2 as cv
import numpy as np
import sys

tracker = cv.TrackerCSRT_create()

if len(sys.argv) > 2:
    video = cv.VideoCapture(sys.argv[1])
else:
    video = cv.VideoCapture(0)
    

# Read the first frame, to begin tracking and select ROI
ok, frame = video.read()
bbox = cv.selectROI(frame, False)
cv.destroyAllWindows()
ok = tracker.init(frame,bbox)



while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
         break
# Start timer
    timer = cv.getTickCount()

# Update tracker
    ok, bbox = tracker.update(frame)

# Calculate Frames per second (FPS)
    fps = cv.getTickFrequency() / (cv.getTickCount() - timer);

# Draw bounding box
    if ok:
    # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
    # Tracking failure
        cv.putText(frame, "Tracking failure detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

# Display tracker type on frame
    cv.putText(frame, "GOTURN Tracker", (100,20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

# Display FPS on frame
    cv.putText(frame, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

# Display result
    cv.imshow("Tracking", frame)

# Exit if ESC pressed
    k = cv.waitKey(1) & 0xff
    if k == 27:
        break

