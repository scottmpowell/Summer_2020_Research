from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2 as cv
import numpy as np
import sys



cv.namedWindow("Tracking")

class trackee:

    # Inititializer
    def __init__(self, bbox, number):
        self.tracker = cv.TrackerCSRT_create()
        self.bbox = bbox
        self.number = number
        self.tracker.init(frame, self.bbox)

    def __str__(self):
        data = "Tracking Object: " + str(self.number) + " Bbox:" + str(self.bbox)
        return data



"""
Begin tracking object. Video will pause on current frame and allow selection of a ROI to be tracked
"""
def begin_track():
    global frame, bbox, video, is_tracking, trackers
    tracker_counter = 1
    while True: 
        bbox = cv.selectROI(frame, False)
        if bbox == (0,0,0,0):
            break

        trackers[tracker_counter] = trackee(bbox, tracker_counter)
        tracker_counter += 1

        is_tracking = True
    cv.destroyAllWindows()

if __name__ == "__main__":

    """
    Initialize global variables. Additionally, create an empty dictionary and set is_tracking to False until an object is selected
    """
    global frame, bbox, video, tracker, is_tracking, trackers, pause
    is_tracking = False
    trackers = dict()

    """
    If a file is specified, open the file, otherwise take from the camera
    """
    if len(sys.argv) == 2:
        video = cv.VideoCapture(sys.argv[1])
    else:
        video = cv.VideoCapture(0)

    video.set(cv.CAP_PROP_FPS, 100)



    while True:
        # Read a new frame
        cap_ret, frame = video.read()
        if not cap_ret:
            break
        # Start timer
        timer = cv.getTickCount()

        # Update tracker, if applicable
        if is_tracking:
            for key in trackers:
                track_ret, trackers[key].bbox = trackers[key].tracker.update(frame)
                # Draw bounding box
                if track_ret:
                    # Tracking success
                    p1 = (int(trackers[key].bbox[0]), int(trackers[key].bbox[1]))
                    p2 = (int(trackers[key].bbox[0] + trackers[key].bbox[2]), int(trackers[key].bbox[1] + trackers[key].bbox[3]))
                    cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)

                else :
                    # Tracking failure
                    cv.putText(frame, "Tracking failure detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)


        # Calculate Frames per second (FPS)
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer);


        # Display tracker type on frame
        cv.putText(frame, "GOTURN Tracker", (100,20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

        # Display FPS on frame
        cv.putText(frame, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        cv.imshow("Tracking", frame)

        # Exit if q pressed
        k = cv.waitKey(5) & 0xff
        if k == ord('q'):
            break
        elif k == ord('p'):
            pause = not pause
        elif k == ord('s'):
            cv.destroyAllWindows()
            begin_track()
