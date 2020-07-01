"""
Basketball and Player Tracker
Authors: Scott Powell, Christian Newton
"""
# import packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2 as cv
import numpy as np
import sys
import torch
import os

# Personal Modules
from tracker_utils import trackee, delete_tracker, begin_track

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_fullbody.xml")
cv.namedWindow("Video")


def check_commands():
        global pause, is_deleting, frame, show, trackers, is_tracking, imgno
        cv.setMouseCallback("Video", handler)
        k = cv.waitKey(5) & 0xff
        if k == ord('q'):
            sys.exit()
        elif k == ord('p'):
            pause = not pause
        elif k == ord('s'):
            num_trackers = begin_track(frame, trackers)
            if num_trackers > 0:
                is_tracking = True
            else: 
                is_tracking = False
        elif k == ord('w'):
            text = args["output"] + "/img" + str(imgno) + ".png"
            cv.imwrite(text, frame)
            imgno += 1
        elif k == ord('x'):
            if pause:
                if is_deleting == True:
                    show = frame.copy()
                    redraw()

                is_deleting = not is_deleting

# redraw takes no arguments. It updates the frame and tracked objects, and then shows the image.
def redraw():
    global is_tracking, trackers, show
    if is_tracking:
        for key in trackers:
            track_ret, trackers[key].bbox = trackers[key].tracker.update(show)
            # Draw bounding box
            if track_ret:
                # Tracking success
                p1 = (int(trackers[key].bbox[0]), int(trackers[key].bbox[1]))
                p2 = (int(trackers[key].bbox[0] + trackers[key].bbox[2]), int(trackers[key].bbox[1] + trackers[key].bbox[3]))
                cv.rectangle(show, p1, p2, (255,0,0), 2, 1)
            else :
                # Tracking failure
                cv.putText(show, "Tracking failure detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    cv.imshow("Video", show)
    return 0


def handler(event, x, y, flags, param):
    """Handle mouse events, and draw a red 'X' on mouse if paused and the delete key was pressed."""
    global is_deleting, show, pause
    if pause:
        if is_deleting:
            if event == cv.EVENT_LBUTTONUP:
                delete_tracker(x, y, trackers)
                show = frame.copy()
                redraw()
            cursor_frame = show.copy()
            cv.line(cursor_frame, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), thickness=2)
            cv.line(cursor_frame, (x + 10, y - 10), (x - 10, y + 10), (0, 0, 255), thickness=2)
            cv.imshow("Video",cursor_frame)
    else:
        return



# Plays the video, with trackers. Called at the beginning of the program and also every time pause is set to false
def play_video():
    global frame, bbox, video, is_tracking, trackers, is_deleting

# construct the argument parse and parse the arguments
# ARGUMENTS
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
        help="path to input file")
"""
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
        help="threshold when applyong non-maxima suppression")
"""
ap.add_argument("-o", "--output", default="images",
	help="path to output video")
args = vars(ap.parse_args())

# MAIN
if __name__ == "__main__":

    # Initialize global variables. Additionally, create an empty dictionary and set is_tracking to False until an object is selected
    global frame, bbox, video, tracker, is_tracking, trackers, pause, show, is_deleting, has_ball, ball_tracker, imgno
    imgno = 1
    is_tracking = False
    is_deleting = False
    trackers = dict()
    pause = False

    # If a file is specified, open the file, otherwise take from the camera
    video = cv.VideoCapture(args["input"])
    writer = None
    (W, H) = (None, None)

    # Exit if video not opened
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    cv.setMouseCallback("Video", handler)


    while True:

        if pause == True:
            check_commands()
            continue

        # Read a new frame
        cap_ret, frame = video.read()
        if not cap_ret:
            break

        show = frame.copy()

        # Start timer
        timer = cv.getTickCount()


        #frame = detect_ball(args, frame, net, ln, LABELS)
        gray = cv.cvtColor(show, cv.COLOR_BGR2GRAY)
        
        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        

        # Draw bounding boxes around haar faces
        for (column, row, width, height) in detected_faces:
            cv.rectangle(show,(column, row),(column + width, row + height),(0, 255, 0),2)
        

        # Calculate Frames per second (FPS)
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)


        # Display tracker type on frame
        cv.putText(show, "GOTURN Tracker", (100,20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

        # Display FPS on show
        cv.putText(show, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display result
        redraw()
        check_commands()


# release the file pointers
video.release()
