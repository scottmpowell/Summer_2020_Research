"""
Basketball and Player Tracker, using YOLOv3
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
from yolo_detect import detect_ball
from tracker_utils import trackee, delete_tracker, begin_track

# Modules from other examples
#from visualization_utils import show_results
#from detector import detect_faces

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_fullbody.xml")
cv.namedWindow("Tracking")


def check_commands():
        global pause, is_deleting, frame, show, trackers, is_tracking, yoloing
        cv.setMouseCallback("Tracking", handler)
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
            cv.imwrite("slice.png", frame)
        elif k == ord('r'):
            yoloing = not yoloing
        elif k == ord('x'):
            if pause:
                if is_deleting == True:
                    show = frame.copy()
                    redraw()

                is_deleting = not is_deleting

"""
# Begin tracking object. Video will pause on current frame and allow selection of a ROI to be tracked
def begin_track():
    global frame, bbox, video, is_tracking, trackers, tracker_counter
    tracker_counter = 1
    while True: 
        bbox = cv.selectROI("Tracking", frame, False)
        if bbox == (0,0,0,0):
            break

        trackers[tracker_counter] = trackee(bbox, tracker_counter, frame)
        tracker_counter += 1

        is_tracking = True
"""


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
    cv.imshow("Tracking", show)
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
            cv.imshow("Tracking",cursor_frame)
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
ap.add_argument("-y", "--yolo", default="/home/scott/summer2020/Summer_2020_Research/src/yolo-coco",
        help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
        help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())
ap.add_argument("-o", "--output", required=True,
	help="path to output video")


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# MAIN
if __name__ == "__main__":

    # Initialize global variables. Additionally, create an empty dictionary and set is_tracking to False until an object is selected
    global frame, bbox, video, tracker, is_tracking, trackers, pause, show, is_deleting, has_ball, ball_tracker, yoloing
    is_tracking = False
    is_deleting = False
    yoloing = False
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

    cv.setMouseCallback("Tracking", handler)


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

        if yoloing:
            frame = detect_ball(args, frame, net, ln, LABELS)
        #gray = cv.cvtColor(show, cv.COLOR_BGR2GRAY)
        
#        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)


        # Update tracked objects
        show = frame.copy()
        redraw()
        #for (column, row, width, height) in detected_faces:
            #cv.rectangle(show,(column, row),(column + width, row + height),(0, 255, 0),2)

        # Calculate Frames per second (FPS)
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

        # Display FPS on show
        cv.putText(show, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display result
        cv.imshow("Tracking", show)
        check_commands()


# release the file pointers
video.release()
