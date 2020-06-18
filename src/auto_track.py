from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2 as cv
import numpy as np
import sys



face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
cv.namedWindow("Tracking")
#net = cv.dnn.readNetFromCaffe('goturn.prototext', 'goturn.caffemodel')


class trackee:
    # Inititializer
    def __init__(self, bbox, number):
        self.tracker = cv.TrackerMOSSE_create()
        self.bbox = bbox
        self.number = number
        self.tracker.init(frame, self.bbox)

    def __str__(self):
        data = "Tracking Object: " + str(self.number) + " Bbox:" + str(self.bbox)
        return data
"""
Deletes the first trackee in trackers that has a bbox encapsulating given coordinates
"""
def delete_tracker(x, y):
    global trackers, frame, show
    print(x,y)
    
    for key in trackers:
        if trackers[key].bbox[0] <= x <= trackers[key].bbox[0] + trackers[key].bbox[2]:
            if trackers[key].bbox[1] <= y <= trackers[key].bbox[1] + trackers[key].bbox[3]:
                trackers.pop(key)
                show = frame.copy()
                redraw()
                break
    return

def check_commands():
        global pause, is_deleting, frame, show
        cv.setMouseCallback("Tracking", handler)
        k = cv.waitKey(5) & 0xff
        if k == ord('q'):
            sys.exit()
        elif k == ord('p'):
            pause = not pause
        elif k == ord('s'):
            begin_track()
        elif k == ord('x'):
            if is_deleting == True:
                show = frame.copy()
                redraw()

            is_deleting = not is_deleting

"""
Begin tracking object. Video will pause on current frame and allow selection of a ROI to be tracked
"""
def begin_track():
    global frame, bbox, video, is_tracking, trackers, tracker_counter
    tracker_counter = 1
    while True: 
        bbox = cv.selectROI("Tracking", frame, False)
        if bbox == (0,0,0,0):
            break

        trackers[tracker_counter] = trackee(bbox, tracker_counter)
        tracker_counter += 1

        is_tracking = True


"""
redraw takes no arguments, but updates the frame and tracked objects
"""
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
    global is_deleting, show, pause
    if pause:
        if is_deleting:
            if event == cv.EVENT_LBUTTONUP:
                  delete_tracker(x, y)
            cursor_frame = show.copy()
            cv.line(cursor_frame, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), thickness=2)
            cv.line(cursor_frame, (x + 10, y - 10), (x - 10, y + 10), (0, 0, 255), thickness=2)
            cv.imshow("Tracking",cursor_frame)
         
    else:
        return
        


"""
Plays the video, with trackers. Called at the beginning of the program and also every time pause is set to false
"""
def play_video():
    global frame, bbox, video, is_tracking, trackers, is_deleting

if __name__ == "__main__":

    """
    Initialize global variables. Additionally, create an empty dictionary and set is_tracking to False until an object is selected
    """
    global frame, bbox, video, tracker, is_tracking, trackers, pause, show, is_deleting
    is_tracking = False
    is_deleting = False
    trackers = dict()
    pause = False

    """
    If a file is specified, open the file, otherwise take from the camera
    """
    if len(sys.argv) == 2:
        video = cv.VideoCapture(sys.argv[1])
    else:
        video = cv.VideoCapture(0)

    # Exit if video not opened
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    video.set(cv.CAP_PROP_FPS, 100)

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

        gray = cv.cvtColor(show, cv.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for bbox in detected_faces:
            trackers[tracker_counter] = trackee(bbox, tracker_counter)
            tracker_counter += 1

        # Update tracked objects
        redraw()

        # Calculate Frames per second (FPS)
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer);


        # Display tracker type on frame
        cv.putText(show, "GOTURN Tracker", (100,20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

        # Display FPS on show
        cv.putText(show, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        cv.imshow("Tracking", show)
        check_commands()
