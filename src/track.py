"""
Basketball and Player Tracker, using YOLOv3
Authors: Scott Powell, Christian Newton
"""
# import packages
from imutils.video import VideoStream
from imutils.video import FPS
from visualization_utils import show_results
from PIL import Image
from detector import detect_faces
import argparse
import imutils
import time
import bcolz 
import cv2 as cv
import numpy as np
import sys
import torch
import os

# Personal Modules
from yolo_detect import detect_ball
from tracker_utils import trackee, delete_tracker


face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
cv.namedWindow("Tracking")
#net = cv.dnn.readNetFromCaffe('goturn.prototext', 'goturn.caffemodel')


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
        elif k == ord('w'):
            cv.imwrite("slice.png", frame)
        elif k == ord('x'):
            if pause:
                if is_deleting == True:
                    show = frame.copy()
                    redraw()

                is_deleting = not is_deleting

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
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# MAIN
if __name__ == "__main__":

    # Initialize global variables. Additionally, create an empty dictionary and set is_tracking to False until an object is selected
    global frame, bbox, video, tracker, is_tracking, trackers, pause, show, is_deleting, has_ball, ball_tracker
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

    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    #video.set(cv.CAP_PROP_FPS, 10)

    cv.setMouseCallback("Tracking", handler)


    while True:

        if pause == True:
            check_commands()
            continue

        # Read a new frame
        cap_ret, frame = video.read()
        if not cap_ret:
            break

        """
        # Translates a BGR image to RGB for PIL functions
        img_RGB = cv.cvtColor(show, cv.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_RGB)

        bounding_boxes, landmarks = detect_faces(im_pil)
        image = show_results(im_pil, bounding_boxes, landmarks)
        im_pil = np.asarray(image)
        im_faces_bgr = cv.cvtColor(show, cv.COLOR_RGB2BGR)
        """
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # if the frame dimension are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # Start timer
        timer = cv.getTickCount()

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        # draw a bounding box rectangle and label on the frame
                        #color = [int(c) for c in COLORS[classIDs[i]]]
                        cv.rectangle(frame, (x, y), (x + w, y + h), (255,0,255), 2)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        cv.putText(frame, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
                fourcc = cv.VideoWriter_fourcc(*"MJPG")
                #writer = cv.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

                # some information on processing single frame
                if total > 0:
                    elap = (end - start)
                    print("[INFO] single frame took {:.4f} seconds".format(elap))
                    print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))


        # write the output frame to disk
        #writer.write(frame)



        """
        gray = cv.cvtColor(show, cv.COLOR_BGR2GRAY)
        
        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (column, row, width, height) in detected_faces:
            cv.rectangle(show,(column, row),(column + width, row + height),(0, 255, 0),2)
        """
        show = frame.copy()
        # Update tracked objects
        redraw()

        # Calculate Frames per second (FPS)
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)


        # Display tracker type on frame
        cv.putText(show, "GOTURN Tracker", (100,20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

        # Display FPS on show
        cv.putText(show, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display result
        cv.imshow("Tracking", show)
        check_commands()


# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
