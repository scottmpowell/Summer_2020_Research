import cv2 as cv
import numpy as np

class trackee:
    # Inititializer
    def __init__(self, bbox, number, frame):
        self.tracker = cv.TrackerCSRT_create()
        self.bbox = bbox
        self.number = number
        self.tracker.init(frame, self.bbox)

    def __str__(self):
        data = "Tracking Object: " + str(self.number) + " Bbox:" + str(self.bbox)
        return data

    # Returns True if coordinates are within the tracker's bounding box, and False otherwise
    def matches(self, x, y):
        return (self.bbox[0] <= x <= self.bbox[0] + self.bbox[2] and self.bbox[1] <= y <= self.bbox[1] + self.bbox[3])

class Ball:

    def __init__(self, bbox=None, ctr=None, has_tracker=False, has_ball=False):
        self.tracker = cv.TrackerKCF_create()
        self.bbox = bbox # bbox is used solely for trackers. It is a fourple saved as (p1.x, p1.y, w, h)
        self.ctr = ctr # ctr is the centerpoint of the ball. Used for distance calculations
        self.has_tracker = has_tracker # Set to False if the ball tracker fails to find the ball from the tracker
        self.has_ball = has_ball # Set to False if has_tracker is false and no ball is detected that frame
        self.conf = None # Confidence of the ball. 


    def contained_in(self, p1, p2):
        """
        Returns True if ball_ctr is between two 2D points
        """
        if self.ctr == None:
            # Ball is unknown
            return False
        return ((p1[0] <= self.ctr[0] <= p2[0]) and (p1[1] <= self.ctr[1] <= p2[1]))

    def set_box(self, p1, p2):
        """
        Changes the bounding box of the ball. Bounding box format is (p1.x, p1.y, width, height)
        """
        self.bbox = (int(p1[0]), int(p1[1]), int(p2[0] - p1[0]), int(p2[1] - p1[1]))
        return self.bbox

    def set_ctr(self, p1, p2):
        """
        Sets self.ctr to be the average of the two points
        """
        self.ctr = (int((p1[0] + p2[0]) // 2), int((p1[1] + p2[1]) // 2))

    def box2ctr(self):
        """
        Set center to be the center point of the rectangle saved in bbox
        """
        self.ctr = (self.bbox[0] + (self.bbox[2] // 2), self.bbox[1] + (self.bbox[3] //2))


    def draw_ctr(self, frame):
        """
        Return the input frame with ball's bounding box.
        """
        if not self.ctr:
            return frame
        if self.has_ball:
            if self.bbox == (0,0,0,0):
                return frame
            cv.rectangle(frame, (int(self.bbox[0]), int(self.bbox[1])), (int(self.bbox[0]) + int(self.bbox[2]), int(self.bbox[1]) + int(self.bbox[3])), (255,0,255), 4, 1)
            #cv.rectangle(frame, (int(self.bbox[0]), int(self.bbox[1])), (100,100), (0,255,0), 2, 1)
            return frame
            
            #cv.rectangle(frame, self.ctr, self.ctr, (0,255,0), 2, 1)
            #cv.rectangle(frame, (self.ctr[0] - 1, self.ctr[1] - 1), (self.ctr[0] + 1, self.ctr[1] + 1), (0,255,0), 2, 1)
        else:
            return frame

    def check_tracker(self, empty_frame):
        """
        Check ball tracker location, and set ball parameters accordingly.
        """
        if not self.has_tracker:
            return
        else:
            # There is a tracker, see if it can track
            # If it can't, self.bbox is set to 0
            ret, self.bbox = self.tracker.update(empty_frame)
            print(ret, self.bbox)
            if ret:
                print(self.bbox)
                # Tracking success
                self.box2ctr()
            else :
                # Tracking failure
                self.has_tracker = False

    def distance(self, p1):
        return np.sqrt(abs(self.ctr[0] - p1[0])**2 + abs(self.ctr[1] - p1[1])**2)


    def update(self, xyxy, conf, empty_frame):
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        self.set_box(c1, c2)
        self.prev_box = self.ctr
        self.set_ctr(c1, c2)
        self.tracker = cv.TrackerKCF_create()
        self.tracker.init(empty_frame, self.bbox)
        self.has_tracker = True
        self.has_ball = True
        self.conf = conf


def delete_tracker(x, y, trackers):
    """
    Delete the first tracker that surround the x and y coordinates
    """
    for key in trackers:
        if trackers[key].matches(x, y):
            trackers.pop(key)
            break
    return

def find_center(p1, p2):
    return (int((p1[0] + p2[0]) // 2), int((p1[1] + p2[1]) // 2))

# Begin tracking object. Video will pause on current frame and allow selection of a ROI to be trackeddef begin_track(empty_frame, frame, trackers):
def begin_track(empty_frame, frame, trackers):
    while True: 
        bbox = cv.selectROI("Video", frame, False)
        if bbox == (0,0,0,0):
            return tracker_counter - 1

        trackers.update({tracker_counter:trackee(bbox, tracker_counter, empty_frame)})
        tracker_counter += 1

    return tracker_counter

def find_next(frames):
    for i in range(len(frames)):
        if frames[i][2] > 0:
            return i
