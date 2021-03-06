import cv2 as cv
import numpy as np

class Player:
    # Inititializer
    def __init__(self, xyxy, number, frame):
        self.tracker = cv.TrackerCSRT_create()
        self.bbox = bbox

        # coordinates of last seen position
        self.xyxy = None
        # Final number, when Player is drawn, will use self.number
        self.number = number
        self.ctr = None

        # Right and left are the individual digits
        self.number_r = None
        self.conf_r = None
        self.number_l = None
        self.conf_l = None
        self.digits = 0 # 0 if Null

        # Doesn't use tracker
        #self.tracker.init(frame, self.bbox)

    def __str__(self):
        data = "Tracking Object: " + str(self.number) + " Bbox:" + str(self.bbox)
        return data

    # Returns True if coordinates are within the tracker's bounding box, and False otherwise
    def matches(self, x, y):
        return (self.bbox[0] <= x <= self.bbox[0] + self.bbox[2] and self.bbox[1] <= y <= self.bbox[1] + self.bbox[3])

    def set_ctr(self, p1, p2):
        """
        Sets self.ctr to be the average of the two points
        """
        self.ctr = (int((p1[0] + p2[0]) // 2), int((p1[1] + p2[1]) // 2))


    def check_number(self, xyxy, value, conf):
        """
        Checks player for detected number, if player has no number, they become this number, and the number is estimated to be left or right number.
        If there already is a number, if this number is a different number and on opposite side, change.
        If there already is a number and this number is the same, do nothing
        If on the same side but the confidence is >= 20%, change the number
        """
        n1, n2 = xyxy2pts(xyxy)
        center = find_center(n1, n2)
        on_left = center[0] < self.ctr
        if self.number is None:
            if center[0] >= self.ctr:
                self.number_right = value
        elif self.digits == 1:
            # Only one number, if on opposite side and different value, combine
            if self.number_r and value != self.number:
                self.number_l = value
                self.number = int(str(number_l) + str(number_r))
            elif self.number_l and value != self.number:
                self.number_r = value
                self.number = int(str(number_l) + str(number_r))
            print(self.number)

            
            
            



class Ball:

    def __init__(self, bbox=None, ctr=None, has_tracker=False, has_ball=False):
        self.tracker = cv.TrackerCSRT_create()
        self.bbox = bbox # bbox is used solely for trackers. It is a fourple saved as (p1.x, p1.y, w, h)
        self.ctr = ctr # ctr is the centerpoint of the ball. Used for distance calculations
        self.has_tracker = has_tracker # Set to False if the ball tracker fails to find the ball from the tracker
        self.has_ball = has_ball # Set to False if has_tracker is false and no ball is detected that frame
        self.last_bbox = (0, 0, 0, 0)
        self.age = 0


    def contained_in(self, p1, p2):
        """
        Returns True if ball_ctr is between two 2D points
        """
        if self.ctr is None:
            # Ball is unknown
            return False
        return ((p1[0] <= int(self.ctr[0]) <= p2[0]) and (p1[1] <= int(self.ctr[1]) <= p2[1]))

    def pts2box(self, p1, p2):
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

    def draw_box(self, frame):
        """
        Return the input frame with ball's bounding box.
        """
        if not self.ctr:
            return frame
        if self.has_ball:
            if self.bbox == (0,0,0,0):
                return frame
            cv.rectangle(frame, (int(self.bbox[0]), int(self.bbox[1])), (int(self.bbox[0]) + int(self.bbox[2]), int(self.bbox[1]) + int(self.bbox[3])), (255,0,255), 4, 1)
            cv.putText(frame, "Ball", ((int(self.bbox[0]), int(self.bbox[1] - 2))), 0, 1, [225, 255, 255], thickness=2, lineType=cv.LINE_AA) 
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
            ret, bbox = self.tracker.update(empty_frame)
            if ret:
                # Tracking success
                self.last_bbox = self.bbox
                self.bbox = bbox
                self.box2ctr()
                self.age += 1
                if self.age >= 7:
                    self.last_bbox = self.bbox
                    self.bbox = (0, 0, 0, 0)
                    self.has_tracker = False
                    
            else :
                self.last_bbox = self.bbox
                self.bbox = bbox # zero
                # Tracking failure
                self.has_tracker = False

    def distance(self, p1):
        if self.ctr is None:
            return 
        return np.sqrt(abs(self.ctr[0] - p1[0])**2 + abs(self.ctr[1] - p1[1])**2)


    def update(self, xyxy, empty_frame):
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        self.last_bbox = self.bbox
        self.pts2box(c1, c2)
        self.set_ctr(c1, c2)
        self.tracker = cv.TrackerKCF_create()
        self.tracker.init(empty_frame, self.bbox)
        self.has_tracker = True
        self.has_ball = True
        self.age = 0


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
def begin_track(empty_frame, trackers):
    tracker_counter = 1
    while True: 
        bbox = cv.selectROI("Video", empty_frame, False)
        if bbox == (0,0,0,0):
            return tracker_counter - 1

        trackers.update({tracker_counter:Player(bbox, tracker_counter, empty_frame)})
        tracker_counter += 1

def find_next_ball(frames):
    for i in range(len(frames)):
        if frames[i][1] > 0:
            return i
    return None

def check_intersect(ball_xyxy, p1, p2):
    """
    Returns true if ball_coords is between two points
    """
    b1, b2 = xyxy2pts(ball_xyxy)
    bctr = find_center(b1, b2)
    return((p1[0] <= bctr[0] <= p2[0]) and (p1[1] <= bctr[1] <= p2[1]))

def xyxy2pts(xyxy):
    
    p1 = (int(xyxy[0]), int(xyxy[1]))
    p2 = (int(xyxy[2]), int(xyxy[3]))
    return p1, p2

def pts2box(p1, p2):
    """
    Returns bounding box given two points. Bounding box format is (p1.x, p1.y, width, height)
    """
    return (int(p1[0]), int(p1[1]), int(p2[0] - p1[0]), int(p2[1] - p1[1]))
    
