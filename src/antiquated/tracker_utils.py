import cv2 as cv

class trackee:
    # Inititializer
    def __init__(self, bbox, number, frame):
        self.tracker = cv.TrackerMOSSE_create()
        self.bbox = bbox
        self.number = number
        self.tracker.init(frame, self.bbox)

    def __str__(self):
        data = "Tracking Object: " + str(self.number) + " Bbox:" + str(self.bbox)
        return data

    # Returns True if coordinates are within the tracker's bounding box, and False otherwise
    def matches(self, x, y):
        return (self.bbox[0] <= x <= self.bbox[0] + self.bbox[2] and self.bbox[1] <= y <= self.bbox[1] + self.bbox[3])


def delete_tracker(x, y, trackers):
    """
    Delete the first tracker that surround the x and y coordinates
    """
    for key in trackers:
        if trackers[key].matches(x, y):
            trackers.pop(key)
            break
    return



# Begin tracking object. Video will pause on current frame and allow selection of a ROI to be tracked
def begin_track(frame, trackers):
    tracker_counter = 1
    while True: 
        bbox = cv.selectROI("Tracking", frame, False)
        if bbox == (0,0,0,0):
            return tracker_counter - 1

        trackers.update({tracker_counter:trackee(bbox, tracker_counter, frame)})
        tracker_counter += 1

    return tracker_counter
