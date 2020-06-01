# Filename: Video.py
# Author: Jason Grant, Middlebury College
# Date Created: 3/12/2020
# Open a video source with OpenCV

import cv2
import numpy as np
import sys

if __name__ == "__main__":

    # The default capture device is the default video source.
    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

        # If the source is a peripheral device, it will
        # be a number. Change the string into a number.
        # If the source beingg passed is a filename, this
        # will throw an exception. Ignore the conversion
        try:
            source = int(source)
        except:
            pass

    cap = cv2.VideoCapture(source)

    while(True):
        ret, frame = cap.read()
        cv2.imshow("frame",frame)

        # If the q key is pressed, the loop will exit
        if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
