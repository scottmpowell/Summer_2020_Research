# Filename: Video.py
# Author: Jason Grant, Middlebury College
# Date Created: 3/12/2020
# Open a video source with OpenCV

import cv2
import argparse
import numpy as np
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Frequently used
    parser.add_argument('-s', '--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('-o', '--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('-n', '--number', type=int, default=1, help='what number to start images at')

    opt = parser.parse_args()

    # The default capture device is the default video source.
    source = 0
    if len(sys.argv) > 1:
        source = opt.source

        # If the source is a peripheral device, it will
        # be a number. Change the string into a number.
        # If the source beingg passed is a filename, this
        # will throw an exception. Ignore the conversion
        try:
            source = int(source)
        except:
            pass

    cap = cv2.VideoCapture(source)
    imgno = opt.number

    while(True):
        ret, frame = cap.read()
        cv2.imshow("frame",frame)

        # If the q key is pressed, the loop will exit
        k = cv2.waitKey(20)
        if k & 0xFF == ord('q'):
                break
        elif k & 0xFF == ord('w'):
            text = opt.output + "/img" + str(imgno) + ".png"
            cv2.imwrite(text, frame)
            print("file saved to", text)
            imgno += 1

    cv2.destroyAllWindows()
