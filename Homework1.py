# Filename: Homework1.py
# Author: Jason Grant, Middlebury College
# Date Created: 2/13/2020
# Date Modified: 3/6/2020
# Homework 1: Getting Started with Images

import numpy as np
import cv2
import sys

def bgr2gray(src):
    """
    Converts a color image into a grayscale image by computing a weighted
    average of the red, blue, and green channels
    """

    rows = src.shape[0]
    cols = src.shape[1]
    dst = np.zeros((rows,cols), np.uint8)

    dst = np.floor(0.114 * src[:,:,0] + 0.587* src[:,:,1] + 0.299 * src[:,:,2])

    return dst

def threshold(src, threshold):
    """
    Converts a grayscale image to a binary image
    """

    rows = src.shape[0]
    cols = src.shape[1]
    dst = np.zeros((rows,cols), np.uint8)

    temp = src >= threshold
    dst[temp] = 255

    return dst

def padding(src, bordersize):
    """
    Uses zero padding to add a black border to the images
    """

    rows = src.shape[0]
    cols = src.shape[1]

    dst = np.zeros((rows + 2*bordersize, cols + 2*bordersize), np.uint8)
    dst[bordersize:bordersize + rows, bordersize:bordersize + cols] = src

    return dst

def convolution(src, kernel):
    """
    Convoles an input image with the given kernel.
    """

    border = kernel.shape[0]//2;

    rows = src.shape[0]
    cols = src.shape[1]

    krows = kernel.shape[0]
    kcols = kernel.shape[1]
    kweight = np.sum(kernel)

    dst = np.zeros(src.shape, np.float32)
    padded = padding(src,border);

    for row in range(rows):
        for col in range(cols):

            dst[row,col] = (np.sum(np.multiply(kernel, \
                            padded[row:row+krows,col:col+kcols])))

    if abs(kweight) > 1:
        dst = dst / kweight
    return dst

if __name__ == "__main__":

    if len(sys.argv) < 3:
        sys.exit("Invalid usage: Homework1.py [filename] [threshold] [pad width]")

    filename = sys.argv[1]


    # Read in the image in color
    src = cv2.imread(filename);
    cv2.imshow("Source", src)
    cv2.waitKey(30)

    # Part 1: Convert the image to grayscale
    gray = bgr2gray(src);
    cv2.imshow("Grayscale", gray)
    #cv2.imwrite("gray.png",gray)
    cv2.waitKey(30)

    # Part 2: Using the grayscale image, create
    # a binary image from a user-specified threshold
    thresh = int(sys.argv[2])
    bw = threshold(gray, int(thresh));
    cv2.imshow("Threshold", bw)
    #cv2.imwrite("bw.png",bw)
    cv2.waitKey(30)

    #Part 3: Zero Padding image
    pad = int(sys.argv[3])
    padded = padding(gray,int(pad));
    cv2.imshow("Padded Image", padded)
    #cv2.imwrite("padded.png",padded)
    cv2.waitKey(30)

    # Part 4: Blur the image
    kernel = np.ones((7,7), np.uint8)
    blur = convolution(gray, kernel)
    blur[blur > 255] = 255
    blur[blur < 0] = 0

    cv2.imshow("Blurred Image",blur.astype(np.uint8))
    #cv2.imwrite("blur.png",blur)
    cv2.waitKey(30)

    # Part 5: Sharpen the image
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = convolution(gray, kernel)
    sharp[sharp > 255] = 255
    sharp[sharp < 0] = 0

    cv2.imshow("Sharpened Image",sharp.astype(np.uint8))
    #cv2.imwrite("sharp.png",sharp)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
