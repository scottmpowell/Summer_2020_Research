from PIL import ImageDraw

import cv2 as cv

def show_results(img, bounding_boxes, facial_landmarks = []):
    """Draw bounding boxes and facial landmarks.
    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].
    Returns:
        an instance of PIL.Image.
    """
    img_copy = img.copy()
    img_cv = cv.imread("slice.png")
#    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        cv.rectangle(img_cv, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255,0,0), 2, 1)
        """
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline = 'white')
        """

    inx = 0
    for p in facial_landmarks:
        for i in range(5):
            break
            draw.ellipse([
                (p[i] - 1.0, p[i + 5] - 1.0),
                (p[i] + 1.0, p[i + 5] + 1.0)
            ], outline = 'blue')

#    cv.imshow("window", img_cv)
    return img_copy
