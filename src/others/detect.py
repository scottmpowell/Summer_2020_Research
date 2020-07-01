from PIL import Image
from detector import detect_faces
from visualization_utils import show_results
import sys

img = Image.open(sys.argv[1]) # modify the image path to yours
bounding_boxes, landmarks = detect_faces(img) # detect bboxes and landmarks for all faces in the image
image = show_results(img, bounding_boxes, landmarks) # visualize the results
image.show()
