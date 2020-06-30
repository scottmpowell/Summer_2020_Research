import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import numpy as np
from PIL import Image
from os import listdir
from matplotlib import pyplot
import argparse

"""
Adapted from 'How to Develop a Face Recognition System Using
FaceNet in Keras' by Jason Brownlee

URL: https://machinelearningmastery.com/
how-to-develop-a-face-recognition-system-using-facenet-in-keras
-and-an-svm-classifier/
"""

def extract_face(image, detector, model):
    
    # open the image and ensure that it is in RGB format for MTCNN
    image = Image.open(image)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    
    # obtain the faces detected with MTCNN
    results = detector.detect_faces(pixels)
    if results == []:
        return
    
    # only extracting the first face, good for embedding creation
    x1, y1, width, height = results[0]['box']
    x1, y1, = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    
    image = Image.fromarray(face)
    
    # change to (160, 160) for original keras facenet model,
    # (224, 224) for keras vggface model
    if model == "facenet":
        image = image.resize((160, 160))
    elif model == "vgg":
        image = image.resize((224, 224))
    else:
        raise Exception("Recognition model not selected")
    face_array = np.asarray(image)
    
    return face_array

def load_faces(directory, detector, model):
    faces = list()
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path, detector, model)
        faces.append(face)
    return faces

def load_dataset(directory, model):
    x, y = list(), list()
    current = os.getcwd()
    directory = current + '\\' + directory
    
    # initialize the detector here to avoid loading for each image
    detector = MTCNN()
    for subdir in listdir(directory):
        path = directory + '\\' + subdir + '\\'
        if not os.path.isdir(path):
            continue
        faces = load_faces(path, detector, model)
        labels = [subdir for i in range(len(faces))]
        print('[INFO] loaded %d examples for class: %s' % (len(faces), subdir))
        x.extend(faces)
        y.extend(labels)
    return np.asarray(x), np.asarray(y)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train", required=True,
    help="path to input directory of faces")
    ap.add_argument("-o", "--output", required=True,
    help="name of file to save")
    ap.add_argument("-m", "--model", required=True,
    help="face recognition model")
    args = vars(ap.parse_args())
    
    train = '\\' + args["train"]
    model = args["model"]
    trainx, trainy = load_dataset(train, model)
    #print(testx.shape, testy.shape)
    
    name = args["output"].split('/')
    file_name = name[0] + '/' + 'data' + name[1] + '.npz'
    np.savez_compressed(file_name, trainx, trainy)
    print('[INFO] file written successfully')