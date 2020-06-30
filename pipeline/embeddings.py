import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.models import load_model
import argparse
from keras.engine import Model

"""
Adapted from 'How to Develop a Face Recognition System Using
FaceNet in Keras' by Jason Brownlee

URL: https://machinelearningmastery.com/
how-to-develop-a-face-recognition-system-using-facenet
-in-keras-and-an-svm-classifier/
"""

def get_embedding(model, face_pixels):
    
    # if no face was detected in the cropped image
    if face_pixels is None:
        return
    face_pixels = face_pixels.astype(np.float32)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    samples = np.expand_dims(face_pixels, axis=0)

    # embedding is generated
    yhat = model.predict(samples)
    return yhat[0]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True,
    help="name of data file")
    ap.add_argument("-o", "--output", required=True,
    help="name of file to save")
    args = vars(ap.parse_args())
    
    data = np.load(args["data"], allow_pickle=True)
    trainx, trainy = data['arr_0'], data['arr_1']
    #print('Loaded: ', trainx.shape, trainy.shape)

    # load model to avoid loading for each embedding generation
    model = load_model('models/facenet_keras.h5')
    print('[INFO] loaded model')

    new_trainx = list()
    for face_pixels in trainx:
        embedding = get_embedding(model, face_pixels)
        new_trainx.append(embedding)
    new_trainx = np.asarray(new_trainx)
    
    name = args["output"].split('/')
    file_name = name[0] + '/' + 'embeddings' + name[1] + '.npz'
    np.savez_compressed(file_name, new_trainx, trainy)
    print('[INFO] file written successfully')