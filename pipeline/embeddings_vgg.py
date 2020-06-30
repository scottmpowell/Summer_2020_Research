import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import argparse

from keras.engine import Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input, decode_predictions

"""
Adapted from 'How to Develop a Face Recognition System Using
FaceNet in Keras' by Jason Brownlee

URL: https://machinelearningmastery.com/
how-to-develop-a-face-recognition-system-using-facenet
-in-keras-and-an-svm-classifier/

Keras model manipulation adapted from 'Keras VGGFace' by
Refik Can Malli

URL: https://github.com/rcmalli/keras-vggface
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
    ap.add_argument("-v", "--vector", required=True,
    help="dimension for embeddings vector")
    args = vars(ap.parse_args())
    
    data = np.load(args["data"], allow_pickle=True)
    trainx, trainy = data['arr_0'], data['arr_1']
    #print('Loaded: ', trainx.shape, trainy.shape)

    # load vggface keras model, switch between different sized
    # vectors to experiment with accuracy/speed tradeoffs, this
    # is done by choosing a specific layer from within the model
    if args["vector"] == "2048":
        print('[INFO] using 2048-d embeddings')
        vector = 2048
        vgg_features = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    elif args["vector"] == "128":
        print('[INFO] using 128-d embeddings')
        vector = 128
        vgg_test = VGGFace(model='senet50', include_top=False, input_shape=(224,224,3))
        last_layer = vgg_test.get_layer('activation_79').output
        custom_vgg = Model(vgg_test.input, last_layer)
    else:
        raise Exception('Embedding dimension not selected')
    print('[INFO] loaded model')

    # choose 2048d vector embeddings or 128d
    new_trainx = list()
    for face_pixels in trainx:
        embedding = np.expand_dims(face_pixels, axis=0)
        if vector == 2048:
            embedding = vgg_features.predict(embedding)
            new_trainx.append(embedding[0])
        else:
            embedding = custom_vgg.predict(embedding)
            new_trainx.append(embedding[0][0][0])
        
    new_trainx = np.asarray(new_trainx)
    
    name = args["output"].split('/')
    file_name = name[0] + '/' + 'embeddings' + name[1] + '.npz'
    np.savez_compressed(file_name, new_trainx, trainy)
    print('[INFO] file written successfully')