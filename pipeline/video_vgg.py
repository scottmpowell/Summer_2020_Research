import numpy as np
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
import argparse

# my additional imports for modification of image script
import os
# suppress tensorflow log messages from mtcnn unless error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from PIL import Image

# keras testing
from keras.models import load_model
from keras.engine import Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input, decode_predictions

# my additional imports for modification to video script
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils
import cv2
import time
from collections import Counter

# onnx ultra-light modificaiton
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import detect_onnx

"""
Functions adapted from 'How to Develop a Face Recognition System
Using FaceNet in Keras' by Jason Brownlee

URL: https://machinelearningmastery.com/
how-to-develop-a-face-recognition-system-using-facenet-in-keras
-and-an-svm-classifier/

Additonal contributions:
LR Information: 'Extremely Small Face Recognition Project'
by Kevin Wu
    URL: https://kevincodeidea.wordpress.com/2020/01/14/
    an-extremely-small-facerecog-project-for-extreme-beginners
    -and-a-few-thoughts-on-future-part-ii-transfer-learning
    -and-keras/
ONNX Image Processing: 'Real-Time Face Recognition with CPU'
by Yirui Feng
    URL: https://towardsdatascience.com/real-time-face
    -recognition-with-cpu-983d35cc3ec5
OpenCV Manipulation: 'OpenCV Face Recognition' by Adrian
Rosebrock
    URL: https://www.pyimagesearch.com/2018/09/24/
    opencv-face-recognition/
VGGFace Model: 'Keras VGGFace' by Refik Can Malli
    URL: https://github.com/rcmalli/keras-vggface
"""

def extract_face(image, boxes, required_size=(224,224)):
    
    # using the bounding box returned by the detector to return
    # resized images of each detected face
    num_faces = len(boxes)
    final_faces = []
    final_bounds = []
    
    # decide whether face is large enough to recognize
    image_height, image_width = image.shape[:2]
    scale_height, scale_width = image_height//15, image_width//15
    
    if num_faces != 0:
        for i in range(num_faces):
            
            # extract a single bounding box and perform calculations
            box = boxes[i]
            x1, y1, x2, y2 = abs(int(box[0])), abs(int(box[1])), abs(int(box[2])), abs(int(box[3]))
            width, height = (x2 - x1), (y2 -y1)
            
            if (width > scale_width or height > scale_height) and (width < height*1.25):
                bound = [x1, y1, x2, y2]
                face = image[y1:y2, x1:x2]
                
                single = Image.fromarray(face)
                single = single.resize(required_size)
                face_array = np.asarray(single)
                        
                final_faces.append(face_array)
                final_bounds.append(bound)
    
    return final_faces, final_bounds

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--embeddings", required=True,
    help="name of embeddings file")
    ap.add_argument("-p", "--predictor", required=True,
    help="select SVC predictor or logistic regression")
    ap.add_argument("-v", "--video", required=True,
    help="path to video")
    ap.add_argument("-c", "--vector", required=True,
    help="dimension for embeddings vector")
    ap.add_argument("-s", "--scrub", required=True,
    help="navigate within the video by moving through frames")
    ap.add_argument("-w", "--write", required=True,
    help="write video file")
    args = vars(ap.parse_args())

    data = np.load(args["embeddings"], allow_pickle=True)
    trainX, trainy = data['arr_0'], data['arr_1']
    trainX_unmod, trainy_unmod = data['arr_0'], data['arr_1']
    
    # get directory
    current = os.getcwd()
    video_path = current + '\\' + args["video"]

    # load onnx ultra-light detector
    onnx_path = current + '/models/ultra_light_640.onnx'
    onnx_model = onnx.load(onnx_path)
    predictor = prepare(onnx_model)
    options = ort.SessionOptions() # suppress warning messages
    options.log_severity_level = 4
    ort_session = ort.InferenceSession(onnx_path, options)
    input_name = ort_session.get_inputs()[0].name
    print('[INFO] ONNX Detector loaded')
    
    # choose betweeen SVC and LR [Wu]
    if args["predictor"] == "SVC":
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)
        
        predictor = "SVC"
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainy)
    elif args["predictor"] == "LR":
        predictor = "LR"
        model = LR().fit(trainX_unmod, trainy_unmod)
    else:
        raise Exception('Predictor not selected')
    print('[INFO] Predictor set')
    
    # vggface keras model for testing [Malli]
    if args["vector"] == "2048":
        vector = 2048
        vgg_features = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    elif args["vector"] == "128":
        vector = 128
        vgg_test = VGGFace(model='senet50', include_top=False, input_shape=(224,224,3))
        last_layer = vgg_test.get_layer('activation_79').output
        custom_vgg = Model(vgg_test.input, last_layer)
    else:
        raise Exception('Embedding dimension not selected')
    print('[INFO] Model loaded')
    
    # initialize write/scrub variables
    write = False
    scrub = False
    if args["write"] == "True" and args["video"] != "cam":
        write = True
    if args["scrub"] == "True":
        scrub = True
    
    # read video from proper source and set scrubbing interval
    if args["video"] == "cam":
        vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        vs = cv2.VideoCapture(video_path)
        frame_guess = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        if scrub == True:
            print('[INFO] scrubbing initiated')
            interval = int(input('[INPUT] Enter scrubbing interval: '))
            frame_idxs = np.linspace(0, frame_guess, interval, endpoint=False, dtype=np.int)
    
    # buffer time
    time.sleep(2)
    
    # start fps counter [Rosebrock]
    fps = FPS().start()
    
    # dictionary to return list of identities
    final_name = {}
    
    # set up list of available names to iterate through
    name_list = []
    for name in trainy_unmod:
        if name not in name_list:
            name_list.append(name)
    
    # actual frame count
    frame_count = 0
    
    # initialize last press variable for playing in reverse
    last_press = ''
    
    # set up writing object
    if write == True:
        print("[INFO] writing initiated")
        frame_width = int(vs.get(3))
        frame_height = int(vs.get(4))
        video = args["video"]
        s = video.split('.')
        file_name = s[0] + 'scan.' + 'avi'
        out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc('M',
                    'J', 'P', 'G'), 10, (frame_width, frame_height), 1)
    
    while True:
        
        # scrub over frames by only reading certain indices
        if scrub == True:
            i = 0
            for frame_idx in range(frame_guess):
                if frame_idx == frame_idxs[i]:
                    ret, frame = vs.read()
                    i += 1
                    if i >= len(frame_idxs):
                        break
        
        # read all frames
        else:
            ret, frame = vs.read()
        
        # no frame to be read [Rosebrock]
        if not ret:
            cv2.waitKey(1000)
            print('[INFO] Video file read to completion')
            break
        
        # add to frame count and resize
        frame_count += 1
        h, w = frame.shape[:2]
        
        # preprocess frame [Feng]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        
        # pass preprocessed frame to onnx for face detection
        confidences, boxes = ort_session.run(None, {input_name: img})
        boxes = detect_onnx.predict(w, h, confidences, boxes, 0.7)
    
        # array of empty arrays is returned if no face detected    
        if len(boxes[0]) == 0:
            num_faces = 0
            # continue does not display the frame w/o face
            pass
        else:
            face_pixels, bound = extract_face(frame, boxes)
            num_faces = len(bound)
            
            # get embedding for each face detected in frame
            new_test = list()
            for face in face_pixels:
                embedding = np.expand_dims(face, axis=0)
                if vector == 2048:
                    embedding = vgg_features.predict(embedding)
                    new_test.append(embedding[0])
                else:
                    embedding = custom_vgg.predict(embedding)
                    new_test.append(embedding[0][0][0])
            new_test = np.asarray(new_test)
            
            # predict identity using SVM classifier or
            # with logistic regression model
            name_array = []
            prob_array = []
            for identity in new_test:
                middle = np.expand_dims(identity, axis=0)
                if predictor == "SVC":
                    yhat_class = model.predict(middle)
                    yhat_prob = model.predict_proba(middle)
                 
                    class_index = yhat_class[0]
                    class_probability = yhat_prob[0,class_index] * 100
                    predict_names = out_encoder.inverse_transform(yhat_class)
                    
                    name = predict_names[0]
                    name_array.append(name)
                else:
                    yhat_class = model.predict(middle)
                    yhat_prob = model.predict_proba(middle)
                    
                    class_index = yhat_class[0]
                    name_index = name_list.index(class_index)
                    class_probability = yhat_prob[0][name_index] * 100                
                    
                    name_array.append(class_index)
                    name = class_index
                
                prob_array.append(class_probability)
                
                # update frame identification for multiple faces
                if name not in final_name:
                    final_name[name] = 0
                final_name[name] += 1
            
            # scale the font size and line thickness
            if w > h:
                font_size = int(w/60)
                thickness = max(int((w) / 500), 2)
            else:
                font_size = int(h/60)
                thickness = max(int((h) / 500), 2)
                    
            # draw the bounding box and predicted name
            for i in range(num_faces):
                x1, y1, x2, y2 = bound[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
                cv2.rectangle(frame, (x1-1, y2 + 15), (x2+1, y2), (80,18,236), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"ID: {name_array[i]}" + " " + "{:.2f}%".format(prob_array[i])
                cv2.putText(frame, text, (x1 + 5, y2 + 9), font, 0.45, (255, 255, 255), 1)
        
        # only calculate frame position for scrubbing
        if args["video"] != "cam":
            current_frame = int(vs.get(cv2.CAP_PROP_POS_FRAMES))
        
            # current frame position
            frame_pos = 'Frame Position: ' + str(current_frame) + ' / ' + str(frame_guess)
            pos = (20, h-50)
            cv2.putText(frame, frame_pos, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,255), 1, cv2.LINE_AA)    
           
        # update fps and show frame
        fps.update()
        cv2.imshow('video', frame)
        
        # resize to write, otherwise file will not be playable
        if write == True:
            frame = cv2.resize(frame, (frame_width, frame_height))
            out.write(frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # reset the last press
        if key == ord('\r'):
            last_press = ''
            
        # pause the video
        if key == ord('p'):
            cv2.waitKey()
        
        # operations for local video file only
        if args["video"] != "cam":
            
            # go back one frame
            if key == ord('b'):
                last_press = 'b'
                if scrub == True:
                    vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame-2*interval)
                else:
                    vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 2)
                cv2.waitKey()
            
            # play the video in reverse
            if key == ord('v') or last_press == 'v':
                last_press = 'v'
                if scrub == True:
                    vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame-2*interval)
                else:
                    vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 2)
            
            # set the frame or skip by a certain number of frames
            if key == ord('e'):
                last_press = 'e'
                move_type = str(input('[INPUT] Enter movement type: '))
                number = int(input('[INPUT] Enter frames to jump/skip: '))
                if scrub == True:
                    subtract = interval
                else:
                    subtract = 1
                if move_type == 'j':  
                    vs.set(cv2.CAP_PROP_POS_FRAMES, number-subtract)
                if move_type == 's':
                    direction = str(input('[INPUT] Enter direction to skip: '))
                    if direction == 'f':
                        vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame+number-subtract)
                    if direction == 'b':
                        vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame-number-subtract)
                
        # close the video
        if key == ord('q'):
            break
    
    # return FPS info [Rosebrock]
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    # return number of frames viewed
    print("[INFO] total frames: " + str(frame_count))
    
    # organize dictionary and return names and associated frames
    c = Counter(final_name)
    most_likely = c.most_common()
    for i in most_likely:
        name = i[0]
        frames = i[1]
        print("[INFO] likely identity in video: " + name + " " + str(frames))
    
    # perform cleanup
    vs.release()
    cv2.destroyAllWindows()