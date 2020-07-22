"""
Basketball and Player Tracker
Authors: Scott Powell, Christian Newton
"""
# import packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2 as cv
import numpy as np
import sys
import torch
import os

# Personal Modules
from utils.tracker_utils import *

# Yolov5 modules
from utils.datasets import *
import torch.backends.cudnn as cudnn
from utils.utils import *

# Haar cascade classification is fast, but is nowhere near as accurate as yolo or anything recent
# If yolo cannot be used to track the ball effectively, I'm considering constructing a Haar cascade classifier for purely the ball, to use in conjunction with yolo

#face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_fullbody.xml")

#net = cv.dnn.readNet("goturn.caffemodel", "goturn.prototxt")

def pause_video():
    global pause 
    cv.setMouseCallback("Video", handler)
    while pause:
        k = cv.waitKey(5) & 0xff
        check_commands(k)

def check_commands(k):
        global empty_frame, pause, is_deleting, frame, trackers, is_tracking, imgno, im0 
        cv.setMouseCallback("Video", handler)
        if k == ord('q'):
            sys.exit()
        elif k == ord('p'):
            pause = not pause 
            pause_video()
        elif k == ord('s'):
            num_trackers = begin_track(empty_frame, frame, trackers)
            if num_trackers > 0:
                is_tracking = True
            else: 
                is_tracking = False
        elif k == ord('w'):
            text = opt["output"] + "/img" + str(imgno) + ".png"
            cv.imwrite(text, frame)
            imgno += 1
        elif k == ord('x'):
            if pause:
                if is_deleting == True:
                    frame = im0.copy()
                    redraw()

                is_deleting = not is_deleting
        elif k == ord('l'):
            pass
            # List bounding box of ball


def redraw():
    """redraw takes no arguments. It updates the frame and tracked objects, and then shows the image."""
    global is_tracking, trackers, has_ball, ball_tracker, ball_bbox, empty_frame, has_ball_tracker
    if is_tracking:
        for key in trackers:
            track_ret, trackers[key].bbox = trackers[key].tracker.update(empty_frame)
            # Draw bounding box
            if track_ret:
                # Tracking success
                p1 = (int(trackers[key].bbox[0]), int(trackers[key].bbox[1]))
                p2 = (int(trackers[key].bbox[0] + trackers[key].bbox[2]), int(trackers[key].bbox[1] + trackers[key].bbox[3]))
                cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else :
                # Tracking failure
                cv.putText(frame, "Tracking failure detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    if has_ball:
        ball.draw_ctr
        ball_ret, ball_bbox = ball_tracker.update(empty_frame)
        # Draw bounding box
        if ball_ret:
            # Tracking success
            p1 = (int(ball_bbox[0]), int(ball_bbox[1]))
            p2 = (int(ball_bbox[0] + ball_bbox[2]), int(ball_bbox[1] + ball_bbox[3]))
            cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            ball.set_ctr(int(ball_bbox[0] + ball_bbox[2] // 2), int(ball_bbox[1] + ball_bbox[3] // 2))

        else :
            # Tracking failure
            has_ball_tracker = False

    return frame

def handler(event, x, y, flags, param):
    """Handle mouse events, and draw a red 'X' on mouse if paused and the delete key was pressed."""
    global is_deleting, frame, pause
    if pause:
        if is_deleting:
            if event == cv.EVENT_LBUTTONUP:
                delete_tracker(x, y, trackers)
                frame = im0.copy()
                redraw()
            cursor_frame = frame.copy()
            cv.line(cursor_frame, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), thickness=2)
            cv.line(cursor_frame, (x + 10, y - 10), (x - 10, y + 10), (0, 0, 255), thickness=2)
            cv.imshow("Video", cursor_frame)
    else:
        return
        
     
# Notes on global variables
# empty_frame: raw footage of video
# frame: final image to show after both detection and tracking
# im0: original "to show frame" from yolov5. shows detection but not tracking

def detect(opt, ball, save_img=False):
    global empty_frame, frame, bbox, video, tracker, is_tracking, trackers, pause, is_deleting, has_ball, ball_tracker, imgno, im0

    # Set arguments
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # webcam is bool
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    #print("device,", device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    #google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    # If using a webcam, view_img will open a feed and not save, we should have this on
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # path is the filename of video/image, img is a resized np array for frame
    # img0s is the raw frame, vid_cap is the VideoCapture object
    right = 0
    left = 0
    for path, img, im0s, vid_cap in dataset:
        #print("path ", path, "img", img, "im0s", im0s, "cap", vid_cap)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        empty_frame = im0s.copy()

        # Check tracker to see if a center can be determined
        ball_tracked = ball.check_tracker(empty_frame)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            #s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                
                # Find ball first
                found = 0
                for *xyxy, conf, cls in det:
                    # If detected object is ball
                    if names[int(cls)] == "ball":
                        # There is no tracker on the ball, and no ball has been found on this frame
                        if ball.has_ball:
                            # A ball is already being tracked.
                            # For now, replace it
                            if ball.conf > conf:
                                # Only update if the conf is greater than previous
                                found += 1
                                continue

                            # Set tracker on this ball, update ball properties
                            # update the ball center
                            ball.update(xyxy, conf, empty_frame)
                            #plot_one_box(xyxy, im0, label=names[int(cls)], color=colors[int(cls)], line_thickness=3)
                            found += 1

                        else:
                            # If a ball has been detected
                            if found > 0:
                                if ball.conf > conf:
                                    # Only update if the conf is greater than previous
                                    found += 1
                                    continue
                                else:
                                    # update the ball center, more likely to be ball
                                    ball.update(xyxy, conf, empty_frame)
                            #        plot_one_box(xyxy, im0, label=names[int(cls)], color=colors[int(cls)], line_thickness=3)
                                    found += 1

                            else:
                                ball.update(xyxy, conf, empty_frame)
                            #    plot_one_box(xyxy, im0, label=names[int(cls)], color=colors[int(cls)], line_thickness=3)
                                found += 1
 
                ball.frames.append([empty_frame,ball.bbox])
                if len(ball.frames) > 30:
                    ball.frames.pop(0)

                # If no ball detected, check to see if tracker has anything
                if found == 0:
                    if ball.has_tracker == False:
                        # Ball is lost
                        ball.has_ball = False

                #if found > 1:
                    #print("detected", found, "balls")


                # Write results
                for *xyxy, conf, cls in det:
                    """
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    """

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        if "ball" in label:
                            if found == 0:
                                print("an error occured")
                            pass
                        else:
                            if "net" in label:
                                
                                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                if ball.contained_in(c1,c2):
                                    print("basket")
                                    if ball.ctr[1] > frame.shape[1]//2:
                                        print("score right")
                                    else:
                                        print("score left")
                            elif "backboard" in label:
                                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                if ball.contained_in(c1,c2):
                                    pass
                                    #print("shot")

                            if opt.all or ("person" in label and opt.people):
                                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                if ball.contained_in(c1,c2) or ball.distance(find_center(c1,c2)) < 200:
                                    plot_one_box(xyxy, im0, label=label, color=[0,0,255], line_thickness=3)
                                else:
                                    plot_one_box(xyxy, im0, label=label, color=[255,0,0], line_thickness=3)

            # Print time (inference + NMS)
            frame = ball.draw_ctr(im0)
            # Stream results
            if view_img:
                #cv2.imshow(p, im0)
                frame = redraw()
                
               # cv.putText(frame, )
                cv2.imshow("Video", frame)
                cv2.imshow("backup",ball.frames[0][0])
                k = cv2.waitKey(1)
                if k == ord('q'):  # q to quit
                    raise StopIteration
                else:
                    check_commands(k)

            """
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        #print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
            """

    #print('Done. (%.3fs)' % (time.time() - t0))

# MAIN
if __name__ == "__main__":

    
    # construct the argument parse and parse the arguments
    # ARGUMENTS
    parser = argparse.ArgumentParser()


    # Frequently used
    parser.add_argument('-p', '--people', action='store_true', help='show detected people')
    parser.add_argument('-a', '--all', action='store_true', help='show everything')

    parser.add_argument('-s', '--source', type=str, default='/home/scott/MiddballHam.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('-o', '--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('-c', '--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='model.pt path')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-v', '--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    #opt.img_size = check_img_size(opt.img_size)

    
    cv.namedWindow("Video")

    # Initialize global variables. Additionally, create an empty dictionary and set is_tracking to False until an object is selected
    global frame, bbox, video, tracker, is_tracking, trackers, pause, is_deleting, has_ball, ball_tracker, imgno, has_ball_tracker
    imgno = 1
    is_tracking = False
    is_deleting = False
    trackers = dict()
    pause = False
    has_ball = False
    ball_tracker = cv2.TrackerKCF_create()
    has_ball_tracker = False
    ball = Ball()

    # If a file is specified, open the file, otherwise take from the camera
    video = cv.VideoCapture(opt.source)

    # Exit if video not opened
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    cv.setMouseCallback("Video", handler)

    with torch.no_grad():
        detect(opt, ball)
    """
        # Start timer
        timer = cv.getTickCount()
        # Calculate Frames per second (FPS)
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        # Display tracker type on frame
        cv.putText(frame, "Basketball Tracker", (100,20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        # Display FPS on frame
        cv.putText(frame, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        # Display result
        redraw()
        k = cv.waitKey(5) & 0xff
        check_commands(k)
    """


# release the file pointers
video.release()

