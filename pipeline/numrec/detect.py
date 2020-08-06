import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *

import numpy as np

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
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

    # create dict of players and identifications
    player_dict = {'2' : 'Kornaker', '3' : 'McDonald',
        '5' : 'You Mak', '11' : 'Palli', '13' : 'Baharozian',
        '15' : 'Mulloy', '22' : 'Davis', '23' : 'Kuntz',
        '24' : 'Young', '30' : 'Waldman', '35' : 'Miller',
        '42' : 'Knox', '55' : 'Sullivan'}
    
    # create dict of colors from different games
    game = opt.game
    color = ''
    color_dict = {'Hamilton' : 'blue', 'Castleton' : 'white'}
    if game in color_dict:
        color = color_dict[game]
    else:
        raise Exception('[ERROR] game not selected')
        quit
    
    # dictionary of identifications in last frame
    old_ids = {}
    
    # counter dictionary and unique digits
    count_dict = {}
    unique_digits = {}
    empty = []
    
    # identify unique numbers in dictionary
    for jersey in player_dict:
        for digit in jersey:
            
            # add to count dictionary
            if digit not in count_dict:
                count_dict[digit] = []
                
            # prevent adding jerseys with identical digits twice   
            if jersey not in count_dict[digit]:
                count_dict[digit].append(jersey)
    
    # check for any unique digits
    for digit in count_dict:
        if len(count_dict[digit]) == 1:
            unique_digits[digit] = count_dict[digit][0]
    
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

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
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            # get height and width of frame
            h, w = im0.shape[:2]
            
            if det is not None and len(det):
                
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                
                # create list of boxes, labels, confs in detection
                boxes = []
                labels = []
                confs = []
                
                # collect results
                for *xyxy, conf, cls in det:
                    
                    # convert tensor to int list and expand by 25 pixels in each direction
                    box = [int(xyxy[0])-25, int(xyxy[1])-25, int(xyxy[2])+25, int(xyxy[3])+25]
                    boxes.append(box)
                    confs.append(conf)
                    labels.append(names[int(cls)])
                    
                # temp boxes
                temp_boxes = boxes.copy()
                temp_labels = labels.copy()
                temp_confs = confs.copy()
                
                # intermediate lists
                int_boxes = []
                int_labels = []
                int_confs = []
                
                # iterate through old boxes and compare expansions
                for number in old_ids:
                    
                    # retrieve an old box
                    old_box = old_ids.get(number)
                    for i in range(len(boxes)):
                        
                        # retrieve the current box
                        new_box = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]
                            
                        x1_diff = abs(old_box[0] - new_box[0])
                        y1_diff = abs(old_box[1] - new_box[1])
                        x2_diff = abs(old_box[2] - new_box[2])
                        y2_diff = abs(old_box[3] - new_box[3])
                            
                        if (x1_diff < 25 and y1_diff < 25) or (x2_diff < 25 and y2_diff < 25):
                            
                            # box not yet appended
                            label = labels[i]
                            if boxes[i] not in int_boxes:
                                if label not in number:
                                    
                                    # check location of new digit
                                    if old_box[0] < new_box[0]:
                                        
                                        # check if single detection and order digits
                                        if len(number) == 1 and len(boxes) == 1:
                                            number = label
                                        else:
                                            number += label
                                    else:
                                        
                                        # check if single detection and order digits
                                        if len(number) == 1 and len(boxes) == 1:
                                            number = label
                                        else:
                                            number = label + number
                                
                                int_boxes.append(new_box)
                                int_labels.append(number)
                                int_confs.append(confs[i])
                                
                                # pop box
                                pop_index = temp_boxes.index(boxes[i])
                                temp_boxes.pop(pop_index)
                                temp_labels.pop(pop_index)
                                temp_confs.pop(pop_index)
                
                # iterate through boxes not matched to old boxes
                if temp_boxes:
                    for i in range(len(temp_boxes)):
                        int_boxes.append(temp_boxes[i])
                        int_labels.append(temp_labels[i])
                        int_confs.append(temp_confs[i])
                    
                # if no old ids
                else:
                    for i in range(len(boxes)):
                        int_boxes.append(boxes[i])
                        int_labels.append(labels[i])
                        int_confs.append(confs[i])
                
                # clear original structures
                boxes = int_boxes.copy()
                labels.clear()
                confs.clear()
                
                # last boxes and boxes to be looped through
                loop_boxes = int_boxes.copy()
                last_boxes = []
                    
                # iterate through int boxes and find overlaps
                for i in range(len(int_boxes)):
                    
                    # empty temp structures
                    temp_box = []
                    temp_label = ''
                    temp_conf = 0
                    for j in range(len(loop_boxes)):
                        
                        x1_diff = abs(int_boxes[i][0] - loop_boxes[j][0])
                        y1_diff = abs(int_boxes[i][1] - loop_boxes[j][1])
                        x2_diff = abs(int_boxes[i][2] - loop_boxes[j][2])
                        y2_diff = abs(int_boxes[i][3] - loop_boxes[j][3])
                        
                        if (x1_diff < 25 and y1_diff < 25) or (x2_diff < 25 and y2_diff < 25):
                            
                            # compare conf
                            if int_confs[j] < int_confs[i]:
                                temp_conf = int_confs[i]
                            else:
                                temp_conf = int_confs[j]
                            
                            # continue updating temp box to only add a single box
                            temp_box = loop_boxes[j]
                            
                            # remove boxes to prevent multiple overlaps
                            pop_index = boxes.index(loop_boxes[j])
                            boxes.pop(pop_index)
                            
                            # compare labels last to enable break from loop
                            label_i, label_j = int_labels[i], int_labels[j]
                            combo_1 = label_i + label_j
                            combo_2 = label_j + label_i
                            
                            # if combinations to double digit are in dictionary
                            if combo_1 in player_dict:
                                
                                # check leftmost box to get correct ordering
                                if int_boxes[i][0] < loop_boxes[j][0]:
                                    temp_label = combo_1
                                    break
                                
                            # check other combination
                            if combo_2 in player_dict:
                                    
                                # check leftmost box to get correct ordering
                                if loop_boxes[j][0] < int_boxes[i][0]:
                                    temp_label = combo_2
                                    break
                        
                            # single digits only and no change to temp label
                            if temp_label == '':
                                
                                # if single digits are in dictionary
                                if label_i in player_dict:
                                    temp_label = label_i
                                elif label_j in player_dict:
                                    temp_label = label_j
                                    break
                                
                                # check unique digits
                                if label_i in unique_digits:
                                    temp_label = unique_digits[label_i]
                                elif label_j in unique_digits:
                                    temp_label = unique_digits[label_j]
                    
                    # remove newly acquired overlapping boxes
                    loop_boxes = boxes.copy()
                    if temp_box:
                        
                        # run color check on box to determine team
                        check = color_check(im0, temp_box, color)
                        
                        # add box if correct color and non repeat label
                        if (check) and (temp_label not in labels):
                            last_boxes.append(temp_box)
                            labels.append(temp_label)
                            confs.append(temp_conf)
                        
                print('\n[INFO] old ids:', old_ids)
                
                # clear dictionary and add new boxes
                old_ids.clear()
                for i in range(len(last_boxes)):
                    old_ids[labels[i]] = last_boxes[i]
                    
                print('[INFO] final boxes:', last_boxes)
                print('[INFO] final labels:', labels)
                print('[INFO] final confs:', confs)
                    
                # iterate once more with condensed structures
                for i in range(len(last_boxes)):
                    
                    # write to file
                    if save_txt:  
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    
                    # draw the bounding boxes on the image
                    if save_img or view_img:
                        
                        # check the label against the player dictionary
                        number = labels[i]
                        if number in player_dict:
                            player = player_dict[number]
                        else:
                            player = 'unknown'
                            
                        label = '%s %.2f' % (player, confs[i])
                        text = 'ID: ' + label
                        
                        # draw rectangle around number detection
                        border = last_boxes[i]
                        cv2.rectangle(im0, (border[0], border[1]), (border[2], border[3]), (80,18,236), 3)
                        
                        # check if text is being written off frame and add label
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        if border[3] >= h:
                            cv2.rectangle(im0, (border[0]-2, border[1]-18), (border[2]+2, border[1]), (80,18,236), cv2.FILLED)
                            cv2.putText(im0, text, (border[0]+4, border[1]-7), font, 0.5, (255, 255, 255), 1)
                        else:
                            cv2.rectangle(im0, (border[0]-2, border[3]+18), (border[2]+2, border[3]), (80,18,236), cv2.FILLED)
                            cv2.putText(im0, text, (border[0]+4, border[3]+11), font, 0.5, (255, 255, 255), 1)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # q to quit
                    raise StopIteration
                if key == ord('p'):
                    cv2.waitKey()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    
def color_check(frame, box, color):
    
    # get relevant pixels and convert frame to hsv
    x1, y1, x2, y2 = abs(box[0]), abs(box[1]), abs(box[2]), abs(box[3])
    selection = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(selection, cv2.COLOR_BGR2HSV)
        
    # set range for color identification
    if color == 'blue':
        lower_range = np.array([40, 0, 0])
        upper_range = np.array([150, 130, 120])
    
    if color == 'white':
        lower_range = np.array([0, 0, 130])
        upper_range= np.array([255, 255, 255])
    
    # create mask, find non zero points and calculate area
    mask = cv2.inRange(hsv, lower_range, upper_range)
    points = cv2.findNonZero(mask)
    area = (x2 - x1) * (y2 - y1)
    
    # check jersey color
    #res = cv2.bitwise_and(selection, selection, mask=mask)
    #cv2.imshow('res', res)
    #cv2.waitKey()
    
    # compare colored points to area of box
    check = False
    if points is None:
        num_color = 0
    else:
        num_color = len(points)
    print('color number: ', num_color)
    print('area: ', area)
    if num_color > (area / 3):
        check = True
    return check
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--game', type=str, required=True, help='opponent to determine jersey color')
    opt = parser.parse_args()
    print(opt)
    
    # confidence threshold default 0.4, iou default 0.5

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                print('[INFO] using prepackaged model')
                detect()
                create_pretrained(opt.weights, opt.weights)
        else:
            print('[INFO] using custom model')
            detect()