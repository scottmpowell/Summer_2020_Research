import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *


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
#     player_dict = {'2' : 'Kornaker', '3' : 'McDonald',
#         '5' : 'You Mak', '11' : 'Palli', '13' : 'Baharozian',
#         '15' : 'Mulloy', '22' : 'Davis', '23' : 'Kuntz',
#         '24' : 'Young', '30' : 'Waldman', '35' : 'Miller',
#         '42' : 'Knox', '55' : 'Sullivan'}
    
    player_dict = {'3' : 'McDonald', '5' : 'You Mak',
        '11' : 'Palli', '13' : 'Baharozian',
        '15' : 'Mulloy', '22' : 'Davis', '23' : 'Kuntz',
        '24' : 'Young', '30' : 'Waldman', '35' : 'Miller',
        '42' : 'Knox',}
    
    old_ids = {}

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
                    
                    # convert tensor to int list
                    box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    boxes.append(box)
                    confs.append(conf)
                    labels.append(names[int(cls)])
                
                # final lists
                final_boxes = []
                final_labels = []
                final_confs = []
                
                # check for players with same digit in number
                combo_boxes = []
                
                # iterate through boxes and compare corners
                for i in range(len(boxes)):
                    for j in range(len(boxes)):
                            
                        x_diff = abs(boxes[i][2] - boxes[j][0])
                        y1_diff = abs(boxes[i][1] - boxes[j][1])
                        y2_diff = abs(boxes[i][3] - boxes[j][3])
                            
                        if (x_diff < 12 and y1_diff < 12) or (x_diff < 12 and y2_diff < 12):
                
                            class_i, class_j = labels[i], labels[j]
                            new_num = class_i + class_j
                            
                            # catch repeated combinations of boxes
                            if new_num not in final_labels:
                                
                                print('[INFO] new label:', new_num)
                                final_labels.append(new_num)
                                
                                conf_i, conf_j = confs[i], confs[j]    
                                if conf_i > conf_j:
                                    final_confs.append(conf_i)
                                else:
                                    final_confs.append(conf_j)
                                
                                new_box = [boxes[i][0], boxes[i][1], boxes[j][2], boxes[j][3]]
                                final_boxes.append(new_box)
                                combo_boxes.append(new_box)
                                
                # iterate again and add box if not duplicate detection            
                for i in range(len(boxes)):
                    
                    x1, y1, x2, y2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                    current_label = labels[i]
                    conf = confs[i]
                        
                    if combo_boxes:
                        
                        # only add boxes if non overlapping
                        for k in range(len(combo_boxes)):
                                
                            c_x1, c_y1, c_x2, c_y2 = combo_boxes[k][0], combo_boxes[k][1], combo_boxes[k][2], combo_boxes[k][3]
                            x1_diff = abs(x1 - c_x1)
                            y1_diff = abs(y1 - c_y1)
                            x2_diff = abs(x2 - c_x2)
                            y2_diff = abs(y2 - c_y2)
                                
                            if (x1_diff > 15 and y2_diff > 15) or (x2_diff > 15 and y2_diff > 15):
                                    
                                final_boxes.append(boxes[i])
                                final_labels.append(current_label)
                                final_confs.append(conf)
                    else:
                        
                        # non overlapping, but possible repeat
                        if current_label not in final_labels:
                                
                            final_boxes.append(boxes[i])
                            final_labels.append(current_label)
                            final_confs.append(conf)
                
                # temp lists
                temp_addbox = []
                temp_addnumber = []
                
                # copy boxes for popping and track labels
                temp_labels = final_labels.copy()
                
                print('old ids:', old_ids)
                # check dictionary for identifications in last frame
                for number in old_ids:
                
                    old_box = old_ids.get(number)
                    for i in range(len(final_boxes)):
                        
                        # check if label exists
                        if final_labels[i] not in player_dict:
                            final_labels[i] = final_labels[i][0]
                        
                        new_box = final_boxes[i]
                        x1_diff = abs(old_box[0] - new_box[0])
                        y1_diff = abs(old_box[1] - new_box[1])
                        x2_diff = abs(old_box[2] - new_box[2])
                        y2_diff = abs(old_box[3] - new_box[3])
                        
                        # previous box and new box are overlapping
                        if (x1_diff < 20 and y1_diff < 20) or (x2_diff < 20 and y2_diff < 20):
                            
                            # check for combo boxes
                            if new_box in combo_boxes:
                                temp_addnumber.append(final_labels[i])
                            
                            else:
                                final_labels[i] = number
                                temp_addnumber.append(number)
                        
                        else:
                            final_labels[i] = number
                            temp_addnumber.append(number)
                        
                        temp_addbox.append(new_box)

                # if no identifications in last frame
                if not old_ids:
                    if boxes:
                        for i in range(len(final_boxes)):
                            temp_addbox.append(final_boxes[i])
                            temp_addnumber.append(labels[i])
                
                # clear dictionary and add new boxes
                old_ids.clear()
                for i in range(len(final_boxes)):
                    old_ids[temp_addnumber[i]] = temp_addbox[i]
                    
                # iterate once more with condensed structures
                for i in range(len(final_boxes)):
                    
                    print('final boxes:', final_boxes)
                    print('final labels:', final_labels)
                    print('final confs:', final_confs)
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        number = final_labels[i]
                        if number in player_dict:
                            player = player_dict[number]
                        else:
                            player = 'unknown'
                        
                        label = '%s %.2f' % (player, final_confs[i])
                        plot_one_box(final_boxes[i], im0, label=label, color=colors[i], line_thickness=3)

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
