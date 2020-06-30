from PIL import Image
from detector import detect_faces
from align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import os
from tqdm import tqdm
import argparse
from imutils import paths

"""
Adapted from 'face.evoLVe: High-Performance Face Recognition
Library Based on PyTorch' by Jian Zhao
*Dependencies also come from same source

URL: https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
"""

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description = "face alignment")
    ap.add_argument("-source_root", "--dataset",
            help = "specify your source directory", type = str)
    ap.add_argument("-dest_root", "--align",
            help = "specify your destination directory", type = str)
    ap.add_argument("-crop_size", "--cropsize", default=112,
            help = "specify size of aligned faces, align and crop with padding", type = int)
    args = ap.parse_args()
    
    # assign arguments to correct variables
    new_path = os.getcwd()
    source_path = args.dataset.split("/")
    for i in source_path:
        new_path += "\\" + str(i)
    source_root = new_path
    dest_root = args.align
    crop_size = args.cropsize
    
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale
 
    # if destination folder does not exist, create it
    if not os.path.isdir(dest_root):
        os.mkdir(dest_root)
 
    for subfolder in tqdm(os.listdir(source_root)):
        if not os.path.isdir(os.path.join(dest_root, subfolder)):
            os.mkdir(os.path.join(dest_root, subfolder))
        for image_name in os.listdir(os.path.join(source_root, subfolder)):
            print("Processing\t{}".format(os.path.join(source_root, subfolder, image_name)))
            img = Image.open(os.path.join(source_root, subfolder, image_name))
            try: # Handle exception
                _, landmarks = detect_faces(img)
            except Exception:
                print("[INFO] {} is discarded due to exception!".format(os.path.join(source_root, subfolder, image_name)))
                continue
            if len(landmarks) == 0: # If the landmarks cannot be detected, the img will be discarded
                print("[INFO] {} is discarded due to non-detected landmarks!".format(os.path.join(source_root, subfolder, image_name)))
                continue
            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
            img_warped = Image.fromarray(warped_face)
            if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg']: #not from jpg
                image_name = '.'.join(image_name.split('.')[:-1]) + '.jpg'
            img_warped.save(os.path.join(dest_root, subfolder, image_name))

def align(image, crop):
    
    # specify size of aligned faces, align and crop with padding
    crop_size = crop 
    scale = crop_size / 112
    reference = get_reference_facial_points(default_square = True) * scale

    img = image
    try: # Handle exception
        _, landmarks = detect_faces(img)
    except Exception:
        print("[INFO] Discarded due to exception!")
    if len(landmarks) == 0: # If the landmarks cannot be detected, the img will be discarded
        print("[INFO] Discarded due to no landmarks detected!")
    facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
    img_warped = Image.fromarray(warped_face)
    return(img_warped)