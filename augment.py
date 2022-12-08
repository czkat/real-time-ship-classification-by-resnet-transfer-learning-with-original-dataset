import cv2
import numpy as np
import re
from tqdm import tqdm
import os
import random
from PIL import Image, ImageEnhance

def augment(image):    
    def transform():
        return random.choice([0,1,2])

    # every image has to flip    
    transform_seed = transform()         
    if transform_seed == 0:
        image = cv2.flip(image, 0) #horizontal 
    elif transform_seed == 1:
        image = cv2.flip(image, 1) #vert
    else:
        image = cv2.flip(image, -1) #both

    # every image also has to rotate
    transform_seed2 = transform()
    if transform_seed2 == 0:
        image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE) 
    elif transform_seed2 == 1:
        image = cv2.rotate(image, cv2.ROTATE_180)
    else:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image

# read img
# def fast_scandir(dirname):
#     subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
#     for dirname in list(subfolders):
#         subfolders.extend(fast_scandir(dirname))
#     return subfolders

# read_dir = fast_scandir('/media/zheng/backup/shipclassification/dataset/split_aug/split/train/')

read_dir = ['/mnt/data2/Projects/BuildingDetection/ShipClassification/split/train2','/mnt/data2/Projects/BuildingDetection/ShipClassification/split/test2']

expand_times = 4

for dire in read_dir:    
    for filename in os.listdir(dire):                    
        path = dire + '/' + filename
        image = cv2.imread(path)
        filename = filename[:-4]
        
        for i in range(expand_times): 
            img_aug = augment(image)
            filename = filename + '_' + str(i) + '.png'
            if dire == '/mnt/data2/Projects/BuildingDetection/ShipClassification/split/train2':
                save_dir = '/mnt/data2/Projects/BuildingDetection/ShipClassification/split/train3'
            else:
                save_dir = '/mnt/data2/Projects/BuildingDetection/ShipClassification/split/test3'
                
            cv2.imwrite(os.path.join(save_dir, filename), img_aug)
            filename = filename[:-6]

print('augment finished')

