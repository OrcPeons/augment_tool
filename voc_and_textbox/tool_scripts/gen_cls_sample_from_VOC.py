#!/usr/bin/env python
#-- coding:utf-8 --
import os.path as osp
import os
import cv2
from pascal_voc_io import  *


cur_dir = osp.dirname(__file__)

#size for classification net's input.
RESIZE_WIDTH = 224
RESIZE_HEIGHT = 224
IMAGE_SIZE = (RESIZE_WIDTH, RESIZE_HEIGHT)

#pathes of input xml and image, and output image path.
xml_folder        = '/home/wz/Data/VIN/cut_num_eng/VOC2007/xml_pva_black'
image_src_folder  = '/home/wz/Data/VIN/cut_num_eng/VOC2007/img_pva_black'
image_save_root   = '/home/wz/Data/VIN/cut_num_eng/VOC2007/CROP'
classes_name_txt  = '/home/wz/Data/VIN/cut_num_eng/only_num_eng_VOC/classes_name.txt'

if not os.path.exists(image_save_root):
    os.mkdir(image_save_root)

CLASSES = []
fin = open(classes_name_txt)
for line in fin.readlines(  ):
    CLASSES.append(line[:-1])
    if not os.path.isdir(os.path.join(image_save_root, line[:-1])):
        os.mkdir(os.path.join(image_save_root, line[:-1]))

class_index = dict(zip(CLASSES, len(CLASSES)*[0]))
print class_index

for xml_name in os.listdir(xml_folder):
    xml_reader = PascalVocReader(os.path.join(xml_folder, xml_name))
    if len(xml_reader.getShapes()) > 0:
        img = cv2.imread(os.path.join(image_src_folder, xml_name[:-4] + '.jpg'))
        if img is None:
            img = cv2.imread(os.path.join(image_src_folder, xml_name[:-4] + '.JPG'))
        for shape in xml_reader.getShapes():
            label = shape[0]
            class_index[label] += 1
            points = shape[1]
            left_top = points[0]
            right_top = points[1]
            right_bottom = points[2]
            left_bottom = points[3]
            crop_img = img[max(0,left_top[1]): min(img.shape[0],left_bottom[1]), max(0,left_top[0]): min(img.shape[1],right_top[0])]
            if crop_img is None:
                continue
            if crop_img.shape[0] <= 0 or crop_img.shape[1]<=0:continue
            resized_crop_img = cv2.resize(crop_img, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(image_save_root, label, label + '_' + str(class_index[label]) + ".jpg"), resized_crop_img)