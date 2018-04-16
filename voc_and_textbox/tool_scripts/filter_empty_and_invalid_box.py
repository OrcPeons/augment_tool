#-- coding:utf-8 --
###################################################################
# **************************wz-2017-11-28********************      #
# 过滤不包含物体的xml以及因为增强而导致的bbox在图外的框
# 可以根据需要选择过滤特定的类别
###################################################################
from pascal_voc_io import *

import os
import shutil
import cv2
import numpy as np

xml_folder = '/home/wz/Data/VIN/cut_num_eng/VOC2007/Annotations'
image_folder = '/home/wz/Data/VIN/cut_num_eng/VOC2007/JPEGImages'
filtered_xml_folder = '/home/wz/Data/VIN/cut_num_eng/VOC2007/xml_filter'

if not os.path.exists(filtered_xml_folder):
    os.mkdir(filtered_xml_folder)
else:
    os.rename(filtered_xml_folder, filtered_xml_folder+'_backup')
    os.mkdir(filtered_xml_folder)

empty_num = 0
filtered_num = 0
image_names = os.listdir(image_folder)

def iou_in_img(rect, img_shape):
    '''
    :param rect: [lefttop, rtbt]
    :param img_shape: width,height
    :return: ratio of rect in img.
    '''
    x_min = max(rect[0][0], 0)
    y_min = max(rect[0][1], 0)
    x_max = min(rect[1][0], img_shape[1] - 1)
    y_max = min(rect[1][1], img_shape[0] - 1)
    area_inside = (x_max - x_min) * (y_max - y_min)
    area_rect = (rect[1][0] - rect[0][0]) * (rect[1][1] - rect[0][1])
    return area_inside * 1.0 / area_rect

iou_num = 0
for xml_name in os.listdir(xml_folder):
    xml_reader = PascalVocReader(os.path.join(xml_folder, xml_name))
    if xml_reader.width < 10 or xml_reader.height < 10 or len(xml_reader.getShapes()) < 4:
        empty_num += 1
        continue
    else:
        if not os.path.exists(os.path.join(xml_folder, xml_name)):
            continue
        if not os.path.exists(os.path.join(image_folder, xml_name[:-4] + '.jpg')):
            continue
        img = cv2.imread(os.path.join(image_folder, xml_name[:-4] + '.jpg'))
        xml_writer = PascalVocWriter(foldername=image_folder, filename=xml_name[:-4],
                                     imgSize=img.shape)
        for shape in xml_reader.getShapes():
            points = shape[1]
            left_top = points[0]
            right_top = points[1]
            right_bottom = points[2]
            left_bottom = points[3]
            label = shape[0]

            if label == 'OTHER' or label == 'VIN': continue

            iou = iou_in_img([left_top, right_bottom], img.shape)
            if iou < 0.7:
                iou_num += 1
                continue
            xml_writer.addBndBox(int(left_top[0]),
                                 int(left_top[1]),
                                 int(right_bottom[0]),
                                 int(right_bottom[1]),
                                 label)
        xml_writer.save(os.path.join(filtered_xml_folder, xml_name))

# rename folder
shutil.move(xml_folder, xml_folder+'_bef_filter')
shutil.move(filtered_xml_folder, xml_folder)

print('Total filtered file number is', iou_num)
