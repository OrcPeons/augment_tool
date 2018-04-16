#-- coding:utf-8 --
import sys
sys.path.append("..")

from pascal_voc_io import  *
import os
import cv2
import numpy as np


###################################################################
#
# 制作行驶证车辆识别代号样本
# 策略：
# 1）进入的图片先按照固定长宽比缩放（640×1056）
# 2）尽量保持字符的形状比例，不足的地方补黑(或者补白)
###################################################################

if __name__ == '__main__':
    ori_xml_dir = '/home/wz/Data/VIN/cut_num_eng/xml_only_eng_num_split'
    ori_img_dir = '/home/wz/Data/VIN/cut_num_eng/img_only_eng_num_split'

    res_xml_dir = '/home/wz/Data/VIN/cut_num_eng/xml_pva_black'
    res_img_dir = '/home/wz/Data/VIN/cut_num_eng/img_pva_black'

    xml_names = os.listdir(ori_xml_dir)
    img_names = os.listdir(ori_img_dir)

    for xml in xml_names:
        xml_path = os.path.join(ori_xml_dir, xml)
        img_name = (xml[:-4]+'.jpg')
        if not img_name in img_names:
            continue

        img_path = os.path.join(ori_img_dir, img_name)
        ori_img = cv2.imread(img_path)

        height = ori_img.shape[0]
        width = ori_img.shape[1]
        if width > 1056:
            continue

        scale = width * 1.0 / height

        standard_width = 1056
        standard_height =int(standard_width / scale)

        hori_border = True
        if standard_height > 640:
            scale = height * 1.0 / width
            standard_height = 640
            standard_width = int(standard_height / scale)
            hori_border = False

        res_img = np.zeros((640, 1056, 3), dtype=np.uint8)
        standard_img = cv2.resize(ori_img, dsize=(standard_width, standard_height), interpolation=cv2.INTER_CUBIC)
        margin = 0
        if hori_border:
            margin = (640 - standard_height) / 2
            res_img[margin:margin + standard_height, :, :] = standard_img
        else:
            margin = (1056 - standard_width) / 2
            res_img[:, margin:margin + standard_width, :] = standard_img
        cv2.imwrite(os.path.join(res_img_dir, xml[:-4] + '.jpg'), res_img)

        xml_reader = PascalVocReader(xml_path)
        xml_writer = PascalVocWriter(foldername = ori_img_dir, filename=xml[:-4], imgSize=res_img.shape)

        for shape in xml_reader.getShapes():
            points = shape[1]
            left_top = points[0]
            right_top = points[1]
            right_bottom = points[2]
            left_bottom = points[3]

            label = shape[0]

            zoomout_scale_w = standard_width * 1.0 / width
            zoomout_scale_h = standard_height * 1.0 / height

            if hori_border:
                xml_writer.addBndBox(int(left_top[0] * zoomout_scale_w),
                                     int(left_top[1] * zoomout_scale_h + margin),
                                     int(right_bottom[0] * zoomout_scale_w),
                                     int(right_bottom[1] * zoomout_scale_h + margin),
                                     label)
            else:
                xml_writer.addBndBox(int(left_top[0] * zoomout_scale_w + margin),
                                     int(left_top[1] * zoomout_scale_h),
                                     int(right_bottom[0] * zoomout_scale_w + margin),
                                     int(right_bottom[1] * zoomout_scale_h),
                                     label)

        xml_writer.save(os.path.join(res_xml_dir, xml[:-4] + XML_EXT))
        print('Processed img %s.' % xml)