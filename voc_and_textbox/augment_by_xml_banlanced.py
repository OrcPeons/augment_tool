#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function, division
import sys
sys.path.append('../')

from imgaug import augmenters as iaa
import imgaug as ia
import cv2#scimage不支持中文路径读图
from xml.etree import ElementTree
from lxml import etree
import codecs
import os

import copy
from collections import Counter

XML_EXT = '.xml'

#####################################################
# 1.读取xml并将每一类的类名称和对应的box位置记录下来；
# 2.根据不同类别之间的比例关系决定增强参数，尽量平衡样本数量；
#
#
######################################################
class PascalVocReader:
    def __init__(self, filepath):
        self.polygons = []
        self.classes = [] #add by wz
        self.filepath = filepath
        self.verified = False
        self.parseXML()

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True)

    # parse XML to get polygons
    def parseXML(self):
        assert self.filepath.endswith('.xml'), "Unsupport file format"
        parser = etree.XMLParser(encoding='utf-8')
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        self.root = xmltree
        filename = xmltree.find('filename').text
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find('bndbox')
            self.addPolygon(bndbox)

            name = object_iter.find('name')
            self.classes.append(name.text)
        return True

    # get polygons
    def getPolygons(self):
        return self.polygons

    #add by wz
    def getClasses(self):
        return self.classes

    # form xml get polygon and append it in polygons
    def addPolygon(self, polygon):
        xmin = eval(polygon.find('xmin').text)
        ymin = eval(polygon.find('ymin').text)
        xmax = eval(polygon.find('xmax').text)
        ymax = eval(polygon.find('ymax').text)
        points = [(xmin,ymin),(xmin,ymax),(xmax,ymin),(xmax,ymax)]
        self.polygons.append(points)

    # edit the polygons(data augmentation)
    def editPolygons(self,polygons):
        self.polygons = polygons

    # edit the filename(data augmentation)
    def editFilename(self, filename):
        fl_name = self.root.find('filename')
        fl_name.text = filename

    # save xml(data augmentation)
    def saveXML(self, xml_tree, targetFile=None):
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding='utf-8')
        else:
            out_file = codecs.open(targetFile, 'w', encoding='utf-8')
        prettifyResult = self.prettify(xml_tree)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()

    # convert polygon to bndbox(pascal voc xml)
    def convertPolygon2BndBox(self,polygon):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in polygon:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)
        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1
        if ymin < 1:
            ymin = 1
        return (int(xmin), int(ymin), int(xmax), int(ymax))

    def savePascalVocXML(self, targetFile = None):
        pascal_voc_tree = copy.deepcopy(self.root)
        num = 0
        for object_iter in pascal_voc_tree.findall('object'):
            bnd_box = self.convertPolygon2BndBox(self.polygons[num])
            #bndbox = SubElement(object_iter, 'bndbox')
            bndbox = object_iter.find("bndbox")
            xmin = bndbox.find('xmin')
            xmin.text = str(bnd_box[0])
            ymin = bndbox.find('ymin')
            ymin.text = str(bnd_box[1])
            xmax = bndbox.find('xmax')
            xmax.text = str(bnd_box[2])
            ymax = bndbox.find('ymax')
            ymax.text = str(bnd_box[3])
            num = num + 1
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding='utf-8')
        else:
            out_file = codecs.open(targetFile, 'w', encoding='utf-8')
        prettifyResult = self.prettify(pascal_voc_tree)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


if __name__ == '__main__':
    augthresh = 100
    voc_dir = "/home/wz/Desktop/丽水卡口数据/模型和训练数据/carlogo20171008/VOCdevkit2007/VOC2007"
    xml_dir = os.path.join(voc_dir,'Annotations_preaug')
    xml_save_dir = os.path.join(voc_dir,'Annotations_aug/')
    img_dir = os.path.join(voc_dir,'JPEGImages_preaug')
    img_save_dir = os.path.join(voc_dir,'JPEGImages_aug/')
    xmls = os.listdir(xml_dir)

    #统计每一类出现的个数
    all_classes_num = []
    for xml_ in xmls:
        reader = PascalVocReader(os.path.join(xml_dir, xml_))
        cur_classes = reader.getClasses()
        for cls in cur_classes:
            all_classes_num.append(cls)

    all_classes_num_counter = Counter(all_classes_num)
    print(all_classes_num_counter)
    for item in all_classes_num_counter:
        print(item,":",all_classes_num_counter[item])


    #############################################
    # 暂时只支持单个类别，即一张图里面只有一种类别，比如#
    # 车标类，一张车的图当中只会存在一个类型的车标。   #
    ############################################
    for xml_ in xmls:
        img_name = xml_[0:-4]+'.jpg'
        image = cv2.imread(os.path.join(img_dir, img_name))

        reader = PascalVocReader(os.path.join(xml_dir,xml_))
        #add by wz
        names = reader.getClasses()
        for name in names:
            if all_classes_num_counter[name] > augthresh:
                continue

        polygons = reader.getPolygons()
        keypoints = []
        for polygon in polygons:
            keypoints.extend([ia.Keypoint(point[0], point[1]) for point in polygon])
        keypoints = [ia.KeypointsOnImage(keypoints, shape=image.shape)]

        # draw on image
        image_keypoints = keypoints[0].draw_on_image(image, size=5)
        seq = iaa.Sequential([
            iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
            # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            # iaa.Affine(rotate=(-90,90)),
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                       # scale images to 80-120% of their size, individually per axis
                       translate_px={"x": (-16, 16), "y": (-16, 16)}, shear=(-16, 16)),
            iaa.Multiply((0.8, 1.2)),  # change brightness of images (50-150% of original value)
            iaa.ContrastNormalization((0.8, 1.2))
        ])
        for i in range(int(augthresh / all_classes_num_counter[names[0]])):
            # to_deterministic to make image and keypoints to a batch, so
            seq_det = seq.to_deterministic()
            image_aug = seq_det.augment_image(image)
            keypoints_aug = seq_det.augment_keypoints(keypoints)[0]
            # convert from keypoints to array, or else can't make it to list in below loop
            new_points = keypoints_aug.get_coords_array()
            new_polygons = []
            bndBoxs = []
            num = 0
            for polygon in polygons:
                # __str__ is class Keypoint's function
                # eval -- change str to tuple,list,dist and so on
                new_polygon = [(new_points[num + ind][0], new_points[num + ind][1]) for ind in range(len(polygon))]
                num = num + len(polygon)
                new_polygons.append(new_polygon)
            reader.editFilename(str(i))
            reader.editPolygons(new_polygons)
            # reader.saveXML(xml_tree,'my/'+str(i)+'.xml')
            reader.savePascalVocXML(xml_save_dir + xml_[0:-4] + str(i) + '.xml')
            cv2.imwrite(img_save_dir + "/" + xml_[0:-4] + str(i) + '.jpg', image_aug)
