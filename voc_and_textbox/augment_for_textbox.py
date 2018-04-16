#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function, division
import sys
sys.path.append('../')

import numpy as np
from scipy import misc
from imgaug import augmenters as iaa
import imgaug as ia

from scipy import ndimage, misc
from skimage import data

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs

import cv2 #just for debug
import os

# copy xml to generate pascal voc xlm
# copy.deepcopy
import copy

XML_EXT = '.xml'

class TextboxReader:
    def __init__(self, filepath):
        self.polygons = []
        self.filepath = filepath
        self.verified = False
        self.image_width = 0
        self.image_height = 0
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

        size = xmltree.find('size')
        self.image_width = int(size.find('width').text)
        self.image_height = int(size.find('height').text)
        print('image height, image width:', self.image_height, self.image_width)

        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find('bndbox')
            self.addPolygon(bndbox)
        return True

    # get polygons
    def getPolygons(self):
        return self.polygons

    # form xml get polygon and append it in polygons
    def addPolygon(self, polygon):
        x1 = eval(polygon.find('x1').text)
        y1 = eval(polygon.find('y1').text)
        x2 = eval(polygon.find('x2').text)
        y2 = eval(polygon.find('y2').text)
        x3 = eval(polygon.find('x3').text)
        y3 = eval(polygon.find('y3').text)
        x4 = eval(polygon.find('x4').text)
        y4 = eval(polygon.find('y4').text)
        xmin = eval(polygon.find('xmin').text)
        ymin = eval(polygon.find('ymin').text)
        xmax = eval(polygon.find('xmax').text)
        ymax = eval(polygon.find('ymax').text)
        points = [(x1,y1),(x2,y2),(x3,y3),(x4,y4),(xmin,ymin),(xmax,ymax)]
        self.polygons.append(points)

    # edit the filename(data augmentation)
    def editFilename(self, filename):
        fl_name = self.root.find('filename')
        fl_name.text = filename

    # convert polygon to bndbox(pascal voc xml)
    def convertPolygon2BndBox(self, polygon):
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
        if xmax > self.image_width - 1:
            xmax = self.image_width - 1
        if ymax > self.image_height - 1:
            ymax = self.image_height - 1
        return (int(xmin), int(ymin), int(xmax), int(ymax))

    def editPolygons(self, polygons):
        self.polygons = polygons

    def savePascalVocXML(self,targetFile=None):
        pascal_voc_tree = copy.deepcopy(self.root)
        num = 0

        for object_iter in pascal_voc_tree.findall('object'):
            pts = self.polygons[num]
            valid_pts = []
            for pt in pts[:4]:
                x = max(0, pt[0])
                x = min(x, self.image_width - 1)

                y = max(0, pt[1])
                y = min(y, self.image_height - 1)
                valid_pts.append([x,y])

            # print(len(valid_pts))
            # todo:若box超过一半在图片外则舍弃
            # area = abs((pts[4][0] - pts[5][0]) * (pts[4][1] - pts[5][1]))
            # area_inside = abs((valid_pts[4][0] - valid_pts[5][0]) * (valid_pts[4][1] - valid_pts[5][1]))
            # if area_inside * 1.0 / area < 0.5:
            #     continue

            #增强后点的顺序以及本身的xmin,ymin,xmax,ymax关系已经改变，重新确定
            valid_pts = np.array(valid_pts)
            sort_index = valid_pts[:, 0].argsort()
            valid_pts = valid_pts[sort_index]
            bndbox = object_iter.find("bndbox")

            leftpts = valid_pts[[0,1],:]
            rtpts = valid_pts[[2,3],:]
            sort_index = leftpts[:,1].argsort()
            leftpts = leftpts[sort_index]
            sort_index = rtpts[:, 1].argsort()
            rtpts = rtpts[sort_index]

            x1 = bndbox.find('x1')
            x1.text = str(leftpts[0][0])
            y1 = bndbox.find('y1')
            y1.text = str(leftpts[0][1])

            x2 = bndbox.find('x2')
            x2.text = str(rtpts[0][0])
            y2 = bndbox.find('y2')
            y2.text = str(rtpts[0][1])

            x3 = bndbox.find('x3')
            x3.text = str(rtpts[1][0])
            y3 = bndbox.find('y3')
            y3.text = str(rtpts[1][1])

            x4 = bndbox.find('x4')
            x4.text = str(leftpts[1][0])
            y4 = bndbox.find('y4')
            y4.text = str(leftpts[1][1])

            xmin = bndbox.find('xmin')
            xmin.text = str(np.min(valid_pts[:,0]))
            ymin = bndbox.find('ymin')
            ymin.text = str(np.min(valid_pts[:,1]))

            xmax = bndbox.find('xmax')
            xmax.text = str(np.max(valid_pts[:,0]))
            ymax = bndbox.find('ymax')
            ymax.text = str(np.max(valid_pts[:,1]))
            num = num + 1

        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding='utf-8')
        else:
            out_file = codecs.open(targetFile, 'w', encoding='utf-8')
        prettifyResult = self.prettify(pascal_voc_tree)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


if __name__ == '__main__':
    voc_dir = "/home/wz/Data/VIN/viechleLicense/VOC2007/CROP/augment_textbox_merge"
    xml_dir = os.path.join(voc_dir, 'xml')
    img_dir = os.path.join(voc_dir, 'img')
    xmls = os.listdir(xml_dir)
    for xml_ in xmls:
        img_name = xml_[0:-4]+'.jpg'
        if not os.path.exists(os.path.join(img_dir, img_name)):
            continue
        image = ndimage.imread(os.path.join(img_dir, img_name))

        reader = TextboxReader(os.path.join(xml_dir, xml_))
        polygons = reader.getPolygons()
        keypoints = []
        for polygon in polygons:
            keypoints.extend([ia.Keypoint(point[0], point[1]) for point in polygon])
        keypoints = [ia.KeypointsOnImage(keypoints, shape=image.shape)]

        # print(keypoints)
        # # draw on image
        # image_keypoints = keypoints[0].draw_on_image(image, size=5)
        seq = iaa.Sequential([
            iaa.Crop(px=(0, 10)),  # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            # iaa.Flipud(0.5),
            iaa.Affine(rotate=(-10, 10)),#仿she增强没问题，是在画图的时候最小外接举行已经不再是4,5了，而是得重新计算
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                       # scale images to 80-120% of their size, individually per axis
                      translate_px={"x": (-8, 8), "y": (-8, 8)}),
            #iaa.Affine(scale=(0.8,1.0)),
            #     shear=(-30, 30)),
            iaa.Multiply((0.8, 1.2)),  # change brightness of images (50-150% of original value)
            iaa.ContrastNormalization((0.8, 1.2))
        ])
        for i in range(20):
            seq_det = seq.to_deterministic()
            image_aug = seq_det.augment_image(image)
            keypoints_aug = seq_det.augment_keypoints(keypoints)[0]
            # convert from keypoints to array, or else can't make it to list in below loop
            new_points = keypoints_aug.get_coords_array()
            new_polygons = []
            bndBoxs = []
            num = 0
            for polygon in polygons:
                new_polygon = [(new_points[num + ind][0], new_points[num + ind][1]) for ind in range(len(polygon))]
                num = num + len(polygon)
                new_polygons.append(new_polygon)
            reader.editFilename(str(i))
            reader.editPolygons(new_polygons)

            #Just for debug
            # for poly in new_polygons:
            #     poly = np.array(poly)
            #     xmin = min(poly[:,0])
            #     ymin = min(poly[:,1])
            #     xmax = max(poly[:,0])
            #     ymax = max(poly[:,1])
            #     cv2.rectangle(image_aug, (xmin, ymin), (xmax, ymax), (0,255,0))
            #     cv2.imshow('imgaug', image_aug)
            #     cv2.waitKey(0)

            reader.savePascalVocXML(voc_dir+'/xml_aug/' + xml_[0:-4] + '_'+str(i) + '.xml')
            misc.imsave(voc_dir+'/img_aug/' + xml_[0:-4] + '_'+str(i) + '.jpg', image_aug)
