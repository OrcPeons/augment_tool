import json
import codecs
import os
import cv2
import random

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree


class TextboxWriter:

    def __init__(self, foldername, filename, imgSize):
        self.foldername = foldername
        self.filename = filename
        self.imgSize = imgSize
        self.boxlist = []
        self.verified = False
        self.XML_EXT = '.xml'

    def setFolderName(self, foldername):
        self.foldername = foldername

    def setFileName(self,filename):
        self.filename = filename

    def setSize(self, imgsize):
        self.imgSize = imgsize


    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True)

    def addBndBox(self,x1,y1,x2,y2,x3,y3,x4,y4, xmin, ymin, xmax, ymax, content = '###'):
        bndbox = {'x1':x1,'y1':y1,
                  'x2':x2,'y2':y2,
                  'x3':x3,'y3':y3,
                  'x4':x4,'y4':y4,
                  'xmin': xmin, 'ymin': ymin,
                  'xmax': xmax, 'ymax': ymax}
        bndbox['content'] = content
        self.boxlist.append(bndbox)

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        top.set('verified', 'yes' if self.verified else 'no')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])

        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        return top


    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')

            difficult = SubElement(object_item, 'difficult')
            difficult.text = "0"
            content = SubElement(object_item, 'content')
            content.text=each_object['content']
            name = SubElement(object_item, 'name')
            name.text = 'text'

            bndbox = SubElement(object_item, 'bndbox')
            x1 = SubElement(bndbox, 'x1')
            x1.text = str(each_object['x1'])
            y1 = SubElement(bndbox, 'y1')
            y1.text = str(each_object['y1'])
            x2 = SubElement(bndbox, 'x2')
            x2.text = str(each_object['x2'])
            y2 = SubElement(bndbox, 'y2')
            y2.text = str(each_object['y2'])
            x3 = SubElement(bndbox, 'x3')
            x3.text = str(each_object['x3'])
            y3 = SubElement(bndbox, 'y3')
            y3.text = str(each_object['y3'])
            x4 = SubElement(bndbox, 'x4')
            x4.text = str(each_object['x4'])
            y4 = SubElement(bndbox, 'y4')
            y4.text = str(each_object['y4'])

            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + self.XML_EXT, 'w', encoding='utf-8')
        else:
            out_file = codecs.open(targetFile, 'w', encoding='utf-8')

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


def transform_json_to_xml():
    # data root dir includes json file and images.
    data_root = '/home/wz/Data/VIN/viechleLicense/VOC2007/CROP/License'
    save_dir = '/home/wz/Data/VIN/viechleLicense/VOC2007/CROP/xml'

    file_names = os.listdir(data_root)
    for file_name in file_names:
        if file_name.endswith('.json'):
            image = cv2.imread(os.path.join(data_root, file_name[:-5] + '.jpg'))
            if image is None:
                continue
            with open(os.path.join(data_root, file_name), 'r') as f:
                js_data = json.load(f)

                xml_writer = TextboxWriter(None, None, None)
                xml_writer.setFolderName('License')
                xml_writer.setSize(image.shape)

                for key in js_data.keys():
                    if key == u'shapes':
                        for item_dict in js_data[key]:
                            pts = item_dict[u'points']
                            if len(pts) != 4: break
                            x1, y1, x2, y2, x3, y3, x4, y4 = int(pts[0][0]), int(pts[0][1]), int(pts[1][0]), int(
                                pts[1][1]), \
                                                             int(pts[2][0]), int(pts[2][1]), int(pts[3][0]), int(
                                pts[3][1])

                            xmin = min(x1, x2, x3, x4)
                            xmax = max(x1, x2, x3, x4)
                            ymin = min(y1, y2, y3, y4)
                            ymax = max(y1, y2, y3, y4)
                            xml_writer.addBndBox(x1, y1, x2, y2, x3, y3, x4, y4, xmin, ymin, xmax, ymax, '###')

                    if key == 'imagePath':
                        xml_writer.setFileName(js_data[key])
                xml_writer.save(os.path.join(save_dir, file_name[:-5] + '.xml'))

def gen_train_testset():
    xml_root_path = '/home/wz/Data/VIN/viechleLicense/VOC2007/CROP/augment_textbox/xml_aug'
    img_root_path = '/home/wz/Data/VIN/viechleLicense/VOC2007/CROP/augment_textbox/img_aug'
    xmls = os.listdir(xml_root_path)
    random.shuffle(xmls)
    test_ratio = 0.1
    train_xmls = xmls[:int(test_ratio*len(xmls))]
    test_xmls = xmls[int(test_ratio*len(xmls)):]
    with open('train.txt', 'w') as fout:
        for xml in train_xmls:
            fout.writelines(os.path.join(img_root_path, xml[:-4]+'.jpg') + ' '+
                            os.path.join(xml_root_path, xml) + '\n')
    with open('test.txt', 'w') as fout:
        for xml in test_xmls:
            fout.writelines(os.path.join(img_root_path, xml[:-4]+'.jpg') + ' '+
                            os.path.join(xml_root_path, xml) + '\n')


if __name__ == '__main__':
    gen_train_testset()
