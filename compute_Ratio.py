#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import glob
from decode_xml import XmlReader


def getXmlInfo(xml_path, ratio_list, diagonal_list):
    reader = XmlReader(xml_path)
    for obj in reader.root.findall("object"):
        if reader.getObjectName(obj) not in ["smoke", "smoking"]:
            continue
        print("Object Name: {}".format(reader.getObjectName(obj)))
        bbox = reader.getObjectBndbox(obj)
        width = int(bbox[2]) - int(bbox[0])
        height = int(bbox[3]) - int(bbox[1])
        diagonal = (width ** 2 + height ** 2) ** 0.5
        ratio = float(width) / height
        ratio_list[reader.getObjectName(obj)].append(ratio)
        diagonal_list[reader.getObjectName(obj)].append(diagonal)


def countRatioAndDiagonal(xml_dir):
    assert os.path.isdir(xml_dir)    
    ratio_list = {"smoke":[], "smoking":[]}
    diagonal_list = {"smoke":[], "smoking":[]}
    xml_list = glob.glob(os.path.join(xml_dir, "*.xml"))
    for xml_path in xml_list:
        assert os.path.isfile(xml_path)
        getXmlInfo(xml_path, ratio_list, diagonal_list)
    
    return ratio_list, diagonal_list
    

import matplotlib.pyplot as plt
def drawPlot(ratio_list, diagonal_list):
    for key in ratio_list:
        x = ratio_list[key]
        y = diagonal_list[key]
        plt.scatter(x, y)
        plt.title(key)
        plt.xlabel("ratio")
        plt.ylabel("diagonal")
        plt.savefig("./{}.png".format(key))
        plt.show()


def main():
    xml_dir = "/home/wangtf/ShareDataset/dataset/Smoking/data/VOC2007/Annotations"
    ratio_list, diagonal_list = countRatioAndDiagonal(xml_dir)
    drawPlot(ratio_list, diagonal_list)
    

if __name__ == "__main__":
    main()

