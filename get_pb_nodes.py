#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
There is a tensorflow model frozen pb file.
Using this script we can get and save all nodes' imformation
in pb file.
"""

import tensorflow as tf
import collections
import tqdm
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pb')
    args = parser.parse_args()
    return args


class PbReader(object):
    """load and read a pb file
    Aegument:
        pb_file_path: str, the path of pb file, like ./test.pb.
    """
    def __init__(self, pb_file_path):
        self.pb_file_path = pb_file_path
        self.graph_def = self.loadGraphDef()

    def loadGraphDef(self):
        """get graphdf of pb file"""
        with tf.Session() as sess:
            with open(self.pb_file_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
        return graph_def

    def getGraphNodes(self):
        """get nodes information in graph def"""
        self.nodes = collections.OrderedDict()
        for node in self.graph_def.node:
            self.nodes[node.name] = node
        return self.nodes

    def saveNodes2Text(self, save_path):
        """save all nodes to file"""
        nodes_list = [str(node) for node in self.nodes.values()]
        with open(save_path, 'w') as f:
            for node in tqdm.tqdm(nodes_list):
                f.write(node + '\n')
        print('Saved {} nodes to the file: {}'.format(len(nodes_list), save_path))


if __name__ == "__main__":
    args = get_args()
    pb_file_path = args.pb  # '/home/wangtf/ShareDataset/test_for_tensorrt.pb'
    save_path = './graph_nodes.txt'
    pb_reader = PbReader(pb_file_path)
    nodes = pb_reader.getGraphNodes()
    pb_reader.saveNodes2Text(save_path)

