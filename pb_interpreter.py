#!/usr/bin/env python3
# coding:utf-8


import numpy as np
import os
import sys
import random
import tensorflow as tf
import time
from object_detection.utils import dataset_util, label_map_util
# import evaluate
import cv2
import json
import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from absl import app
from absl import flags

voc_root = '/diskb/GlodonDataset/HeadDet/v0.1/testin/VOC2012'


def get_args():
    flags.DEFINE_string('image_path', "", './abc/.../def/,an empty folder path to write answers down')
    flags.DEFINE_string('main_set', "test", 'a dir like VOCdevkit/VOC2007/ImageSets/Main')
    flags.DEFINE_string('pb', "frozen_inference_graph.pb", 'an inference pb file')
    flags.DEFINE_string('size', '416,416', 'a txt file which is arranged as `id:class_name` each line')
    flags.DEFINE_string('label_path', '/diskb/GlodonDataset/HeadDet/v0.2/tf_label_map.pbtxt',
                    'a txt file which is arranged as `id:class_name` each line')
    flags.DEFINE_integer('test_num', 0, "numbers of picture to test")
    flags.DEFINE_float('conf_thresh', 0.8, "confidence threshold")
    flags.DEFINE_string('Type', 'val', 'Which dataset to test,only[train,test]')

    flags.mark_flag_as_required('pb')

def decode_tf_label_file(tf_label_file_path):
  label_map = label_map_util.load_labelmap(tf_label_file_path)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1000, use_display_name=True)
  category_dic = {}
  for v in categories:
    category_dic[v['id']] = v['name']
  return category_dic


class PBInterpreter():
  def __init__(self, pb_file, label_file_path):
    self.pb_file_path = pb_file
    self.graph = self.load_graph(pb_file)
    self.input_tensor = self.get_input_tensor_name()
    self.output_tensor = self.get_output_tensor_name()
    self.id_category_map = self.get_labels(label_file_path)

    # self.graph.as_default()
    self.sess = tf.Session(graph=self.graph)

  def get_labels(self, label_file_path):
    if '.pbtxt' == label_file_path[-6:]:
      id_category_map = decode_tf_label_file(label_file_path)
    return id_category_map

  def load_graph(self, pb_file_path):
    self.graph = tf.Graph()
    with self.graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(pb_file_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return self.graph

  def preprocess(self, image, input_width, input_height):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_np = cv2.resize(image, (input_width, input_height))
    return image_np

  def get_input_tensor_name(self):
    # input tensor
    image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
    return {'image_tensor': image_tensor}

  def get_output_tensor_name(self):
    ops = self.graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 
                'detection_boxes', 
                'detection_scores',
                'detection_classes',
                'detection_masks']:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = self.graph.get_tensor_by_name(tensor_name)
    return tensor_dict

  def tf_inference_image(self, image_path, conf_thresh, input_width, input_height):
    # load image
    assert os.path.exists(image_path), image_path
    image_input = cv2.imread(image_path)
    image_id = os.path.basename(image_path).replace('.jpg', '')

    
    # preprocess
    image_np = self.preprocess(image_input, input_width, input_height)

    # inference
    start = time.time()
    output_dict = self.sess.run(self.output_tensor, feed_dict={self.input_tensor['image_tensor']: np.expand_dims(image_np, 0)})
    end = time.time()
    using_time = end - start

    results, image_drawed = self.postprocess(output_dict, conf_thresh, image_input.copy())

    return results

  def postprocess(self, output_dict, conf_thresh, image_input):
    # postprocess
    output_dict['num_detections'] = output_dict['num_detections'][0]
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    one_image_json_result = {k:[] for k, v in self.id_category_map.items()}
    for i in range(len(output_dict['detection_scores'])):
      category_id = output_dict['detection_classes'][i]
      if output_dict['detection_scores'][i] > conf_thresh:
        draw_arr =output_dict['detection_boxes'][i]*[image_input.shape[0], image_input.shape[1], image_input.shape[0], image_input.shape[1]]
        xmin_f = float(draw_arr[1])
        ymin_f = float(draw_arr[0])
        xmax_f = float(draw_arr[3])
        ymax_f = float(draw_arr[2])
        cv2.rectangle(image_input, (int(draw_arr[1]), int(draw_arr[0])), (int(draw_arr[3]), int(draw_arr[2])), (0,255,0), 1)

        score = float(output_dict['detection_scores'][i])
        one_image_json_result[category_id].append([xmin_f, ymin_f, xmax_f, ymax_f, score])

    return one_image_json_result, image_input

  def tf_inference_list(self, image_list, conf_thresh, input_width, input_height):
    assert isinstance(image_list, list)

    results_list = []
    for image_path in tqdm.tqdm(image_list):
      one_image_results = self.tf_inference_image(image_path, conf_thresh, input_width, input_height)
      results_list.append(one_image_results)
    return results_list

  def tf_inference_voc(self, main_path, conf_thresh, input_width, input_height):
    assert os.path.exists(main_path), main_path

    jpg_dir = os.path.join(os.path.dirname(main_path), '../../JPEGImages')
    with open(main_path) as f:
      names = f.read().strip().split('\n')
    image_list = [os.path.join(jpg_dir, name+'.jpg') for name in names]

    results_list = self.tf_inference_list(image_list, conf_thresh, input_width, input_height)
    return results_list


def main(_):
    FLAGS = flags.FLAGS
    print(FLAGS.main_set)
    width, height = list(map(int, FLAGS.size.split(',')))
    print(width, height)

    if FLAGS.image_path:
        image_path_list = [FLAGS.image_path]
    elif FLAGS.main_set:
        main_set = FLAGS.main_set

        with open(os.path.join(voc_root, 'ImageSets/Main', main_set+'.txt')) as f:
            names = f.read().strip().split('\n')
        image_path_list = [os.path.join(voc_root, 'JPEGImages', name+'.jpg') for name in names]
    else:
        print(FLAGS.image_path)
        print(FLAGS.main_set)
        raise


    interpreter = PBInterpreter(FLAGS.pb)
    results = interpreter.tf_inference_list(image_path_list, FLAGS.conf_thresh, width, height)


if __name__ == '__main__':
    get_args()
    app.run(main)

