#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2
import sys

class VideoEditor(object):
    def __init__(self, video_path):
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)

        self.fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_size = (0, 0) # (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.save_name = os.path.basename(self.video_path).split('.')[0]
        # self.writer = cv2.VideoWriter('{}.mp4'.format(self.save_name), cv2.CAP_ANY, self.fourcc, self.fps, self.frame_size)

    def rotate(self):
        while self.capture.isOpened():
            ok, frame = self.capture.read()
            if not ok:
                break
            frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if self.frame_size[0] != frame_rotated.shape[1] or self.frame_size[1] != frame_rotated.shape[0]:
                self.frame_size = (frame_rotated.shape[1], frame_rotated.shape[0])
                self.writer = cv2.VideoWriter('{}_rotated.mp4'.format(self.save_name), cv2.CAP_ANY, self.fourcc, self.fps, self.frame_size)
            self.writer.write(frame_rotated)

    def play(self):
        is_write = False
        index = 0
        while self.capture.isOpened():
            ok, frame = self.capture.read()
            index += 1
            if not ok:
                break
            cv2.imshow('frame', frame)
            if self.frame_size[0] != frame.shape[1] or self.frame_size[1] != frame.shape[0]:
                self.frame_size = (frame.shape[1], frame.shape[0])
                self.writer = cv2.VideoWriter('{}_clip.avi'.format(self.save_name), cv2.CAP_ANY, self.fourcc, self.fps, self.frame_size)
            control = cv2.waitKey()
            if control == 0xff & ord('s'):
                is_write = True
            elif control == 0xff & ord('e'):
                is_write = False
            if is_write:
                self.writer.write(frame)
                print('save index: {}'.format(index))
        self.writer.release()


    def resize(self):
        index = 0
        while self.capture.isOpened():
            ok, frame = self.capture.read()
            index += 1
            if not ok:
                break
            
            #frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            frame = cv2.resize(frame, (160, 120))
            if self.frame_size[0] != frame.shape[1] or self.frame_size[1] != frame.shape[0]:
                self.frame_size = (frame.shape[1], frame.shape[0])
                save_name = self.save_name +  '_resized.mp4'
                print(save_name)
                self.writer = cv2.VideoWriter(save_name, cv2.CAP_ANY, self.fourcc, self.fps, self.frame_size)
            self.writer.write(frame)
        self.writer.release()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = '/home/wangtf/ShareDataset/dataset/videos/IMG_2661.MOV'
    editor = VideoEditor(video_path)
    #editor.rotate()
    #editor.play()
    editor.resize()


