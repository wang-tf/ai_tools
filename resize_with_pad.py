#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
BLACK = [0, 0, 0]


def resize(image, height_org=299, width_org=299, keep_min=True):
    shape = list(image.shape)
    height_input, width_input = shape[0], shape[1]

    # if original image is smaller than output size, and don't want to keep the original small size
    if not keep_min and height_input < height_org and width_input < width_org:
        rate = min(height_org / height_input, width_input / width_org)
        image = cv2.resize(image, None, fx=rate, fy=rate)
        shape = list(image.shape)
        height_input, width_input = shape[0], shape[1]
 
    # let mask keep the aspect ratio with output image size
    mask_width, mask_height = width_org, height_org
    if height_input > mask_height:
        rate = (height_input / mask_height)
        mask_height = height_input
        mask_width *= rate
    
    if width_input > mask_width:
        rate = (width_input / mask_width)
        mask_width = width_input
        mask_height *= rate

    dif_h = int(mask_height) - height_input
    left = dif_h // 2
    right = dif_h - left

    dif_w = int(mask_width) - width_input
    top = dif_h // 2
    bottom = dif_h - top

    # pad to mask image size
    print(top, bottom, left, right)
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    # resize to output size
    resized_image = cv2.resize(constant, (height_org, width_org))

    return resized_image

