import numpy as np
import cv2
import os

from functools import lru_cache
from scipy.stats import multivariate_normal


@lru_cache()
def get_gauss_pdf(sigma):
    n = sigma * 8

    x, y = np.mgrid[0:n, 0:n]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    rv = multivariate_normal([n / 2, n / 2], [[sigma ** 2, 0], [0, sigma ** 2]])
    pdf = rv.pdf(pos)

    heatmap = pdf / np.max(pdf)

    return heatmap


def to_int(num):
    return int(round(num))


@lru_cache()
def get_binary_mask(diameter):
    d = diameter
    _map = np.zeros((d, d), dtype = np.float32)

    r = d / 2
    s = int(d / 2)

    y, x = np.ogrid[-s:d - s, -s:d - s]
    mask = x * x + y * y <= r * r

    _map[mask] = 1.0

    return _map


def get_binary_heat_map(shape, is_present, centers, diameter = 9):
    n = diameter
    r = int(n / 2)
    hn = int(2 * n)
    qn = int(4 * n)
    pl = np.zeros((shape[0], shape[1] + qn, shape[2] + qn, shape[3]), dtype = np.float32)

    for i in range(shape[0]):
        for j in range(shape[3]):
            my = centers[i, 0, j] - r
            mx = centers[i, 1, j] - r

            if -n < my < shape[1] and -n < mx < shape[2] and is_present[i, j]:
                pl[i, my + hn:my + 3 * n, mx + hn:mx + 3 * n, j] = get_binary_mask(diameter)

    return pl[:, hn:-hn, hn:-hn, :]


def get_gauss_heat_map(shape, is_present, mean, sigma = 5):
    n = to_int(sigma * 8)
    hn = to_int(n / 2)
    dn = int(2 * n)
    qn = int(4 * n)
    pl = np.zeros((shape[0], shape[1] + qn, shape[2] + qn, shape[3]), dtype = np.float32)

    for i in range(shape[0]):
        for j in range(shape[3]):
            my = mean[i, 0, j] - hn
            mx = mean[i, 1, j] - hn

            if -n < my < shape[1] and -n < mx < shape[2] and is_present[i, j]:
                pl[i, my + dn:my + 3 * n, mx + dn:mx + 3 * n, j] = get_gauss_pdf(sigma)
                # else:
                #     print(my, mx)

    return pl[:, dn:-dn, dn:-dn, :]


def resize_with_pad(image, height=299, width=299):
    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image


def read_video(video, resize=True, rotate=0, image_dir=None):
    video_frames = []
    video = cv2.VideoCapture(video)

    ret, frame = video.read()
    i = 1
    while ret:
        if resize:
            frame = resize_with_pad(frame)
        frame = np.rot90(frame, k=rotate)
        # if image_dir is not None:
        #     im_name = '000' + str(i)
        #     im_name = im_name[-4:] + '.jpg'
        #     im_save_path = os.path.join(image_dir, im_name)
        #     cv2.imwrite(im_save_path, frame)
        video_frames.append(frame)
        ret, frame = video.read()
        i += 1
    return video_frames


def read_images(image_dir):
    video_frames = []
    im_list = os.listdir(image_dir)
    im_list.sort()
    for im_name in im_list:
        im_path = os.path.join(image_dir, im_name)
        img = cv2.imread(im_path)
        img = resize_with_pad(img)
        video_frames.append(img)
    return video_frames
