import imageio
import cv2
import numpy as np
import os
import pickle
import re
import glob
import random

from PIL import Image
from scipy.misc.pilutil import imresize


def get_label_id_map(label_txt_path):
    """read a text file to get a label id dict
    Arguments:
        label_txt_path: str, the path of the label file.
    Returns:
        label_id_map: dic, a map dic about lable and id number.
    """
    with open(label_txt_path) as label_file:
        label_list = label_file.readlines()
        label_list = [line.rstrip().replace('\n', '') for line in label_list]

    label_id_map = {}
    for i in range(len(label_list)):
        label_id_map[label_list[i]] = i
    print(label_id_map)
    return label_id_map


def make_raw_dataset(video_list, categories, dataset="train"):
    data = []

    for category in categories:
        print(category)
        video_list = sorted(video_list)

        for filepath in video_list:
            print(filepath, end='\r')
            filename = os.path.basename(filepath)

            vid = imageio.get_reader(filepath, "ffmpeg")

            frames = []

            # Add each frame to correct list.
            for i, frame in enumerate(vid):
                # Convert to grayscale.
                frame = Image.fromarray(np.array(frame))
                frame = frame.resize((160, 120))
                frame = frame.convert("L")
                frame = np.array(frame.getdata(), dtype=np.uint8).reshape((120, 160))
                frame = imresize(frame, (60, 80))
                frames.append(frame)

            data.append({
                "filename": filename,
                "category": category,
                "frames": frames    
            })

    pickle.dump(data, open("data/%s.p" % dataset, "wb"))


def make_optflow_dataset(video_list, categories, dataset="train"):
    # Setup parameters for optical flow.
    farneback_params = dict(winsize=20, iterations=1,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
        pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)

    data = []

    for category in categories:
        print(category)
        # Get all files in current category's folder.
        folder_path = os.path.join("..", "dataset", category)
        filenames = sorted(os.listdir(folder_path))

        for filepath in video_list:
            filename = os.path.basename(filepath)
            # Get id of person in this video.
            person_id = int(filename.split("_")[0][6:])

            vid = imageio.get_reader(filepath, "ffmpeg")

            flow_x = []
            flow_y = []

            prev_frame = None
            # Add each frame to correct list.
            for i, frame in enumerate(vid):
                # Convert to grayscale.
                frame = Image.fromarray(np.array(frame))
                frame = frame.convert("L")
                frame = np.array(frame.getdata(), dtype=np.uint8).reshape((120, 160))
                frame = imresize(frame, (60, 80))

                if prev_frame is not None:
                    # Calculate optical flow.
                    flows = cv2.calcOpticalFlowFarneback(prev_frame, frame, **farneback_params)
                    subsampled_x = np.zeros((30, 40), dtype=np.float32)
                    subsampled_y = np.zeros((30, 40), dtype=np.float32)

                    for r in range(30):
                        for c in range(40):
                            subsampled_x[r, c] = flows[r*2, c*2, 0]
                            subsampled_y[r, c] = flows[r*2, c*2, 1]

                    flow_x.append(subsampled_x)
                    flow_y.append(subsampled_y)

                prev_frame = frame

            data.append({
                "filename": filename,
                "category": category,
                "flow_x": flow_x,
                "flow_y": flow_y    
            })

    pickle.dump(data, open("data/%s_flow.p" % dataset, "wb"))


if __name__ == "__main__":
    video_list = glob.glob('../dataset/*/*.avi')
    video_list += glob.glob('../dataset/*/*.mp4')
    random.shuffle(video_list)
    train_num = int(len(video_list) * 0.8)
    train_list = video_list[:train_num]
    val_list = video_list[train_num:]

    categories = get_label_id_map('/home/wangtf/ShareDataset/dataset/GAR/GAR_v4/label_name_gar.txt')
    print(categories)

    print("Making raw train dataset")
    make_raw_dataset(train_list, categories, dataset="train")
    print("Making raw dev dataset")
    make_raw_dataset(val_list, categories, dataset="dev")

    print("Making optical flow features for train dataset")
    make_optflow_dataset(train_list, categories, dataset="train")
    print("Making optical flow features for dev dataset")
    make_optflow_dataset(val_list, categories, dataset="dev")
