import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
import random
import h5py

import multiprocessing

import FeatureExtraction

#car, bus, truck, pedestrian, and van.

def load_image(path):
    img = cv2.imread(path)
    return img


def image_resize(img, size=(800, 800)):
    size_img = cv2.resize(img, size)
    return size_img


def load_description(csv_path):
    assert os.path.exists(csv_path)
    csv_path = Path(csv_path)
    assert os.path.isfile(csv_path)
    root = csv_path.parent.parent
    vclip_name = os.path.basename(csv_path).split('.')[0]
    desc_df = pd.read_csv(csv_path, names=['frame_index', 'target_id',
                                           'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height',
                                           'score', 'object_category', 'truncation', 'occlusion'])

    def filename(row):
        return os.path.join(root, 'sequences', vclip_name, f"{row:07d}.jpg")

    desc_df['filename'] = desc_df['frame_index'].apply(filename)
    return desc_df


def load_description_dir(file_dir):
    assert os.path.exists(file_dir)
    assert os.path.isdir(file_dir)
    csv_dir = os.path.join(file_dir, 'annotations')
    dfs = []
    for file in tqdm(os.listdir(csv_dir)):
        file = os.path.join(csv_dir, file)
        if not os.path.isfile(file):
            continue
        else:
            this_df = load_description(file)
            this_df['clip_name'] = os.path.basename(file).split('.')[0]
            dfs.append(load_description(file))
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)
    return df


def extract_box_img(df_frames):
    assert df_frames['filename'].nunique() == 1
    df_frames = df_frames.reset_index(drop=True)
    img = cv2.imread(df_frames.loc[0, 'filename'])
    h, w = img.shape[:2]
    boxes_dict = {k: [] for k in range(12)}
    for i_row, row in df_frames.iterrows():
        if row['bbox_height'] < 24 or row['bbox_height'] < 24:
            continue
        box = img[row['bbox_top']:row['bbox_top'] + row['bbox_height'],
                  row['bbox_left']:row['bbox_left'] + row['bbox_width'],
                  :]
        boxes_dict[row['object_category']].append(box)
    # neg sampling
    for i in range(64):
        bbox_top, bbox_left = random.randint(0, h - 1), random.randint(0, w - 1)
        scale = random.sample([(1, 2), (2, 1), (1, 1)], 1)[0]
        size = random.sample([16, 24, 32, 40, 52, 64, 128], 1)[0]
        bbox_height, bbox_width = (scale[0] * size), (scale[1] * size)
        box = img[bbox_top:bbox_top + bbox_height,
                  bbox_left:bbox_left + bbox_width,
                  :]
        boxes_dict[0].append(box)
    return boxes_dict





if __name__ == '__main__':
    root_dir = 'VisDrone2019-MOT/val'
    df = load_description_dir(root_dir)

#cv2.data.haarcascades