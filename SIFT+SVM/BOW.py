import numpy as np
import pandas as pd
from functools import partial
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
import pickle

import Preprocessing
import FeatureExtraction


def process_df_frames_des(item, method='sift'):
    import FeatureExtraction
    filename, df_frames = item
    if df_frames['frame_index'].iloc[0] % 50 != 0:
        return [], []
    img = Preprocessing.load_image(filename)
    kps, dess = FeatureExtraction.feature_extraction(img, method)
    return kps, dess


def extract_descriptors(df, method='sift'):
    kps, dess = [], []
    f = partial(process_df_frames_des, method=method)
    for kp, des in tqdm(map(f, df.groupby('filename')), total=df['filename'].nunique()):
        kps.extend(kp)
        dess.extend(des)
    return kps, dess


def load_kmeans(path='kmeans_surf.pkl'):
    kmeans = pickle.load(open(path, 'rb'))
    return kmeans


def compute_histogram(dess, kmeans, method, norm='l1'):
    if method == 'sift':
        vec_length = 128
    elif method == 'surf':
        vec_length = 64
    else:
        raise ValueError("method param should be sift or surf")
    des = np.array(dess).reshape(-1, vec_length).astype(np.float64)
    labels = kmeans.predict(des)
    histogram = [0 for _ in range(128)]
    for label in labels:
        histogram[label] += 1
    if norm == 'l1':
        histogram = np.array(histogram) / sum(histogram)
    elif norm == 'l0':
        histogram = (np.array(histogram) > 0).astype(np.float64)
    return histogram


def compute_histogram_from_img(img, kmeans, method, norm='l1'):
    if method == 'sift':
        vec_length = 128
    elif method == 'surf':
        vec_length = 64
    else:
        raise ValueError("method param should be sift or surf")
    kps, dess = FeatureExtraction.feature_extraction(img, method)
    if dess is None:
        return None
    des = np.array(dess).reshape(-1, vec_length).astype(np.float64)
    labels = kmeans.predict(des)
    histogram = [0 for _ in range(128)]
    for label in labels:
        histogram[label] += 1
    if norm == 'l1':
        histogram = np.array(histogram) / sum(histogram)
    elif norm == 'l0':
        histogram = (np.array(histogram) > 0).astype(np.float64)
    return histogram



if __name__ == '__main__':
    root_dir = 'VisDrone2019-MOT/val'
    method = 'sift'
    n_clusters = 128
    _, dess = extract_descriptors(Preprocessing.load_description_dir(root_dir), method=method)
    kmeans = KMeans(n_clusters=n_clusters, verbose=1)
    kmeans.fit(dess)
    pickle.dump(kmeans, open(f'kmeans_{method}_{n_clusters}.pkl', 'wb'))
    kmeans_load = load_kmeans()
