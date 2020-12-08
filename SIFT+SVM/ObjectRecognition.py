import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from functools import partial
from tqdm.auto import tqdm
import multiprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.cluster import MeanShift
from sklearn.preprocessing import OneHotEncoder

import FeatureExtraction
import Preprocessing
import BOW
import pickle

"""
   0: other
   1: people
   2: bicycle
   3: car
   4: van
   5: truck
   6: tricycle
   7: bus
   8: motor
   """

class_mapping = {0: 0,
                     1: 1,
                     2: 1,
                     3: 2,
                     4: 3,
                     5: 4,
                     6: 5,
                     7: 6,
                     8: 6,
                     9: 7,
                     10: 8,
                     11: 0,}
# w, h
test_ratio_dict = {
    0: [(1, 1), (3, 2), (2, 1)],
    1: [(1, 2)],
    2: [(1, 2), (2, 3)],
    3: [(2, 3), (1, 1), (4, 3)],
    4: [(4, 3), (3, 2), (5, 3)],
    5: [(1, 1), (4, 3), (3, 2)],
    6: [(1, 1), (3, 2), (2, 1)],
    7: [(1, 1)],
    8: [(2, 3), (1, 1), (4, 3)],
    9: [(1, 1), (3, 2), (2, 1)],
    10: [(2, 3), (1, 1)],
    11: [(5, 2)]
}


def process_df_frames_train_set(item, method='surf', histogram_norm='l1'):
    import FeatureExtraction
    import random
    filename, df_frames = item
    if df_frames['frame_index'].iloc[0] % 100 != 24:
        return []
    boxes_dict = Preprocessing.extract_box_img(df_frames)

    train_set = []

    kmeans = BOW.load_kmeans(f'kmeans_{method}_128.pkl')

    for category, boxes in boxes_dict.items():
        for box in boxes:
            _, des = FeatureExtraction.feature_extraction(box, method)
            if des is not None:
                histogram = BOW.compute_histogram(des, kmeans, method=method, norm=histogram_norm)
                train_set.append((histogram, category))


    return train_set


def build_dataset(df, method='surf', histogram_norm='l0'):
    train_set = []
    f = partial(process_df_frames_train_set, method=method, histogram_norm=histogram_norm)

    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    #for item in tqdm(pool.imap_unordered(f, df.groupby('filename')), total=df['filename'].nunique()):
    #for item in tqdm(map(f, df.groupby('filename')), total=df['filename'].nunique()):
    for item in tqdm(df.groupby('filename'), total=df['filename'].nunique()):
        train_set.extend(f(item))
    return train_set


def build_dataset_and_save(root_dir='VisDrone2019-MOT/val', method='surf'):
    df = Preprocessing.load_description_dir(root_dir)
    train_set = build_dataset(df, method=method)
    pickle.dump(train_set, open(f'train_set_{method}.pkl', 'wb'))


def train_clf(train_set_path='train_set_sift.pkl'):
    """
       0: other
       1: people
       2: bicycle
       3: car
       4: van
       5: truck
       6: tricycle
       7: bus
       8: motor
       """

    train_set = pickle.load(open(train_set_path, 'rb'))
    print()
    X, y = zip(*train_set)
    y = list(map(lambda x: class_mapping[x], y))
    X, y = np.array(X), np.array(y)
    clf = make_pipeline(StandardScaler(),
                        SVC(C=5, kernel='poly', degree=5, class_weight='balanced', probability=True),
                        #LogisticRegressionCV(class_weight='balanced', multi_class='ovr')
                        )

    clf.fit(X, y)
    y_hat = clf.predict(X)
    print(f1_score(y, y_hat, average='macro'))
    print(confusion_matrix(y, y_hat))
    return clf, y, y_hat


def load_clfs(path='clfs.pkl'):
    clfs = pickle.load(open(path, 'rb'))
    return clfs


def get_box_ratio(df):
    df = df.copy()
    df['bbox_ratio'] = df['bbox_width'] / df['bbox_height']
    ratio_dict = {k: [] for k in range(12)}
    for category, df_cat in df.groupby('object_category'):
        ratio_dict[category].extend(df_cat['bbox_ratio'].tolist())
    percentile_dict = {k: [] for k in range(12)}
    median_dict = {k: 0 for k in range(12)}
    for k, v in ratio_dict.items():
        percentile_dict[k].append(np.percentile(v, 25))
        percentile_dict[k].append(np.percentile(v, 75))
        median_dict[k] = np.median(v)
    return ratio_dict, percentile_dict, median_dict


def clfs_rolling_windows(img, clf, stride=8, padding=0, ratios=((1, 1), (3, 1), (1, 3)), scale=1.5,
                         prob_threshold=0.75, min_side_length=16, method='sift', kmeans_path=None):
    if method not in ('sift', 'surf'):
        raise ValueError("method param should be sift or surf")
    if kmeans_path is None:
        kmeans_path = f'kmeans_{method}_128.pkl'
    h, w = img.shape[:2]
    padded_image = np.zeros((h + padding * 2, w + padding * 2, img.shape[2]), dtype='uint8')
    padded_image[padding:h + padding, padding:w + padding, :] = img
    h, w = padded_image.shape[:2]

    box_hist_dict = {'box': [],
                     'histogram': [],
                     'high_prob': [],
                     'high_category': []}

    kmeans = BOW.load_kmeans(kmeans_path)
    kp, des = FeatureExtraction.feature_extraction(img, method)
    df_kp = pd.DataFrame({'x': [item.pt[0] for item in kp],
                          'y': [item.pt[1] for item in kp]})
    for ratio in ratios:
        print(f'ratio {ratio}')
        ratio_h, ratio_w = ratio
        if ratio_h < ratio_w:
            # (h, w)
            start_window = (min_side_length, int(min_side_length * ratio_w / ratio_h))
        else:
            start_window = (int(min_side_length * ratio_h / ratio_w), min_side_length)
        window = start_window
        this_stride = stride
        tmp = window
        count = 0
        while tmp[0] <= h and tmp[1] <= w:
            count += 1
            tmp = (int(tmp[0] * scale), int(tmp[1] * scale))
        pbar = tqdm(total=count)
        while window[0] <= h and window[1] <= w:
            for i in range(0, h - window[0], this_stride):
                for j in range(0, w - window[1], this_stride):
                    #print(i, j)
                    box = (i, j, i + window[0], j + window[1])
                    #img_this = padded_image[box[0]:box[2], box[1]:box[3]]
                    #_, des_this = FeatureExtraction.feature_extraction(img_this, method)

                    des_this = des[np.logical_and.reduce((box[0] < df_kp['y'],
                                                          df_kp['y'] < box[2],
                                                          box[1] < df_kp['x'],
                                                          df_kp['x'] < box[3]))]
                    if des_this is None or len(des_this) == 0:
                        continue
                    histogram = BOW.compute_histogram(des_this, kmeans, method=method, norm='l0')
                    if histogram is None:
                        continue
                    #print(clf.predict(histogram.reshape(1, -1)))

                    """cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('image', img_this)
                    cv2.waitKey()
                    cv2.destroyAllWindows()"""
                    box_hist_dict['box'].append(box)
                    box_hist_dict['histogram'].append(histogram)
            window = (int(window[0] * scale), int(window[1] * scale))
            this_stride = int(this_stride * np.sqrt(scale))
            pbar.update(1)
        pbar.close()
    X = np.array(box_hist_dict['histogram'])
    y_prob = clf.predict_proba(X)
    high_prob = np.max(y_prob, axis=1).ravel().tolist()
    high_category = np.argmax(y_prob, axis=1).ravel().tolist()
    box_hist_dict['high_prob'] = high_prob
    box_hist_dict['high_category'] = high_category
    df_box_hist = pd.DataFrame(box_hist_dict)
    df_box_hist = df_box_hist[df_box_hist['high_category'] != 0]
    df_box_hist = df_box_hist[df_box_hist['high_prob'] > prob_threshold]
    return df_box_hist


def object_localization(img, clf, **kwargs):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 122, 0),
              (122, 255, 0), (0, 122, 255), (255, 0, 122), (122, 122, 122)]

    df_box = clfs_rolling_windows(img, clf, **kwargs)
    if df_box.shape[0] == 0:
        return img, [], []
    boxes = np.array(df_box['box'].to_list())
    labels = np.array(df_box['high_category'])
    labels_onehot = np.array([[1 if item == j + 1 else 0 for j in range(8)] for item in labels])
    boxes_std = (boxes - boxes.min(axis=0)) / (boxes.max(axis=0) - boxes.min(axis=0) + 1)
    clustering_X = np.concatenate([boxes_std, labels_onehot], axis=1)
    print('clustering')
    meanshift = MeanShift(n_jobs=-1).fit(clustering_X)
    centers = meanshift.cluster_centers_
    final_boxes = centers[:, :4]
    final_boxes = final_boxes * (boxes.max(axis=0) - boxes.min(axis=0)) + boxes.min(axis=0)
    final_boxes = final_boxes.astype('int')
    final_labels = centers[:, 4:].argmax(axis=1) + 1
    plt.imshow(img)
    plt.show()
    for i, box in enumerate(final_boxes):
        img = cv2.rectangle(img,
                            (box[1], box[0]),
                            (box[3], box[2]),
                            colors[final_labels[i]], 2)
    plt.imshow(img)
    plt.show()
    return img, final_boxes, final_labels


def convert_video(input_path, output_path, clf):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 24.0, (640, 480))
    n = 0
    while (cap.isOpened()):
        print(f'frame {n}')
        n += 1
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (640, 480))
            img_boxed, _, _ = object_localization(frame, clf)
            # write the flipped frame
            out.write(img_boxed)
            break
        else:
            break



if __name__ == '__main__':
    root_dir = 'VisDrone2019-MOT/train'
    method = 'surf'
    norm = 'l0'

    """df = Preprocessing.load_description_dir(root_dir)
    for method_ in ['sift', 'surf']:
        print(f'preparing training set {method_}')
        train_set = build_dataset(df, method=method_, histogram_norm='l0')
        pickle.dump(train_set, open(f'train_set_{method_}.pkl', 'wb'))"""
    #print('training sift')
    #clf, y, y_hat = train_clf('train_set_sift.pkl')
    #pickle.dump(clf, open('clfs_sift.pkl', 'wb'))
    """
    print('training surf')
    clf = train_clf('train_set_surf.pkl')
    pickle.dump(clf, open('clfs_surf.pkl', 'wb'))"""
    """train_set = pickle.load(open('train_set_sift.pkl', 'rb'))
    X, y = zip(*train_set)
    y = list(map(lambda x: class_mapping[x], y))
    X, y = np.array(X), np.array(y)"""
    clf = load_clfs('clfs_sift.pkl')
    """y_hat = clf.predict(X)
    print(f1_score(y, y_hat, average='macro'))
    print(confusion_matrix(y, y_hat))"""
    """print(clf.classes_)
    img = Preprocessing.load_image('VisDrone2019-MOT/train/sequences/uav0000099_02109_v/0000001.jpg')
    plt.imshow(img)
    plt.show()
    img_boxed = object_localization(img, clf)
    plt.imshow(img_boxed[0])
    plt.show()"""
    #convert_video('wpi_aerial/DJI_0674.MOV', 'wpi_aerial/DJI_0674_boxed_1.avi', clf)

