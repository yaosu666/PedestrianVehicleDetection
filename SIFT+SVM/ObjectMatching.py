import numpy as np
import pandas as pd
import cv2
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import FeatureExtraction
import Preprocessing


def object_matching(img1, img2, method='sift', max_points=50):
    MIN_MATCH_COUNT = 10
    kp1, des1 = FeatureExtraction.feature_extraction(img1, method=method)
    kp2, des2 = FeatureExtraction.feature_extraction(img2, method=method)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    matches = sorted(matches, key=lambda x: x[1].distance)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    good = good[:min(max_points, len(good))]
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img3)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return img3


if __name__ == '__main__':
    root_dir = 'VisDrone2019-MOT/val'
    df = Preprocessing.load_description_dir(root_dir)
    boxes_dicts = []
    boxes = {k: [] for k in range(12)}
    for count, (_, df_frames) in tqdm(enumerate(df.groupby('filename')), total=df['filename'].nunique()):
        boxes_dict = Preprocessing.extract_box_img(df_frames)
        for k, v in boxes_dict.items():
            if len(boxes[k]) < 10:
                boxes[k].extend(v)
        lens = [len(item) for item in boxes.values()]
        if min(lens) > 0:
            break
    #object_matching(Preprocessing.load_image(df.loc[0, 'filename']),
    #                Preprocessing.load_image(df.loc[0, 'filename'])[:256, :256, :])
    img = object_matching(boxes[1][0],
                    Preprocessing.load_image(df.loc[0, 'filename']))
    cv2.imwrite("1.jpg", img)
    img = object_matching(boxes[1][2],
                    Preprocessing.load_image(df.loc[0, 'filename']))
    cv2.imwrite("2.jpg", img)
    img = object_matching(boxes[1][12],
                    Preprocessing.load_image(df.loc[0, 'filename']))
    cv2.imwrite("3.jpg", img)
    img = object_matching(boxes[1][23],
                    Preprocessing.load_image(df.loc[0, 'filename']))
    cv2.imwrite("4.jpg", img)


"""
ignored regions(0), pedestrian(1), 
people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), 
others(11)
"""
