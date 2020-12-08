import cv2
import numpy as np


def sift_feature(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def surf_feature(img):
    surf = cv2.xfeatures2d.SURF_create()
    kq, des = surf.detectAndCompute(img, None)
    return kq, des


def feature_extraction(img, method):
    if method == 'sift':
        kp, des = sift_feature(img)
    elif method == 'surf':
        # TODO:bug here
        kp, des = surf_feature(img)
    else:
        raise ValueError("method param should be sift or surf")
    return kp, des


def is_point_in_box(point, box_topleft, box_bottomright):
    return np.logical_and(np.logical_and(box_topleft[0] < point[0],  point[0] < box_bottomright[0]),
                          np.logical_and(box_topleft[1] < point[1],  point[1] < box_bottomright[1]))


if __name__ == '__main__':
    from Preprocessing import load_image, image_resize
    img = load_image('VisDrone2019-MOT/train/sequences/uav0000013_00000_v/0000001.jpg')
    kp, des = sift_feature(img)