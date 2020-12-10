# encoding: utf-8
# author: Lidian Lin
# email: lldlld0574@gmail.com
# file: detection_video.py
# time: 2020/11/10 12:12
# desc: object detection in video
import cv2
import numpy as np
import logging
import sys
import os


def logger_init():
    hog_svm_logger = logging.getLogger("HOG+SVM Detection")

    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter

    hog_svm_logger.addHandler(console_handler)

    hog_svm_logger.setLevel(logging.INFO)

    return hog_svm_logger


def video_to_img(video_path, img_dir, interval):
    """
    extract video to frames
    :param video_path:
    :param img_dir:
    :param interval:
    :return:
    """
    logger.info('converting video to images')
    vc = cv2.VideoCapture(video_path)
    c = 0
    if vc.isOpened():
        ret, frame = vc.read()
    else:
        print('open error!')
        ret = False

    while ret:
        if c % interval == 0:
            file_path = img_dir + '%04d.jpg' % c
            logger.info("write frame to ", file_path)
            cv2.imwrite(file_path, frame)
        ret, frame = vc.read()
        c += 1
    vc.release()
    logger.info('video converting finished')
    return


def img_to_video(img_dir, video_path, interval, hog, hog_detect=False, video_size=(4096, 2160)):
    """
    compress frames to video
    :param img_dir:
    :param video_path:
    :param interval:
    :param hog:
    :param hog_detect:
    :param video_size:
    :return:
    """
    logger.info('start transforming images to video with hog_detect=%s' % hog_detect)
    vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 24, video_size)
    frame_files = os.listdir(img_dir)
    for file in frame_files:
        img_no = int(file[0:3])
        logger.info('now come to frame no.%03d' % img_no)
        img = cv2.imread(img_dir + file)
        img = cv2.resize(img, video_size)
        if hog_detect:
            rects, _ = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
            if rects is not None:
                for x, y, w, h in rects:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        a = 0
        while a < interval:
            vw.write(img)
            a += 1

    vw.release()
    logger.info('video writing finished')
    return


def detect_pedestrian(input_dir, output_dir, hog, img_show=False):
    """

    :param input_dir:
    :param output_dir:
    :param hog:
    :param img_show:
    :return:
    """
    input_list = os.listdir(input_dir)
    if img_show:
        cv2.namedWindow('Detect')
    for f in input_list:
        file_path = os.path.join(input_dir, f)
        logger.info('Processing {}'.format(file_path))
        img = cv2.imread(file_path)
        rects, _ = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.1)
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        if img_show:
            sp = img.shape
            width = sp[1]
            height = sp[0]
            ratio = width / height
            new_width = 1600
            new_height = int(new_width / ratio)
            show_img = cv2.resize(img, (new_width, new_height))
            cv2.imshow('Detect', show_img)
            c = cv2.waitKey(0) & 0xff
            if c == 27:
                break
        cv2.imwrite(output_dir+f, img)
    cv2.destroyAllWindows()


def get_svm_detector(svm_name):
    """
    load svm
    :param svm_name:
    :return:
    """
    logger.info('SVM parameter file %s is loaded' % svm_name)
    svm = cv2.ml.SVM_load(svm_name)
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


def pipeline(video2img, categories):
    """

    :param video2img: True or False
    :param categories: pedestrian or vehicle or both|['pedestrian','vehicle']
    :return:
    """
    # convert videos to frames first
    video_files = os.listdir('./TestVideo/')
    # video_files = ['uav0000342_04692_v']
    if video2img:
        for file_name_ in video_files:
            file_name = file_name_.split('.')[0]
            input_video_path = './TestVideo/%s' % file_name_
            video_frame_dir = './VideoFrame/%s/' % file_name
            if not os.path.exists(video_frame_dir):
                os.makedirs(video_frame_dir)
            frame_interval = 4
            video_to_img(input_video_path, video_frame_dir, frame_interval)
    video_frame_dirs = os.listdir('./VideoFrame/')
    for frame_dir in video_frame_dirs:
        video_frame_dir = './VideoFrame/%s/' % frame_dir
        for category in categories:
            video_frame_after_dir = './VideoFrameAfter/%s/%s/' % (frame_dir, category)
            if not os.path.exists(video_frame_after_dir):
                os.makedirs(video_frame_after_dir)
            output_video_path = './VideoAfter/%s_%s.avi' % (frame_dir, category)
            frame_interval = 4
            svm_file = './svm/svm-%s.xml' % category
            if category == 'pedestrian':
                hog_descriptor = cv2.HOGDescriptor()
                hog_descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                # hog_descriptor.setSVMDetector(get_svm_detector(svm_file))
            else:
                hog_descriptor = cv2.HOGDescriptor()
                hog_descriptor.setSVMDetector(get_svm_detector(svm_file))
            detect_pedestrian(video_frame_dir, video_frame_after_dir, hog_descriptor, False)
            img_to_video(video_frame_after_dir, output_video_path, frame_interval, None, False)


if __name__ == '__main__':
    logger = logger_init()
    pipeline(True, ['pedestrian', 'vehicle'])

