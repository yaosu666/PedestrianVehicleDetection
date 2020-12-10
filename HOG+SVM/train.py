# encoding: utf-8
# author: Lidian Lin
# email: lldlld0574@gmail.com
# file: train.py
# time: 2020/11/10 12:12
# desc: hog+svm training
import os
import sys
import cv2
import logging
import numpy as np


def logger_init():
    hog_svm_logger = logging.getLogger("HOG+SVM Detection Training")

    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter

    hog_svm_logger.addHandler(console_handler)

    hog_svm_logger.setLevel(logging.INFO)

    return hog_svm_logger


def load_data_set(pos_dir, neg_dir, test_dir, logger):
    """
    wrap path
    :param pos_dir:
    :param neg_dir:
    :param test_dir:
    :param logger:
    :return:
    """
    if os.path.exists(pos_dir):
        positive = os.listdir(pos_dir)
        logger.info('Positive data path is: %s' % pos_dir)
        logger.info('Positive samples number: %s' % len(positive))

    if os.path.exists(neg_dir):
        negative = os.listdir(neg_dir)
        logger.info('Negative data path is: %s' % neg_dir)
        logger.info('Negative samples number: %s' % len(negative))

    if os.path.exists(test_dir):
        testing = os.listdir(test_dir)
        logger.info('Test data path is: %s' % test_dir)
        logger.info('Test samples number: %s' % len(testing))

    return positive, negative, testing


def load_train_samples(pos_dir, neg_dir, positive, negative, logger):
    """
    give positive label 1 and negative label -1, sample and label add simultaneously
    :param pos_dir:
    :param neg_dir:
    :param positive:
    :param negative:
    :param logger:
    :return samples: wrapped
    :return labels:
    """

    sample = []
    label = []
    for p in positive:
        file_path = pos_dir + '/' + p
        if os.path.exists(file_path):
            sample.append(file_path)
            label.append(1.)

    for n in negative:
        file_path = neg_dir + '/' + n
        if os.path.exists(file_path):
            sample.append(file_path)
            label.append(-1.)

    # convert from list to numpy array
    label = np.int32(label)
    labels_len = len(positive) + len(negative)
    label = np.resize(label, (labels_len, 1))

    logger.info('Samples and labels are prepared')
    return sample, label


def extract_hog(sample, window_size, logger):
    """
    extract hog feature
    :param sample:
    :param window_size:
    :param logger:
    :return train:
    """
    training_data = []
    logger.info('Extracting HOG Descriptors...')
    count = 0.
    total = len(sample)
    for f in sample:
        count += 1.
        logger.info('Processing {} {:2.1f}%'.format(f, count/total*100))
        hog = cv2.HOGDescriptor(window_size, (16, 16), (8, 8), (8, 8), 9)
        img = cv2.imread(f, -1)
        img = cv2.resize(img, window_size)
        descriptors = hog.compute(img)
        logger.info('hog feature descriptor size: {}'.format(descriptors.shape))  # (3780, 1)
        training_data.append(descriptors)

    # convert from list to numpy array
    training_data = np.float32(training_data)
    training_data = np.resize(training_data, (total, 3780))

    return training_data


def get_svm_detector(svm):
    """
    load svm detector
    :param svm:
    :return:
    """
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


def train_svm(training_data, label, svm_name, logger):
    """
    train an svm
    :param training_data:
    :param label:
    :param svm_name:
    :param logger:
    :return:
    """
    logger.info('Configuring SVM classifier.')
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)  # From paper, soft classifier
    svm.setType(cv2.ml.SVM_EPS_SVR)

    logger.info('Starting training the SVM')
    svm.train(training_data, cv2.ml.ROW_SAMPLE, label)
    logger.info('Training finished.')

    svm.save(svm_name)
    logger.info('Trained SVM classifier is saved as: {}'.format(svm_name))

    return get_svm_detector(svm)


def load_svm(svm_name, logger):
    """
    cv2 ml SVM_LOAD
    :param svm_name:
    :param logger:
    :return:
    """
    logger.info('%s is loaded' % svm_name)
    return cv2.ml.SVM_load(svm_name)


def test_hog_detect(test_dir, testing, svm_name, logger, use_svm=True):
    """
    testing on images
    :param test_dir:
    :param testing:
    :param svm_name:
    :param logger:
    :return:
    """
    hog = cv2.HOGDescriptor()
    if use_svm:
        hog.setSVMDetector(get_svm_detector(load_svm(svm_name, logger)))
    else:
        # opencv default people detector
        # only for people
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cv2.namedWindow('Detect Test')
    for f in testing:
        file_path = os.path.join(test_dir, f)
        logger.info('Processing {}'.format(file_path))
        img = cv2.imread(file_path)
        rects, _ = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.1)
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow('Detect', img)
        c = cv2.waitKey(0) & 0xff
        if c == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cv_logger = logger_init()
    # might need to change directory to use pedestrian or vehicle positive data
    positive_dir = './Positive'
    negative_dir = './Negative'
    test_data_dir = './TestData'
    pos, neg, test = load_data_set(positive_dir, negative_dir, test_data_dir, cv_logger)
    samples, labels = load_train_samples(positive_dir, negative_dir, pos, neg, cv_logger)
    train = extract_hog(samples, (64, 128), cv_logger)
    # based on pedestrian or vehicle data
    svm_file = './svm/svm-pedestrian.xml'
    svm_detector = train_svm(train, labels, svm_file, logger=cv_logger)
    test_hog_detect(test, svm_detector, svm_file, cv_logger)

