# encoding: utf-8
# author: Lidian Lin
# email: lldlld0574@gmail.com
# file: image_preprocessing.py
# time: 2020/11/20 15:31
# desc: resize images
import cv2
import os


def resize(img):
    """
    img add border replicate according to the img size and resize to (64, 128)
    :param img: img needed resizing
    :return: img after resized
    """
    sp = img.shape
    h = sp[0]
    w = sp[1]
    ratio = w / h
    # print(sp)
    if ratio < 0.5:
        if h < 64:
            new_height = 64
            new_width = int(64*ratio/2)*2
            img = cv2.resize(img, (new_width, new_height))
            img = cv2.copyMakeBorder(img, 32, 32, int(32-new_width/2), int(32-new_width/2), cv2.BORDER_REPLICATE)
        elif h < 128:
            new_height = int(h/2)*2
            new_width = int(w/2)*2
            img = cv2.resize(img, (new_width, new_height))
            img = cv2.copyMakeBorder(img, int(64-new_height/2), int(64-new_height/2), int(32-new_width/2), int(32-new_width/2), cv2.BORDER_REPLICATE)
        else:
            new_height = 128
            new_width = int(new_height*ratio/2)*2
            img = cv2.resize(img, (new_width, new_height))
            img = cv2.copyMakeBorder(img, 0, 0, int(32 - new_width / 2), int(32 - new_width / 2), cv2.BORDER_REPLICATE)
    else:
        if w < 32:
            new_width = 32
            new_height = int(32/ratio/2)*2
            img = cv2.resize(img, (new_width, new_height))
            img = cv2.copyMakeBorder(img, int(64-new_height/2), int(64-new_height/2), 16, 16, cv2.BORDER_REPLICATE)
        elif w < 64:
            new_height = int(h / 2) * 2
            new_width = int(w / 2) * 2
            img = cv2.resize(img, (new_width, new_height))
            img = cv2.copyMakeBorder(img, int(64 - new_height / 2), int(64 - new_height / 2), int(32 - new_width / 2),
                                     int(32 - new_width / 2), cv2.BORDER_REPLICATE)
        else:
            new_width = 64
            new_height = int(new_width/ratio/2)*2
            img = cv2.resize(img, (new_width, new_height))
            img = cv2.copyMakeBorder(img, int(64 - new_height / 2), int(64 - new_height / 2), 0, 0, cv2.BORDER_REPLICATE)
    return img


def resize_v2(img):
    """
    img add border constant and resize
    :param img:
    :return:
    """
    sp = img.shape
    h = sp[0]
    w = sp[1]
    ratio = w / h
    if ratio < 0.5:
        new_width = int(h*ratio/2)*2
        new_height = 128
        img = cv2.resize(img, (new_width, new_height))
        img = cv2.copyMakeBorder(img, 0, 0, int(32-new_width/2), int(32-new_width/2), cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        new_width = 64
        new_height = int(64/ratio/2)*2
        img = cv2.resize(img, (new_width, new_height))
        img = cv2.copyMakeBorder(img, int(64-new_height/2), int(64-new_height/2), 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img


def resize_force(img):
    """
    img simply resize
    :param img:
    :return:
    """
    img = cv2.resize(img, (64, 96))
    return img


def pipeline(input_path, output_path):
    """
    pipeline function for input and output
    :return:
    """
    dirs = os.listdir(input_path)
    for file_dir in dirs:
        cat_dirs = os.listdir(input_path+file_dir)
        for cat_dir in cat_dirs:
            if cat_dir != 'car':
                continue
            files = os.listdir(input_path+file_dir+'/'+cat_dir)
            if files:
                for file in files:
                    img = cv2.imread(input_path+file_dir+'/'+cat_dir+'/'+file)
                    if img.shape[0] < 40 or img.shape[1] < 20:
                        continue
                    img = resize_force(img)
                    cv2.imwrite(output_path+cat_dir+'/'+file, img)


if __name__ == '__main__':
    pipeline('./crop/', './resize/')
