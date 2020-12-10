# encoding: utf-8
# author: Lidian Lin
# email: lldlld0574@gmail.com
# file: crop_pipeline.py
# time: 2020/11/14 18:25
# desc: img crop based on bounding box parameter
import os
from PIL import Image
import pandas as pd


class ImgCrop:
    def __init__(self, img_path):
        self.img = Image.open(img_path)
        # print(self.img.size)

    def crop(self, img_crop_tuple, img_no, save_path):
        # print(img_crop_tuple)
        cropped = self.img.crop(img_crop_tuple)
        cropped.save(save_path+str(img_no)+'.jpg')


def pipeline():
    files = os.listdir('annotations')
    folder_count = 0
    total_count = 0
    for file_name in files:
        df = pd.read_csv('./annotations/'+file_name, header=None, names=['img_no', 'obj_no', 'x', 'y', 'w', 'h', 'a', 'b', 'c', 'd'])
        df.sort_values('img_no', inplace=True)
        print('-----------------')
        print('now come to filename: %s' % file_name)
        file_root = './crop/'+file_name.replace('.txt', '/')
        if not os.path.exists(file_root):
            os.mkdir(file_root)
            os.mkdir(file_root+'people')
            os.mkdir(file_root+'car')
            os.mkdir(file_root+'van')
        last_img_no = 1
        img_operator = ImgCrop('./sequences/'+file_name.replace('.txt', '/')+'%07d' % 1 + '.jpg')
        i = 0
        target_id_list =[]
        for idx, row in df.iterrows():
            target_id = int(row[1])
            if target_id in target_id_list:
                continue
            else:
                target_id_list.append(target_id)
            obj_cat = int(row[7])
            if obj_cat not in [1,2,4,5]:
                continue
            if row['img_no'] != last_img_no:
                last_img_no = row['img_no']
                img_operator = ImgCrop('./sequences/'+file_name.replace('.txt', '/')+'%07d' % last_img_no + '.jpg')
            i += 1
            total_count += 1
            x = int(row['x'])
            y = int(row['y'])
            w = int(row['w'])
            h = int(row['h'])
            img_crop_tuple = (x, y, x+w, y+h)
            img_operator.crop(img_crop_tuple, i, './crop/'+file_name.replace('.txt', '/')+cat_dict.get(obj_cat)+'/')
        # if folder_count > 4:
        #     break
        folder_count += 1


if __name__ == "__main__":
    cat_dict = {1: 'people', 2: 'people', 4: 'car', 5: 'car'}
    pipeline()
