#-------------------------------------#
#       创建YOLO类
#-------------------------------------#
import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from nets.yolo3 import YoloBody
import torch.backends.cudnn as cudnn
from PIL import Image,ImageFont, ImageDraw
from torch.autograd import Variable
from utils.config import Config
from utils.utils import non_max_suppression, bbox_iou, DecodeBox,letterbox_image,yolo_correct_boxes

#使用yolov3的网络结构模型

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo_weights.pth',
        "classes_path": 'model_data/coco_classes.txt',
        # "model_path" : 'logs/Epoch50-Total_Loss164.7020-Val_Loss184.9570.pth',
        # "classes_path": 'model_data/new_classes.txt',
        "model_image_size" : (416, 416, 3),
        "confidence": 0.5,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.config = Config
        self.generate()
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   加载模型和初始化过程
    #---------------------------------------------------#
    def generate(self):
        self.config["yolo"]["classes"] = len(self.class_names)
        self.net = YoloBody(self.config)

        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        self.yolo_decodes = []  # 对yolov3的网络预测结果进行解码DecodeBox,详情请见utils.py
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.config["yolo"]["anchors"][i], self.config["yolo"]["classes"],  (self.model_image_size[1], self.model_image_size[0])))


        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 设置画框的不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))


    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        # 加灰条
        crop_img = np.array(letterbox_image(image, (self.model_image_size[0],self.model_image_size[1])))
        photo = np.array(crop_img,dtype = np.float32)
        # 图片的归一化
        photo /= 255.0
        # 改变图片的通道数 pytorch 中通道数在前面
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        # 给图片扩充一个维度，batchsize，这样图片才能才能放到网络中去预测
        images = []
        images.append(photo)
        # 将图片的格式从numpy转变成pytorch中的Tensor变量
        images = np.asarray(images)
        images = torch.from_numpy(images)
        if self.cuda:
            images = images.cuda()

        # 对yolov3的网络预测结果（3个有效的特征层）进行解码和筛选
        with torch.no_grad():
            # 将图片传入到网络中去预测，网络的预测结果是13*13,26*26,52*52有效特征层的预测结果
            outputs = self.net(images)

            # 分别对这3个有效特征层进行解码操作（即先验框调整的过程），得到3个特征层解码后的预测框，即是可以真实的画在图片上的框框
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))  # 分别对3个有效特征层解码之后的预测结果进行堆叠，存放在列表中（batchsize,3,13,13,5+numclasses）
            output = torch.cat(output_list, 1)
            # print(output.shape)
            # print(output) # torch.Size([1, 10647, 85])

            # 对3个特征层上的每个网格点上的3个先验框进行非极大抑制进行筛选
            batch_detections = non_max_suppression(output, self.config["yolo"]["classes"],
                                                    conf_thres=self.confidence,
                                                    nms_thres=0.3)

        # 判断进行非极大抑制之后图片上是否还有框框，若没有，则直接返回图片
        try :
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image

        # 对进行非极大值之后的框框进行再次筛选
        # 框的置信度*种类的置信度 进行阈值（置信度）判断，把置信度较高的框框筛选出来
        top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence
        top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
        top_label = np.array(batch_detections[top_index,-1],np.int32)
        top_bboxes = np.array(batch_detections[top_index,:4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

        # 我们在原始图片上加的灰条
        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

