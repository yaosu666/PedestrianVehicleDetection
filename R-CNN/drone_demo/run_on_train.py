# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os
import glob
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )


    PATH = "/home/xiaosongwen0313_gmail_com/VisDrone2019-MOT-train/sequences/uav0000239_03720_v/*.jpg"
    basename = 'uav0000239_03720_v'
    files = sorted(glob.glob(PATH))
    image = cv2.imread(files[0])
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('../../video/%s_output.mp4' %basename, fourcc, 12.0, (image.shape[1],image.shape[0]))
    start = time.time()
    index = 0
    for file in files:
        start_time = time.time()
        image = cv2.imread(file)
        predictions = coco_demo.run_on_opencv_image(image)             
        writer.write(predictions)        
        print(index, '/', len(files),"\tTime: {:.2f} s / img".format(time.time() - start_time))
        index += 1
    print('total time: {:.2f}'.format(time.time() - start), 'total frames: ', len(files))
    #cv2.destroyAllWindows()


if __name__ == "__main__":
    main()