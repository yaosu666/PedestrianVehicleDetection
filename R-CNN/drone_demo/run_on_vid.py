# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os

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
        default=0.45,
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


    path = "../../video/"
#     files = ['DJI_0673.MOV', 'DJI_0680.MOV', 'night_3_25.mp4']
    files = ['daytime_2_30.mp4', 'night_4_25.mp4']
    for file in files:
        input_path = path + file
        print('Working on %s' % input_path)
        cam = cv2.VideoCapture(input_path)
        basename = os.path.basename(input_path).split('.')[0]
        print('basename:', basename)
        ret_val, img = cam.read()
        print(int(cam.get(3)),int(cam.get(4)))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter('../../video/%s_output.mp4' %basename, fourcc, 12.0, (int(cam.get(3)),int(cam.get(4))))
        start = time.time()
        index = 0
        while True:
            ret_val, img = cam.read()
            if not ret_val:                
                print('Done!')
                break 
            composite = coco_demo.run_on_opencv_image(img)
            index += 1
            writer.write(composite)
        print('total time: {:.2f}'.format(time.time() - start), 'total frames: ', index)
    #cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
