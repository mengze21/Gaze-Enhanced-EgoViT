# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# Modified for hand and object fetures extration
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import pandas as pd
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
# from torchviz import make_dot

import torchvision.transforms as transforms
import torchvision.datasets as dset

# os.chdir('hand_object_extractor')
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, \
    vis_detections_filtered_objects_PIL, vis_detections_filtered_objects  # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
# import matplotlib.pyplot as plt
import json

# from tensorboardX import SummaryWriter
# from torchsummary import summary

# from torchviz import make_dot

# range = range  # Python 3


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Extract the hand and object fetures')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save results',
                        default="images_det")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA', default=True, required=False,
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=8, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=132028, type=int, required=False)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        default=True)
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    parser.add_argument('--thresh_hand',
                        type=float, default=0.5,
                        required=False)
    parser.add_argument('--thresh_obj', default=0.5,
                        type=float,
                        required=False)

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def save_data(box, features, image_name, handdata):
    current_wording_dir = os.getcwd()
    # print(f"current_wording_dir is: {current_wording_dir}")
    # Get the base name of the image
    image_name = os.path.basename(image_name)
    image_name, _ = os.path.splitext(image_name)
    # print(f"image_name is: {image_name}")
    # Move one directory up
    current_wording_dir = os.path.dirname(current_wording_dir)

    # np.savetxt('array.txt', features, delimiter=',')
    if handdata:
        path_to_save = current_wording_dir + '/Extracted_HOFeatures_224/hand/' + image_name

        combined_data = {
            "class": "hand",  # "hand" or "object
            "image_name": image_name,
            "handt_box": box.tolist(),
            "hand_features": features.tolist()
        }
        with open(path_to_save + '.json', 'w') as f:
            json.dump(combined_data, f, indent=4)
    else:
        path_to_save = current_wording_dir + '/Extracted_HOFeatures_224/object/' + image_name

        combined_data = {
            "class": "object",  # "hand" or "object
            "image_name": image_name,
            "object_box": box.tolist(),
            "object_features": features.tolist()
        }
        with open(path_to_save + '.json', 'w') as f:
            json.dump(combined_data, f, indent=4)
        # conbine_data = np.concatenate((box, features), axis=1)
        # Convert detected box and it's features to lists and create a DataFrame
        # df = pd.DataFrame({'image': [image_name], 'object_box': [box], 'object_features': [features]})
        # df['object_box'] = df['object_box'].astype(str)
        # df['object_features'] = df['object_features'].astype(str)
        #
        # # Save the DataFrame to a text file
        # df.to_csv(path_to_save, index=False)


if __name__ == '__main__':

    args = parse_args()

    if args.cfg_file is not None:
        print(f"Using cfg_file: {args.cfg_file}")
        cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    np.random.seed(cfg.RNG_SEED)

    # load model
    model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
    print(f"model_dir is : {model_dir}")
    if not os.path.exists(model_dir):
        raise Exception('There is no input directory for loading network from ' + model_dir)
    load_name = os.path.join(model_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]']

    # initilize the network 
    if args.net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        raise NameError("fasterRCNN is not defined")

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))

    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))

    fasterRCNN.load_state_dict(checkpoint['model'])

    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # Initialization of tensor placeholders for image and bounding box data
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1)

    # ship data to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    with torch.no_grad():
        if args.cuda > 0:
            cfg.CUDA = True

        if args.cuda > 0:
            fasterRCNN.cuda()

        fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    thresh_hand = args.thresh_hand
    thresh_obj = args.thresh_obj
    vis = args.vis
    webcam_num = args.webcam_num

    #
    print(f'loaded images dirctory is: {args.image_dir}')
    print(f'demo images will be saved in: {args.save_dir}')

    # List all files in the image directory
    imglist = os.listdir(args.image_dir)
    # print(f"imglist is: {imglist}")
    # # Filter out files with just .jpg extension(for macOS could .DS_Store in imglist)
    # imglist = [file for file in imglist if file.lower().endswith('.jpg')]

    for root, dirs, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                imglist.append(os.path.join(root, file))
                
    # Determine the number of .jpg images in the list
    num_images = len(imglist)

    print('{} images are loaded.'.format(num_images))

    i = 0
    total_tic = time.time()
    while (num_images > 0):
        img_tic = time.time()
        num_images -= 1
        # Load the image from dataset
        im_file = os.path.join(args.image_dir, imglist[num_images])
        # print('Processing image: {}'.format(im_file))
        im_in = cv2.imread(im_file)
        # bgr
        im = im_in

        blobs, im_scales = _get_image_blob(im)

        assert len(im_scales) == 1, "Only single-image batch implemented"

        im_blob = blobs

        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        # Convert the numpy array to a PyTorch tensor
        im_data_pt = torch.from_numpy(im_blob)
        # Rearrange the dimensions of 'im_data_pt' from (batch size, height, width, channels) to
        # (batch size, channels, height, width). This is necessary because PyTorch models expect
        # the channel dimension to be the second dimension.
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        # print(f"im_data_pt size （batch size, channels, height, width)-----> {im_data_pt.size()}")

        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
            box_info.resize_(1, 1, 5).zero_()

        det_tic = time.time()

        # forward pass to get the hand and object features
        # (rois, cls_prob, bbox_pred,
        #  rpn_loss_cls, rpn_loss_box,
        #  RCNN_loss_cls, RCNN_loss_bbox,
        #  rois_label, loss_list) = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

        (rois, cls_prob, bbox_pred,
         pooled_feat, loss_list) = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # extact predicted params
        contact_vector = loss_list[0][0]  # hand contact state info
        offset_vector = loss_list[1][0].detach()  # offset vector (factored into a unit vector and a magnitude)

        lr_vector = loss_list[2][0].detach()  # hand side info (left/right)

        # get hand contact 
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

        # get hand side 
        lr = torch.sigmoid(lr_vector) > 0.5
        lr = lr.squeeze(0).float()

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            # print("1")
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # print("11")
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    # print("111")
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    # print("112")
                    if args.cuda > 0:
                        # print("1121")
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # print("2")
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        # print(f"shape of scores is {scores.shape}")
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic

        misc_tic = time.time()

        if vis:
            im2show = np.copy(im)

        obj_dets, hand_dets = None, None
        for j in range(1, len(pascal_classes)):
            # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
            if pascal_classes[j] == 'hand':
                # get the index of scores > thresh_hand. the index of hand
                inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
            elif pascal_classes[j] == 'targetobject':
                # get the index of scores > thresh_object. the index of object
                inds = torch.nonzero(scores[:, j] > thresh_obj).view(-1)

            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]  # the detected scores
                _, order = torch.sort(cls_scores, 0, True)  # the index of inds
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                pooled_feat_det = pooled_feat[inds]
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds],
                                      offset_vector.squeeze(0)[inds], lr[inds]), 1)
                cls_dets = cls_dets[order]
                cls_boxes_det = cls_boxes[order]
                pooled_feat_det = pooled_feat_det[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                # print(f"the value of keep is:{keep.item()}")
                cls_dets = cls_dets[keep.view(-1).long()]  # detected box of hand or object
                cls_boxes_det = cls_boxes_det[keep.view(-1).long()]
                pooled_feat_det = pooled_feat_det[keep.view(-1).long()]  # the features of detected box
                if pascal_classes[j] == 'targetobject':
                    # convert tensor to numpy
                    obj_dets = cls_dets.cpu().numpy()
                    # obj_dets = cls_boxes_det.cpu().numpy()
                    pooled_feat_det = pooled_feat_det.detach().cpu().numpy()

                    # save data
                    save_data(obj_dets, pooled_feat_det, imglist[num_images], handdata=False)

                if pascal_classes[j] == 'hand':
                    # convert tensor to numpy
                    hand_dets = cls_dets.cpu().numpy()
                    # hand_dets = cls_boxes_det.cpu().numpy()
                    pooled_feat_det = pooled_feat_det.detach().cpu().numpy()

                    # save combined data
                    save_data(hand_dets, pooled_feat_det, imglist[num_images], handdata=True)

        if vis:
            # visualization
            im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)

        img_toc = time.time()
        # nms_time = misc_toc - misc_tic
        perimg_time = img_toc - img_tic
        elapsed_time = img_toc - total_tic
        elapsed_time = elapsed_time / 60
        # sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
        #                  .format(num_images + 1, len(imglist), detect_time, nms_time))
        sys.stdout.write("\rprocessed image: {:d}/{:d} in {:.3f}s ------ total time: {:.3f}min"
                         .format(i + 1, len(imglist), perimg_time, elapsed_time))
        sys.stdout.flush()
        vis = False
        if vis and webcam_num == -1:
            folder_name = args.save_dir
            os.makedirs(folder_name, exist_ok=True)
            result_path = os.path.join(folder_name, imglist[num_images][:-4] + "_det.png")
            im2show.save(result_path)
        # else:
        #     im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
        #     cv2.imshow("frame", im2showRGB)
        #     total_toc = time.time()
        #     total_time = total_toc - total_tic
        #     frame_rate = 1 / total_time
        #     print('Frame rate:', frame_rate)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        i += 1

print("\nThe extraction was finished")
