import math
from random import shuffle

import cv2
import keras
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image

from utils.utils import cvtColor


class RetinanetDatasets(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, train, ignore_threshold = 0.4, overlap_threshold = 0.5):
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        
        self.input_shape        = input_shape
        self.anchors            = anchors
        self.num_anchors        = len(anchors)
        self.batch_size         = batch_size
        self.num_classes        = num_classes
        self.train              = train
        self.ignore_threshold   = ignore_threshold
        self.overlap_threshold  = overlap_threshold

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        image_data      = []
        regressions     = []
        classifications = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.length

            image, box  = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)
            if len(box)!=0:
                boxes               = np.array(box[:,:4] , dtype=np.float32)
                boxes[:, [0, 2]]    = boxes[:,[0, 2]] / self.input_shape[1]
                boxes[:, [1, 3]]    = boxes[:,[1, 3]] / self.input_shape[0]
                one_hot_label   = np.eye(self.num_classes)[np.array(box[:,4], np.int32)]
                box             = np.concatenate([boxes, one_hot_label], axis=-1)
            assignment      = self.assign_boxes(box)
            regression      = assignment[:,:5]
            classification  = assignment[:,5:]

            image_data.append(preprocess_input(image))
            regressions.append(regression)
            classifications.append(classification)

        return np.array(image_data), [np.array(regressions,dtype=np.float32), np.array(classifications,dtype=np.float32)]

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        line = annotation_line.split()

        image   = Image.open(line[0])
        image   = cvtColor(image)

        iw, ih  = image.size
        h, w    = input_shape

        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])


        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box

        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255 # numpy array, 0 to 1

        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box

    def on_epoch_begin(self):
        shuffle(self.annotation_lines)

    def iou(self, box):

        inter_upleft    = np.maximum(self.anchors[:, :2], box[:2])
        inter_botright  = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh    = inter_botright - inter_upleft
        inter_wh    = np.maximum(inter_wh, 0)
        inter       = inter_wh[:, 0] * inter_wh[:, 1]

        area_true = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0])*(self.anchors[:, 3] - self.anchors[:, 1])
        union = area_true + area_gt - inter
        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True, variance=0.2):
        iou = self.iou(box)
        ignored_box         = np.zeros((self.num_anchors, 1))
        assign_mask_ignore  = (iou > self.ignore_threshold) & (iou < self.overlap_threshold)
        ignored_box[:, 0][assign_mask_ignore] = iou[assign_mask_ignore]

        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        assigned_anchors = self.anchors[assign_mask]
        assigned_anchors_w = (assigned_anchors[:, 2] - assigned_anchors[:, 0])
        assigned_anchors_h = (assigned_anchors[:, 3] - assigned_anchors[:, 1])

        encoded_box[:,0][assign_mask] = (box[0] - assigned_anchors[:, 0])/assigned_anchors_w/variance
        encoded_box[:,1][assign_mask] = (box[1] - assigned_anchors[:, 1])/assigned_anchors_h/variance
        encoded_box[:,2][assign_mask] = (box[2] - assigned_anchors[:, 2])/assigned_anchors_w/variance
        encoded_box[:,3][assign_mask] = (box[3] - assigned_anchors[:, 3])/assigned_anchors_h/variance
        
        return encoded_box.ravel(), ignored_box.ravel()

    def assign_boxes(self, boxes):
        assignment          = np.zeros((self.num_anchors, 4 + 1 + self.num_classes + 1))
        assignment[:, 4]    = 0.0
        assignment[:, -1]   = 0.0
        if len(boxes) == 0:
            return assignment

        apply_along_axis_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = np.array([apply_along_axis_boxes[i, 0] for i in range(len(apply_along_axis_boxes))])
        ingored_boxes = np.array([apply_along_axis_boxes[i, 1] for i in range(len(apply_along_axis_boxes))])
        ingored_boxes   = ingored_boxes.reshape(-1, self.num_anchors, 1)
        ignore_iou      = ingored_boxes[:, :, 0].max(axis=0)
        ignore_iou_mask = ignore_iou > 0
        assignment[:, 4][ignore_iou_mask] = -1
        assignment[:, -1][ignore_iou_mask] = -1
        encoded_boxes   = encoded_boxes.reshape(-1, self.num_anchors, 5)
        best_iou        = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx    = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask   = best_iou > 0
        best_iou_idx    = best_iou_idx[best_iou_mask] 
        assign_num      = len(best_iou_idx)
        encoded_boxes   = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,np.arange(assign_num),:4]
        assignment[:, 4][best_iou_mask]     = 1
        assignment[:, 5:-1][best_iou_mask]  = boxes[best_iou_idx, 4:]
        assignment[:, -1][best_iou_mask]    = 1
        return assignment
