# here we define IoU, SEG, MUCov and Dice Loss

import copy, torch, torch.nn as nn, numpy as np

from skimage import measure
from torch import tensor 
from typing import Callable
from dataclasses import dataclass 


def IoU(predictions_bin, targets_bin) -> float:
    ''' Takes predictions and targets as *binary* tensors, 
        and returns a single IoU score. 
    
        Intersection over Union loss. 
        "Conventionally used to measure segmentation accuracy, 
        because it comprehensively measures FP and FN rates" (p. 2) 

            IoU = TP / (TP + FP + FN)

        "However, because IoU is calculated for each image, it cannot
        evaluate whether or not segmentation is accurate (i.e. nuclei 
        are not fused)."
    '''
    count_pos = copy.deepcopy(predictions_bin + targets_bin)
    count_neg = copy.deepcopy(predictions_bin - targets_bin)

    TP = len(np.where(count_pos.reshape(count_pos.size)==2)[0])
    FP = len(np.where(count_neg.reshape(count_neg.size)==1)[0])
    FN = len(np.where(count_neg.reshape(count_neg.size)==-1)[0])

    try:
        iou = TP / float(TP + FP + FN)
    except:
        iou = 0
    return iou


def SEG(y, y_ans): 
    ''' Structural Similarity Index (Instance Segmentation) 
        "Represents the average of IoU of each instance by the sum
        of the numbers of correct nuclear regions" 
        
            SEG = Sum_j ^(N_i) 1/(N_i) • max_i IoU (y_i, y*_j)
        
        Where N_i is the number of segmented nuclei, y is the seg-
        mented nuclear region, y* is the ground truth.  ''' 

    sum_iou = 0
    true_labels = np.unique(y_ans)[1:]
    print(f'ground truth labels {true_labels}')

    for i in true_labels:

        y_ans_mask = np.array((y_ans == i) * 1).astype(np.int8)
        rp = measure.regionprops(y_ans_mask)[0]
        bbox = rp.bbox

        y_roi = y[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        pred_labels = np.unique(y_roi)[1:]
        best_iou, best_thr = 0, 0

        for j in pred_labels:

            y_mask = np.array((y == j) * 1).astype(np.int8)
            iou, thr = IoU(y_mask, y_ans_mask)

            if best_iou <= iou:
                best_iou = iou
                best_thr = np.max([thr, best_thr])

        print(f'c{i:3} best IoU in SEG: {best_iou:.4f}')
        if best_thr > 0.5: 
            sum_iou += best_iou

    return sum_iou / len(true_labels)



def MUCov(predictions, targets):
    ''' Mean Covariance Index (Instance Segmentation)
        "Evaluate individual segmented nuclear regions and repre-
        sents the average of the IoU of each instance by the sum 
        of the numbers of segmentation regions" 

            MuCov = Sum_i ^(N_j) 1/(N_j) • max_j IoU (y_i, y*j)
        
        Where N_j is the number of ground truth objects, y is the
        segmented nuclear region, y* is the ground truth.  '''

    sum_iou = 0
    for i in range(1, predictions.max()+1):
        mask_y = np.array((predictions == i) * 1).astype(np.uint8)
        best_iou = 0
        for j in range(1, targets.max()+1):
            mask_y_ans = np.array((targets == j) * 1).astype(np.uint8)
            best_iou = np.max([IoU(mask_y, mask_y_ans), best_iou])
        print('best IoU: {}'.format(best_iou))
        sum_iou += best_iou
    return sum_iou / predictions.max()


@dataclass 
class Metrics:
    ''' Metrics relevant for this reproduction '''

    # fuuck it functions count as data in python
    # if something is wrong with one of them but you still want to test 
    # the others, you can pass in a function, e.g. Metrics(iou=Callable)

    iou   :Callable[[tensor, tensor], float] = IoU
    seg   :Callable[[tensor, tensor], float] = SEG
    mucov :Callable[[tensor, tensor], float] = MUCov

    def compute(self, predictions, targets) -> dict[str, float]:
        ''' Takes raw image tensors, returns a metric dict. 
            Set full to False to not compute mean & stdev. '''

        prediction_bin = np.array((predictions > 0) * 1).astype(np.uint8)
        target_bin = np.array((targets > 0) * 1).astype(np.uint8)

        return {
            'IoU': self.iou(prediction_bin, target_bin),
            'SEG': self.seg(predictions, targets),
            'MUCov': self.mucov(predictions, targets)
        }


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        smooth = 1e-6

        intersection = torch.sum(predictions * targets)

        dice_score = (2. * intersection + smooth) / (
                torch.sum(predictions) + torch.sum(targets) + smooth)

        return dice_score
