# here we define IoU, SEG, MUCov and Dice Loss

import copy, torch, torch.nn as nn, numpy as np

from skimage import measure
from torch import tensor 
from typing import Callable
from dataclasses import dataclass 



"""
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
"""

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        smooth = 1e-6

        intersection = torch.sum(predictions * targets)

        dice_score = (2. * intersection + smooth) / (
                torch.sum(predictions) + torch.sum(targets) + smooth)

        return dice_score
    
def IoU(predictions, targets):
    intersection = torch.sum(predictions * targets)
    TP = intersection
    FP = torch.sum(predictions) - intersection
    FN = torch.sum(targets) - intersection

    iou = TP / (TP + FP + FN)
    return iou


def SEG(predictions, targets):
    # Get the unique labels for each instance in the targets, excluding background (label 0)
    unique_labels = targets.unique()[1:]
    Ni = len(unique_labels)
    sum_max_iou = 0.0

    # Loop through each unique label in the target
    for label in unique_labels:
        targets_mask = (targets == label)
        
        # Initialize to a value lower than any possible IoU
        max_iou_for_label = -1  
        for pred_label in predictions.unique():
            iou = IoU(predictions == pred_label, targets_mask)
            if iou > max_iou_for_label:
                max_iou_for_label = iou
        # Add the max IoU for this label to the sum
        sum_max_iou += max_iou_for_label

    # Calculate SEG by averaging the sum of max IoUs for each label
    SEG_score = sum_max_iou / Ni
    return SEG_score