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
