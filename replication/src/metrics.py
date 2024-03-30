# here we define IoU, SEG, MUCov and Dice Loss

import copy, torch, torch.nn as nn, numpy as np

from skimage import measure
from torch import tensor 
from typing import Callable, Dict
from dataclasses import dataclass 

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        smooth = 1e-6
        predictions = torch.sigmoid(predictions)

        pred_flat = predictions.contiguous().view(-1)
        target_flat = targets.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice_coeff = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        return 1 - dice_coeff
    
def IoU(predictions, targets):
    intersection = torch.sum(predictions * targets)
    TP = intersection
    FP = torch.sum(predictions) - TP
    FN = torch.sum(targets) - TP

    iou = TP / (TP + FP + FN)
    return iou


def SEG(predictions, targets):
    # Unique lables excluding background (label 0)
    unique_labels_all = predictions.unique()
    unique_labels = unique_labels_all[unique_labels_all != 0]
    Ni = len(unique_labels) # Ni: number of segmented Nuclei
    
    # Loop through each unique label in the target
    sum_max_iou = 0.0
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

def MuCov(predictions, targets):
    # Unique lables excluding background (label 0)
    unique_labels_all = targets.unique()
    unique_labels = unique_labels_all[unique_labels_all != 0]
    Nj = len(unique_labels)  # Number of ground truth objects

    # Loop through each unique label in the predictions
    sum_max_iou = 0.0
    for pred_label in unique_labels:
        # Create a mask for the current prediction label, where 1 is the current label and 0 is everything else
        predictions_mask = (predictions == pred_label)
        
        # Calculate IoU between the prediction mask and each target, take the max
        # Additionally, apply the constraint condition (IoU > 0.5)
        max_iou_for_pred = 0.0
        for target_label in unique_labels:
            targets_mask = (targets == target_label)
            iou = IoU(predictions_mask, targets_mask)
            if iou > 0.5:
                max_iou_for_pred = max(max_iou_for_pred, iou)
        
        sum_max_iou += max_iou_for_pred

    # Calculate MuCov by averaging the sum of max IoUs for each predicted label
    MuCov_score = sum_max_iou / Nj
    return MuCov_score


@dataclass 
class Metrics:
    ''' Metrics relevant for this reproduction '''

    # Functions are first-class objects in Python, so they can be passed as arguments.
    iou: Callable[[torch.Tensor, torch.Tensor], float]
    seg: Callable[[torch.Tensor, torch.Tensor], float]
    mucov: Callable[[torch.Tensor, torch.Tensor], float]

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        ''' Takes raw image tensors, returns a metric dict. '''

        # Ensure that the predictions and targets are boolean tensors.
        prediction_bin = predictions > 0
        target_bin = targets > 0

        # Compute the metrics.
        iou_score = self.iou(prediction_bin, target_bin)
        seg_score = self.seg(predictions, targets)
        mucov_score = self.mucov(predictions, targets)

        # Return a dictionary of the computed metrics.
        return {
            'IoU': iou_score,
            'SEG': seg_score,
            'MUCov': mucov_score
        }