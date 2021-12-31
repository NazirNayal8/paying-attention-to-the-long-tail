"""
Code Adapted from: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/utils/metrics.py
"""

import torch
import numpy as np
from torch import Tensor
from torchmetrics import Metric


class mIoU(Metric):

    def __init__(self, num_classes):
        super().__init__()

        self.add_state('num_classes', default=num_classes, persistent=True)
        self.add_state('total_inter', default=torch.tensor(0))
        self.add_state('total_union', default=torch.tensor(0))

    def update(self, output: Tensor, targets: Tensor) -> None:

        area_inter, area_union = batch_intersection_union(output=output, targets=targets, nclass=self.num_classes)

        self.total_inter += area_inter
        self.total_union += area_union

        return area_inter, area_union

    def compute(self):
        return get_miou(self.total_inter, self.total_union)


class PerPixelAccuracy(Metric):

    def __init__(self, num_classes):
        super().__init__()

        self.add_state('total_correct', default=torch.tensor(0))
        self.add_state('total_label', default=torch.tensor(0))

    def update(self, output: Tensor, targets: Tensor) -> None:

        pixel_correct, pixel_labeled = batch_pix_accuracy(output=output, targets=targets)

        self.total_correct += pixel_correct
        self.total_label += pixel_labeled

        return pixel_correct, pixel_labeled

    def compute(self):
        return get_pixel_accuracy(self.total_correct, self.total_label)


def get_miou(total_inter, total_union):
    """
    Compute mean Intersection over Union
    :param total_inter:
    :param total_union:
    :return:
    """
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()
    return mIoU


def get_pixel_accuracy(total_correct, total_label):
    """
    Compute Total Pixel Accuracy
    :param total_correct:
    :param total_label:
    :return:
    """
    pix_acc = 1.0 * total_correct / (np.spacing(1) + total_label)
    return pix_acc


def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        output: input 4D tensor
        target: label 3D tensor
    """
    _, predict = torch.max(output, 1)

    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(output, 1)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union
