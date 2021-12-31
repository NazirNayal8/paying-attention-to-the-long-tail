"""
Adapted from https://github.com/zhangyongshun/BagofTricks-LT/blob/main/lib/loss/loss_impl/class_balanced_loss.py
"""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from losses.focal_loss import focal_loss


class ClassBalancedLoss(nn.Module):
    """
    An implementation of Class Balanced Loss proposed by  Cui et al. https://arxiv.org/abs/1901.05555
    """

    def __init__(self, loss_type, beta, num_classes, per_class_frequency, focal_loss_gamma=None, device=None):
        super().__init__()

        self.num_classes = num_classes
        self.per_class_frequency = per_class_frequency
        self.loss_type = loss_type
        self.beta = beta
        self.focal_loss_gamma = focal_loss_gamma

        # calculate class balanced_loss
        self.weight = np.array([(1 - self.beta)/(1 - self.beta ** N) for N in self.per_class_frequency])
        self.weight = torch.FloatTensor(self.weight / np.sum(self.weight) * self.num_classes).to(device)

        if self.loss_type == 'cross_entropy':
            self.base_loss = nn.CrossEntropyLoss(weight=self.weight)
        elif self.loss_type == 'focal_loss':
            if focal_loss_gamma is None:
                raise Exception(f'Class Balanced Loss with Focal Loss must be provided with focal_loss_gamma parameter')
            self.base_loss = focal_loss(
                alpha=self.weight,
                gamma=self.focal_loss_gamma
            )
        else:
            raise Exception(f'Class Balanced Loss does not support loss type {self.loss_type}')

    def forward(self, preds: Tensor, targets: Tensor):

        return self.base_loss(preds, targets)
