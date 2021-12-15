import timm
import torch.nn as nn
from torch import Tensor


class TimmViT(nn.Module):

    def __init__(self, model_type: str, pretrained: bool, num_classes: int):
        super().__init__()

        self.num_classes = num_classes
        self.model = timm.create_model(model_type, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x)
