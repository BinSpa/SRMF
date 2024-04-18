import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS

@MODELS.register_module()
class InfoNCELoss(nn.Module):
    """InfoNCE loss
    We don't support weight_class for now.
    Args:
        temperature: softmax temperature.
        loss_name: default is 'loss_infonce'.
        reduction: 'mean' or 'sum' or 'none'.
        ignore_index: default 255.
    """
    def __init__(self, temperature=0.07, loss_name='loss_infonce', reduction='mean', ignore_index=255):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self._loss_name = loss_name
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, img_feature, text_feature):
        """
        img_feature: Tensor of shape [n, dim], where n is batch size
        text_feature: Tensor of shape [n, dim], assuming one-to-one correspondence
        """
        # Normalize features
        img_feature = F.normalize(img_feature, p=2, dim=1)
        text_feature = F.normalize(text_feature, p=2, dim=1)
        
        # Compute similarity matrix
        assert img_feature.device == text_feature.device, "expect all tensor in the same device, but got img_feature:{}, text_feature:{}".format(img_feature.device, text_feature.device)
        similarity_matrix = torch.mm(img_feature, text_feature.t()) / self.temperature
        
        # Diagonal elements are positives, off-diagonals are negatives
        n = img_feature.size(0)
        labels = torch.arange(n, dtype=torch.long, device=img_feature.device)
        
        # Use log-softmax for numerical stability
        loss = F.cross_entropy(similarity_matrix, labels, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name