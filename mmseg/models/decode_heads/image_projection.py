# Copyright (c) OpenMMLab. All rights reserved.
from typing import List
import torch
from torch import Tensor
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.unidecoder_head import UniDecodeHead

from mmseg.registry import MODELS
from mmseg.utils import SampleList
from ..utils import resize


@MODELS.register_module()
class ImageProject_Head(UniDecodeHead):
    """Project the image feature to Nxdim to match the text feature.

    Args:
        - interpolate_mode: ....
        - output_dim: the text feature's hidden dim from clip text encoder.
    """

    def __init__(self, interpolate_mode='bilinear', output_dim=512, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        input_dim = 0
        for in_channel in self.in_channels:
            input_dim += in_channel
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            outs.append(
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = torch.cat(outs, dim=1)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)

        return out
    
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        img_level_texts = [
            data_sample.img_level_text for data_sample in batch_data_samples
        ]
        img_level_texts = torch.stack(img_level_texts)
        assert len(img_level_texts.shape) == 3, "img_level_text shape is {}".format(img_level_texts.shape)
        b, m, d = img_level_texts.shape
        img_level_texts = img_level_texts.reshape(b*m, d)
        # img_level_texts shape is [b*m,d]
        return img_level_texts
    
    def loss_by_feat(self, img_feature: Tensor, batch_data_samples: SampleList) -> dict:
        text_feature = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    img_feature,
                    text_feature)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    img_feature,
                    text_feature)

        return loss

    def predict_by_feat(self, seg_logits: Tensor, batch_img_metas: List[dict]) -> Tensor:
        return seg_logits