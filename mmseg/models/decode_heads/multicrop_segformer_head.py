# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseMultiCropDecodeHead

from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class MultiCrop_SegformerHead(BaseMultiCropDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out
    
    def _stack_batch_gt(self, batch_data_samples):
        if 'mc_seg_map' in batch_data_samples[0]:
            gt_semantic_segs = [
                data_sample.mc_seg_map.data for data_sample in batch_data_samples
            ]
        else:
            gt_semantic_segs = [
                data_sample.gt_sem_seg.data for data_sample in batch_data_samples
            ]
        gt_semantic_segs = torch.stack(gt_semantic_segs, dim=0)
        assert len(gt_semantic_segs.shape) == 4, "gt shape is {}".format(gt_semantic_segs.shape)
        b, m, h, w = gt_semantic_segs.shape
        gt_semantic_segs = gt_semantic_segs.reshape(b*m, h, w)
        return gt_semantic_segs.unsqueeze(1)
