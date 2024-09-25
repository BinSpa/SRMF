# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.text_decode_head import BaseTextDecodeHead

from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class MCText_SegformerHead(BaseTextDecodeHead):
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
            in_channels=self.channels * num_inputs + self.text_nums,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
        self.img_project = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=2048,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )

    def forward(self, inputs, text_features):
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

        outs_feature = torch.cat(outs, dim=1)
        # project image feature to text feature
        project_outs = self.img_project(outs_feature)
        project_text = text_features
        project_outs = project_outs / project_outs.norm(dim=1, keepdim=True)
        project_text = project_text / project_text.norm(dim=2, keepdim=True)
        # cacluate the cosine similarity
        b, c, h, w = project_outs.shape
        _, n, _ = text_features.shape
        # project_outs:(b,c,hw)
        project_outs = project_outs.view(b, c, -1)
        assert project_text.shape[0] == project_outs.shape[0], "text:{}; outs:{}".format(project_text.shape, project_outs.shape)
        assert project_text.device == project_outs.device, "text:{}; img:{}".format(project_text.device, project_outs.device)
        imgtxt_feature = torch.einsum('bnc,bck->bnk', project_text, project_outs)
        imgtxt_feature = imgtxt_feature.view(b,n,h,w)
        outs.append(imgtxt_feature)
        # fusion ori img_feature and text feature
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

