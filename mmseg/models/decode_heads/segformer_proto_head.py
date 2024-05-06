# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseMultiCropDecodeHead

from mmseg.registry import MODELS
from ..utils import resize


class ConvMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ConvMLP, self).__init__()
        # 第一层1x1卷积，相当于MLP的隐藏层
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        # 第二层1x1卷积，相当于MLP的输出层
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 激活函数可以根据需要替换
        x = self.conv2(x)
        return x


@MODELS.register_module()
class Proto_SegformerHead(BaseMultiCropDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, k=10, momentum=0.999, text_path=None, interpolate_mode='bilinear', **kwargs):
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

        text_feature = torch.load(text_path, map_location='cpu')
        self.prototypes = nn.Parameter(text_feature,
                                       requires_grad=False)
        self.k = k
        self.momentum = momentum
        self.img_project = ConvMLP(in_channels=self.channels * num_inputs, hidden_channels=1240, out_channels=1024)

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

        outs_feature = torch.cat(outs, dim=1)
        project_outs = self.img_project(outs_feature)
        # normalize
        norm_outs = project_outs / project_outs.norm(dim=1, keepdim=True)
        norm_protos = self.prototypes / self.prototypes.norm(dim=1, keepdim=True)
        # cosine similarity
        similarity_map = torch.einsum('nd,bdhw->bnhw', norm_protos, norm_outs)
        # select pixels to update prototypes
        n, d = norm_protos.shape
        b, _, h, w  = norm_outs.shape
        
        new_prototypes = self.prototypes.clone()
        for i in range(n):
            similarity_map_i = similarity_map[:, i, :, :]
            # find topk similarity
            topk_values, topk_indices = torch.topk(similarity_map_i.view(b, -1), k=self.k, dim=1, largest=True, sorted=False)
            topk_feature_vectors = project_outs.view(b, d, h*w)[:, :, topk_indices]
            # mean:(b,d)
            topk_feature_vectors = topk_feature_vectors.mean(dim=2)
            # calculate all batch means:(d,)
            mean_feature_vector = topk_feature_vectors.mean(dim=0)
            # momentum
            new_prototypes[i] = self.momentum * self.prototypes[i] + (1 - self.momentum) * mean_feature_vector
        self.prototypes.data.copy_(new_prototypes)
        # fusion feautres
        outs.append(similarity_map)
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
