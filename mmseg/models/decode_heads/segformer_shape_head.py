# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F

from mmseg.models.decode_heads.shape_decode_head import ShapeDecodeHead
from mmseg.registry import MODELS
from ..utils import resize

from .replknet import RepLKNetStage

class Shape_Extract_conv(nn.Module):
    def __init__(self, layers=2, in_channel=1):
        super(Shape_Extract_conv, self).__init__()
        self.layers = layers

        # downsample 4 times
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=5, stride=4, padding=2)
        # downsample 4 times
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=4, padding=2)
        # downsample 2 times
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # dropout
        self.dropout = nn.Dropout(0.1)
        # pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # fc
        self.fc = nn.Linear(64, 32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.gap(x)
        b,_,_,_ = x.shape
        x = x.view(b,-1)
        x = self.fc(x)
        return x
        
class Shape_Extract_lk(nn.Module):
    def __init__(self, layers=2, in_channel=1, 
                 in_channels=[32,64], out_channels=64, num_blocks=[2,2], kernel_sizes=[31,29],
                 drop_path_rate=0.3, small_kernel=5, dw_ratio=1, ffn_ratio=4, 
                 small_kernel_merged=False, norm_intermediate_features=False,):
        super(Shape_Extract_lk, self).__init__()
        self.layers = layers
        self.in_channel = in_channel
        self.in_channels = in_channels
        # stem downsample 4 times
        stem_channel = in_channels[0]
        self.stem_conv = nn.Conv2d(in_channel, stem_channel, kernel_size=5, stride=4, padding=2)
        self.stem_bn = nn.BatchNorm2d(stem_channel)
        # large kernel stage
        self.lk_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        for stage_idx in range(self.layers):
            lk_layer = RepLKNetStage(
                channels=in_channels[stage_idx], num_blocks=num_blocks[stage_idx],
                stage_lk_size=kernel_sizes[stage_idx],
                drop_path=dpr[sum(num_blocks[:stage_idx]):sum(num_blocks[:stage_idx + 1])],
                small_kernel=small_kernel, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
                use_checkpoint=False, small_kernel_merged=small_kernel_merged,
                norm_intermediate_features=norm_intermediate_features)
            self.lk_blocks.append(lk_layer)
        for stage_idx in range(self.layers):
            trans_stem = nn.ModuleList()
            trans_stem.append(nn.Conv2d(in_channel, stem_channel, kernel_size=5, stride=4, padding=2))
            trans_stem.append(nn.BatchNorm2d(stem_channel))
            self.downsamples.append(trans_stem)
        # pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_channels[-1], out_channels)
    
    def forward(self, x):
        # stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        # large kernel
        for stage_index in range(self.layers):
            x = self.lk_blocks[stage_index](x)
            x = self.downsamples[stage_index](x)
        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x


@MODELS.register_module()
class Shape_SegformerHead(ShapeDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', shape_dim=32, shape_nums=4, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.shape_nums = shape_nums
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        # image feature
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        # shape feature
        self.shape_extract = Shape_Extract_conv()
        # feature fusion
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs + shape_dim,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs, shape_maps, shape_index):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        # image feature
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
        # shape feature
        # shape_maps:(b*3,h,w); shape_index:(b,h,w)
        shape_feature = self.shape_extract(shape_maps)
        b, d = shape_feature.shape
        _, h, w = shape_index.shape
        shape_feature = shape_feature.view(b//self.shape_nums, self.shape_nums, d)
        # 下面准备使用gather操作来选择特征
        shape_feature_expand = shape_feature.unsqueeze(2).unsqueeze(2).expand(b//self.shape_nums, self.shape_nums, h, w, d)
        shape_index_expand = shape_index.unsqueeze(1).unsqueeze(4).expand(b//self.shape_nums,1,h,w,d)
        selected_features = torch.gather(shape_feature_expand, 1, shape_index_expand)
        selected_features = selected_features.squeeze(1).permute(0,3,1,2)
        selected_features = resize(
            input=selected_features,
            size=outs[0].shape[2:],
            mode=self.interpolate_mode,
            align_corners=self.align_corners
        )
        # fusion
        outs.append(selected_features)
        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out
