import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import random
import numpy as np
from mmengine.model import BaseModule
from torch import Tensor
import torch.nn.functional as F
from collections import defaultdict
from mmseg.utils import ConfigType, SampleList
from ..builder import build_loss
from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class PureBlockHead(BaseModule, metaclass=ABCMeta):
    """ PureBlockHead for pure-block contrassive learning.
    1. The ``init_weights`` method is used to initialize decode_head's
    model parameters. After segmentor initialization, ``init_weights``
    is triggered when ``segmentor.init_weights()`` is called externally.
    2. The ``loss`` method is used to calculate the loss of pure-block contrassive learning,
    Different from decode_head, we directly pass in the seg_logist and data_sample to calculate 
    the loss, eliminating the need for operations that are only available in deocdehead,
    such as loss_by_feature and predict, and directly connecting to the Encoderdecoder architecture.
    """

    def __init__(
            self, 
            init_cfg=dict(
                type='Normal', 
                layer='Linear', 
                mean=0, 
                std=0.01, 
                bias=0),
            agg_meth='project',
            neg_nums=32,
            block_size=16, 
            temperature=0.07,  
            ignore_index=255,
            need_logist=True,
            align_corners=False,
            interpolate_mode='bilinear',
            ori_size=512,
            input_dim=1024,
            hidden_dim1=512,
            hidden_dim2=512,
            output_dim=256,
            loss_type='amsoftmax',
            margin=0.35,
            scale=30.0,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
    ):
        super().__init__(init_cfg)
        self.blcok_size = block_size
        self.neg_nums = neg_nums
        self.agg_meth = agg_meth
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.need_logist = need_logist
        self.align_corners=align_corners
        self.interpolate_mode = interpolate_mode
        self.ori_size = ori_size
        self.loss_type = loss_type
        self.margin = margin
        self.scale = scale
        if self.agg_meth == 'mean':
            self.agg_layer = nn.ModuleList()
            self.agg_layer.append(nn.AvgPool2d(kernel_size=self.blcok_size, stride=1))
        elif self.agg_meth == 'max':
            self.agg_layer = nn.ModuleList()
            self.agg_layer.append(nn.AvgPool2d(kernel_size=self.blcok_size, stride=1))
        elif self.agg_meth == 'project':
            self.agg_layer = nn.ModuleList()
            self.agg_layer.append(nn.AdaptiveAvgPool2d((1,1)))
            self.agg_layer.append(nn.Linear(input_dim, hidden_dim1))
            self.agg_layer.append(nn.ReLU())
            self.agg_layer.append(nn.Linear(hidden_dim1, hidden_dim2))
            self.agg_layer.append(nn.ReLU())
            self.agg_layer.append(nn.Linear(hidden_dim2, output_dim))
        else:
            raise NotImplementedError("the {} method have not implemented yet.".format(self.agg_meth))
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')
    
    def forward(self, inputs):
        if len(self.agg_layer) == 1:
            x = self.agg_layer(inputs)
            x = x.squeeze(-1).squeeze(-1)
        else:
            x = inputs
            for i, layer in enumerate(self.agg_layer):
                x = layer(x)
                if i == 0:
                    x = x.squeeze(-1).squeeze(-1)
        return x        

    def _stack_boxes(self, batch_data_samples):
        boxes_list = []
        categroy_ids = set()
        # We start by merging box dictionaries of different images in the same list
        for i, data_sample in enumerate(batch_data_samples):
            box_dict = data_sample.pure_blocks
            categroy_ids.update(box_dict.keys())
            for key, value in box_dict.items():
                for box in value:
                    boxes_list.append([i, key, box[0], box[1]])
        return boxes_list, categroy_ids
        

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
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
        out = resize(
            input=out,
            size=(self.ori_size, self.ori_size),
            mode=self.interpolate_mode,
            align_corners=self.align_corners
        )
        boxes_list, categroy_ids = self._stack_boxes(batch_data_samples)
        assert len(categroy_ids) >= 2, "the categroy_ids is:{}".format(categroy_ids) 
        # for every id in categroy_ids, build a dataset for contrassive learning.
        contra_dataset = []
        for cate_id in categroy_ids:
            patches = []
            # random select anchor and positive
            anchor_pos = self.random_select(boxes_list, 2, select_cate=cate_id)
            # random select negative
            negs = self.random_select(boxes_list, self.neg_nums, ignore_cate=cate_id)
            for box in anchor_pos:
                top, left = boxes_list[box][2], boxes_list[box][3]
                block = out[boxes_list[box][0], :, top:top+self.blcok_size, left:left+self.blcok_size]
                patches.append(block)
            for box in negs:
                top, left = boxes_list[box][2], boxes_list[box][3]
                block = out[boxes_list[box][0], :, top:top+self.blcok_size, left:left+self.blcok_size]
                patches.append(block)
            # [neg_num+2,c,b_s,b_s]
            patches = torch.stack(patches, dim=0)
            contra_dataset.append(patches)
        # [b*(neg_num+2),c,b_s,b_s]
        contra_dataset = torch.cat(contra_dataset, dim=0)
        # [b*(neg_num+2),c]
        contra_dataset = self.forward(contra_dataset)
        _, c = contra_dataset.shape
        contra_dataset = contra_dataset.reshape(-1, self.neg_nums+2, c)
        # assert False, "the shape of the contra_dataset:{}".format(contra_dataset.shape)
        # now we can calculate the cosine similarity
        anchor = contra_dataset[:, 0, :]
        if len(anchor.shape) == 2:
            anchor = anchor.unsqueeze(1)
        anchor = F.normalize(anchor, p=2, dim=2)
        samples = contra_dataset[:, 1:, :]
        samples = F.normalize(samples, p=2, dim=2)
        if self.loss_type == 'amsoftmax':
            similarities = torch.bmm(samples, anchor.transpose(1, 2))
            similarities = similarities.squeeze(-1)
            # (b, 16+1)
            similarities[:, 0] -= self.margin
            similarities *= self.scale            
        elif self.loss_type == 'infonce':
            similarities = torch.bmm(samples, anchor.transpose(1, 2)) / self.temperature
            # assert len(similarities.shape) == 5, "get the shape of similarities is {}".format(similarities.shape)
            similarities = similarities.squeeze(-1)
        labels = torch.zeros(similarities.shape[0], dtype=torch.long).to(anchor.device)
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        loss = dict()
        seg_weight = None
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    similarities,
                    labels,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    similarities,
                    labels,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        return loss

    def random_select(self, boxes_list, select_num, select_cate=None, ignore_cate=None):
        assert select_cate!=None or ignore_cate!=None, "select_cate and ignore_cate cannot be None at the same time, but got select_cate:{}, ignore_cate:{}".format(select_cate, ignore_cate)
        boxes_list = np.array(boxes_list)
        if select_cate is not None:
            idx4selectcate = np.where(boxes_list[:, 1] == select_cate)[0]
        else:
            idx4selectcate = np.where(boxes_list[:, 1] != ignore_cate)[0]
        if len(idx4selectcate) == 0:
            raise ValueError("No boxes found for the specified category.")
        # 扩展索引列表满足所需数量
        repeat_times = (select_num + len(idx4selectcate) - 1) // len(idx4selectcate)
        extended_indices = np.tile(idx4selectcate, repeat_times)[:select_num]
        # 随机选择
        selected_indices = np.random.choice(extended_indices, select_num, replace=False)
        return selected_indices.tolist()
        '''
        if select_cate != None:
            idx4selectcate = [i for i,value in enumerate(boxes_list) if value[1]==select_cate]
        else:
            idx4selectcate = [i for i,value in enumerate(boxes_list) if value[1]!=ignore_cate]
        # if the len not enough, we should keep the negative samples base number.
        if select_num > len(idx4selectcate):
            idx4selectcate = [idx4selectcate[i % len(idx4selectcate)] for i in range(select_num)]
        selected_indices = random.sample(idx4selectcate, select_num)
        return selected_indices
        '''

