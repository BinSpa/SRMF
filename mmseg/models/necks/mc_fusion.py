from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.nn import MultiheadAttention
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ..utils import resize

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Channel_Change_Block(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Channel_Change_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

# 交叉注意力
class NCrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(NCrossAttentionModule, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
            'attention': MultiheadAttention(embed_dim, num_heads),
            'norm': nn.LayerNorm(embed_dim),
            'ffn': FFN(embed_dim, embed_dim)
            }) for _ in range(num_layers)])

    def forward(self, query, key, value):
        for layer in self.layers:
            attention_layer = layer['attention']
            norm_layer = layer['norm']
            ffn_layer = layer['ffn']
            next_query, _ = attention_layer(query, key, value)
            query = norm_layer(query + next_query)
            query = ffn_layer(query)
        return query
# 全局通道注意力
class GlobalChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(GlobalChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        """
        x: (b, c, h, w) feature map
        y: (b, 3, c, h, w) feature map
        """
        # Combine the features from different inputs
        y = y.mean(dim=1)  # Reduce along the '3' dimension to get (b, c, h, w)
        combined = x + y  # Element-wise addition
        
        # Compute the channel attention
        b, c, _, _ = combined.size()
        out = self.avg_pool(combined).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        return x * out.expand_as(x)

@MODELS.register_module()
class MCFusion(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level=2,
                 num_heads=4,
                 num_layers=2,
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.start_level = start_level
        self.mhca = nn.ModuleList()
        self.convattn = nn.ModuleList()
        for i in range(self.start_level, self.num_ins):
            in_channel = self.in_channels[i]
            out_channel = self.out_channels[i]
            self.mhca.append(NCrossAttentionModule(embed_dim=in_channel, num_heads=num_heads, num_layers=num_layers))
            self.convattn.append(Channel_Change_Block(in_channels=in_channel*2, hidden_channels=out_channel, out_channels=out_channel))
            # self.glattn.append(GlobalChannelAttention(in_channel))
        
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        fusion_feature = []
        for i in range(0, self.start_level):
            b,c,h,w = inputs[i].shape
            multi_feature = inputs[i].reshape(-1, 4, c, h, w)
            fusion_feature.append(multi_feature[:,0,:,:,:])
        for i in range(self.start_level, self.num_ins):
            b,c,h,w = inputs[i].shape
            multi_feature = inputs[i].reshape(-1, 4, c, h, w)
            sub_b,_,c,h,w = multi_feature.shape
            local_feature = multi_feature[:,0,:,:,:]
            crop_feature = multi_feature[:,1:,:,:,:]
            # glca_feature = self.glattn[i-self.start_level](local_feature, crop_feature)
            local_feature_ca = local_feature.reshape(sub_b,c,-1).permute(2,0,1)
            crop_feature_ca = crop_feature.reshape(sub_b,c,-1).permute(2,0,1)
            mhca_feature = self.mhca[i-self.start_level](local_feature_ca, crop_feature_ca, crop_feature_ca)
            mhca_feature = mhca_feature.permute(1,2,0).reshape(sub_b,c,h,w)
            mhca_feature = torch.cat((mhca_feature, local_feature), dim=1)
            mhca_feature = self.convattn[i-self.start_level](mhca_feature)
            fusion_feature.append(mhca_feature)
        return tuple(fusion_feature)



