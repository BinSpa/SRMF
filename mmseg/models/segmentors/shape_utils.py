import os
import torch
import cv2
from torch import Tensor
import numpy as np

def get_shapes(input: Tensor, coordinate: tuple, ratio=5):
    '''
    inputs:
        - input: origin image,(c,h,w)
        - coordinate: current crop coordinates,(y1,x1,y2,x2)
        - ratio: superpixels image times according current crop
    returns:
        - shape_map_data: (b,3,h,w) tensor
        - shape_index_data: (b,h,w) tensor
    '''
    np_input = input.detach().cpu().numpy()
    np_input = np_input.reshape(2,0,1)
    h,w,c = np_input.shape
    y, x, h, w = coordinate[0], coordinate[1], coordinate[2]-coordinate[0], coordinate[3]-coordinate[1]
    # get multicrop image
    my, mx, ry, rx, offset_y, offset_x = generate_multi_crop(np_input.shape, y, x, h, w, ratio=5)
    multicrop_input = np_input[my:ry, mx:rx, ...]
    # get super pixel labels
    lab_image = cv2.cvtColor(multicrop_input, cv2.COLOR_RGB2LAB)
    superpixel_labels, number_sp = slic_superpixels(lab_image)
    # get roi region
    roi_label = superpixel_labels[offset_y:offset_y+512, offset_x:offset_x+512]
    # detect connects
    connects = []
    for i in range(number_sp):
        segment = (roi_label == i).astype(np.uint8) * 255
        # 检测连通域
        num_labels, seg_labels, stats, centroids = cv2.connectedComponentsWithStats(segment, 8, cv2.CV_32S)
        # 遍历检测出来的连通域
        for j in range(1, num_labels):
            c = tuple(np.int32(centroids[j]))
            # 判断质心是否在连通域中，如果不是，就换一个点
            c = find_point(seg_labels, j, c)
            areas = stats[j, cv2.CC_STAT_AREA]
            if areas > 10000:
                connects.append((i,j,c,areas))
    # 按照连通域面积排序
    connects = sorted(connects, key=lambda x:x[3], reverse=True)
    # 获取形状
    shapes = find_connects(connects, superpixel_labels)
    zero_feature = np.zeros((1, shapes[0].shape[0], shapes[0].shape[1]))
    shapes_feature = np.concatenate((zero_feature, shapes), axis=0)
    # 获取映射坐标
    mapindex = find_mapindex(connects, roi_label)
    shapes_feature = shapes_feature.astype(np.float32)
    mapindex = mapindex.astype(np.int64)
    return shapes_feature, mapindex

def find_point(seg_labels, index, coordinate):
    if seg_labels[coordinate[0], coordinate[1]] == index:
        return (coordinate[0], coordinate[1])
    else:
        component_mask = (seg_labels == index)
        Y, X = np.where(component_mask)
        return (X[0], Y[0])
    
def find_connects(connects, superpixel_label):
    # 对几个最大的连通域，找到在大图中的位置
    shapes = []
    connect_nums = min(3, len(connects))
    for i in range(connect_nums):
        one_connect = connects[i]
        cx, cy = one_connect[2][0], one_connect[2][1]
        segment_label = (superpixel_label == one_connect[0]).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segment_label, connectivity=8)
        label_of_point = labels[cy, cx]
        component_mask = (labels == label_of_point).astype(np.uint8) * 255
        shapes.append(component_mask)
    while len(shapes) < 3:
        black_img = np.zeros_like(superpixel_label)
        shapes.append(black_img)
    return np.array(shapes)

def find_mapindex(connects, roi_label):
    # 小图像的每个像素应当对应哪个shape
    connect_nums = min(3, len(connects))
    mapindex_list = []
    for i in range(connect_nums):
        segment_index = connects[i][0]
        connect_index = connects[i][1]
        black_img = np.zeros_like(roi_label)
        segment_map = (roi_label == segment_index).astype(np.uint32) * 255
        num_labels, seg_labels, stats, centroids = cv2.connectedComponentsWithStats(segment_map, 8, cv2.CV_32S)
        black_img[seg_labels == connect_index] = i+1
        mapindex_list.append(black_img)
    merge_img = np.zeros_like(roi_label)
    for mapindex in mapindex_list:
        merge_img = cv2.bitwise_or(merge_img, mapindex)
    
    return merge_img

def generate_multi_crop(img_shape, y, x, h, w, ratio=5):
        '''
        input : x,y,h,w
        return : x,y,h,w,offset_y,offset_x
        '''
        # mode='keep_ratio', we always keep the shape of the image.
        mh, mw = h*ratio, w*ratio
        centery, centerx = y + h/2, x + w/2 
        my, mx = centery - mh // 2, centerx - mw // 2
        ry, rx = my + mh, mx + mw
        if my < 0 :
            my = 0
            ry = my + mh
        elif mx < 0:
            mx = 0
            rx = mx + mw
        if ry > img_shape[0]:
            ry = img_shape[0]
            my = ry - mh
        elif rx > img_shape[1]:
            rx = img_shape[1]
            mx = rx - mw
        offset_y = y - my
        offset_x = x - mx
        return int(my), int(mx), int(ry), int(rx), int(offset_y), int(offset_x) 

def slic_superpixels(lab_image, num_segments=25, compactness=10, step=20):
    # 创建SLIC对象
    slic = cv2.ximgproc.createSuperpixelSLIC(lab_image, algorithm=cv2.ximgproc.SLIC, region_size=int(np.sqrt(image.size / num_segments)), ruler=compactness)

    # 执行SLIC算法
    slic.iterate(step)  # 你可以调整迭代次数
    # 强制连通性
    slic.enforceLabelConnectivity(min_element_size=25)
    # 获取最终的超像素数量
    number_sp = slic.getNumberOfSuperpixels()
    # 获取超像素标签
    labels = slic.getLabels()

    return labels, number_sp


    