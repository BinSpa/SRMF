import numpy as np
import torch
import torchvision.transforms as transforms

def multi_level_crop(image, crop_bbox, crop_size, level_list):
    coordinate_list = gen_multi_level_coord(image, crop_bbox, crop_size, levels=level_list)
    multi_level_crop_image = []
    for coordinate in coordinate_list:
        cropped_img = crop_img(img=image, crop_bbox=coordinate)
        # cropped_img:(b,c,h,w)
        resized_image = transforms.Resize((crop_size, crop_size))(cropped_img)
        multi_level_crop_image.append(resized_image)
    stacked_images = torch.stack(multi_level_crop_image, dim=1)
    (b,m,c,h,w) = stacked_images.shape
    return stacked_images.reshape(b*m, c, h, w)

def gen_multi_level_coord(image, coordinate:tuple, crop_size:int, levels:list) -> dict:
        """
        get one random coordinate for every level.

        Args:
        - image: original image.
        - coordinate: original crop coordinate.
        - levels: multi level crop levels.

        Return:
        - A dict that contains the coordinate for every key.
        """
        coordinate_list = []
        for level in levels:
            if level == 1:
                coordinate_list.append(coordinate)
            else:
                pool = gen_one_level_pool(image, coordinate, crop_size, level)
                if len(pool) == 0:
                    coordinate_list.append(coordinate)
                else:
                    choice = np.random.randint(0, len(pool))
                    coordinate_list.append(pool[choice])
        return coordinate_list

def gen_one_level_pool(image, coordinate:tuple, crop_size:int, multi:int) -> list:
        """
        get one level crop coordinates pool for the original coordinate.
        
        Args:
        - image: original image.
        - coordinate: original crop coordinate.
        - crop_size: original crop size.
        - multi: scale ratio for original crop size.
        
        Return:
        - pool: After the image has been slice according to the crop_size*multi, 
        the coordinates of all the blocks that contain the original slicing.
        """
        pool = []
        h,w = image.shape[0], image.shape[1]
        multi_crop_size = multi*crop_size
        stride = multi_crop_size-crop_size
        n_crop_h, n_crop_w = (h-multi_crop_size) // stride+1, (w-multi_crop_size) // stride+1
        for i in range(n_crop_h + 1):
            for j in range(n_crop_w + 1):
                start_i = i*stride
                start_j = j*stride
                end_i = min(start_i + multi_crop_size, h)
                end_j = min(start_j + multi_crop_size, w)
                start_i = max(end_i - multi_crop_size, 0)
                start_j = max(end_j - multi_crop_size, 0)
                if Contained((coordinate[0], coordinate[2]), (start_i, start_j), crop_size, multi_crop_size):
                    # y1,y2,x1,x2
                    pool.append((start_i, start_i+multi_crop_size, start_j, start_j+multi_crop_size))
        return pool

def Contained(coordinate1:tuple, coordinate2:tuple, size1:int, size2:int) -> bool:
        """
        Determine if the area defined by coordinate1 and size1 is completely contained
        within the area defined by coordinate2 and size2.

        Parameters:
        - coordinate1: A tuple (top, left) representing the top-left corner of the first area.
        - coordinate2: A tuple (top, left) representing the top-left corner of the second area.
        - size1: A int (size1, size1) representing the size of the first area.
        - size2: A int (size2, size2) representing the size of the second area.

        Returns:
        - True if the first area is completely contained within the second area, otherwise False.
        """
    
        top1, left1 = coordinate1
        top2, left2 = coordinate2
        height1, width1 = size1, size1
        height2, width2 = size2, size2

        # Check if the top-left corner of area1 is within area2
        if left1 >= left2 and top1 >= top2:
            # Check if the bottom-right corner of area1 is within area2
            if (left1 + width1) <= (left2 + width2) and (top1 + height1) <= (top2 + height2):
                return True

        return False

def crop_img(img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
        return img