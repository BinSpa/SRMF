from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class URURDataset(BaseSegDataset):

    METAINFO = dict(
        classes=("background", "building", "farmland", "greenhouse", "woodland", "bareland", "water", "road"),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255, 0], [0, 0, 255], [255, 0, 255], [123, 123, 123]])

    def __init__(self, img_suffix='.png', seg_map_suffix='.png', **kwargs) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)