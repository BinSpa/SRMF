from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class URURDataset(BaseSegDataset):

    METAINFO = dict(
        classes=("background", "building", "farmland", "greenhouse", "woodland", "bareland", "water", "road"),
        palette=[[0, 0, 0], [230, 230, 230], [95, 163, 7], [100, 100, 100], [200, 230, 160], [255, 255, 100], [150, 200, 250], [240, 100, 80]])

    def __init__(self, img_suffix='.png', seg_map_suffix='.png', **kwargs) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)