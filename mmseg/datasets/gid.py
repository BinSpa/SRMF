from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class GIDDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('unlabeled', 'built-up', 'farmland', 'forest', 'meadow', 'water'),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255, 0], [0, 0, 255]])

    def __init__(self, img_suffix='.tif', seg_map_suffix='_5label.png', **kwargs) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)