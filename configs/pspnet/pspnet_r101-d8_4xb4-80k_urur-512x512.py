_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/urur_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    pretrained='open-mmlab://resnet101_v1c', 
    backbone=dict(depth=101),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=8),
    auxiliary_head=dict(num_classes=8),
    test_cfg = dict(mode='slide',crop_size=(512, 512),  stride=(341, 341)))

train_dataloader = dict(batch_size=8, num_workers=12)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    # test
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer', alpha=1.0)
# test
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=10)
)