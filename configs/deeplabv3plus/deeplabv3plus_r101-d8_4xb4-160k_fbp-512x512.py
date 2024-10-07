_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/fbp_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=25),
    auxiliary_head=dict(num_classes=25),
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    test_cfg = dict(mode='slide',crop_size=(512, 512),  stride=(341, 341)))

train_dataloader = dict(batch_size=8, num_workers=10)
test_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = test_dataloader


train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
    logger=dict(type='LoggerHook', interval=1000, log_metric_by_epoch=False),
    # test visualizer
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)
# test visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
   type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer', alpha=1.0)