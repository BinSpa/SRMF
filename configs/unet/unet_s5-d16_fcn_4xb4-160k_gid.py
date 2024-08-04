_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/gid_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide'), crop_size=(512, 1024),  stride=(341, 341))
train_dataloader = dict(batch_size=16, num_workers=10)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=50)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=50),
    logger=dict(type='LoggerHook', interval=1000, log_metric_by_epoch=False),
    # test visualizer
    # visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)
