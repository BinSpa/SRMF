_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/urur_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=8),
    auxiliary_head=dict(num_classes=8),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
train_dataloader = dict(batch_size=16, num_workers=10)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),
    logger=dict(type='LoggerHook', interval=1000, log_metric_by_epoch=False),
    # test visualizer
    # visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)
