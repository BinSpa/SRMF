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

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
)