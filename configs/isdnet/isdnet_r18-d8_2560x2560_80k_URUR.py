_base_ = [
    '../_base_/models/isdnet_r50-d8.py', '../_base_/datasets/urur_2560x2560.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    down_ratio=4,
    backbone=dict(depth=18),
    decode_head=[
        dict(
            type='RefineASPPHead',
            in_channels=512,
            in_index=3,
            channels=128,
            dilations=(1, 12, 24, 36),
            dropout_ratio=0.1,
            num_classes=8,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='ISDHead',
            in_channels=3,
            prev_channels=128,
            down_ratio=4,
            channels=128,
            num_classes=8,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=8))

# 训练设置
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=50)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=50),
    visualization=dict(type='SegVisualizationHook'))