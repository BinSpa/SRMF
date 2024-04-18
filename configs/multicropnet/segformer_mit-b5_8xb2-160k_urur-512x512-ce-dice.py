_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/urur_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.8, direction=['horizontal', 'vertical']),
    dict(type='MultiLevelCrop', crop_size=crop_size, cat_max_ratio=0.75, level_list=[1,2,3,4]),
    dict(type='PackSegMultiInputs')
]

data_preprocessor = dict(
    type='MultiSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=3,
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(type='MultiCrop_SegformerHead', in_channels=[64, 128, 320, 512], num_classes=8,
                     loss_decode=[
                         dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
                         dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
                     ]),
    test_cfg = dict(mode='slide',crop_size=(512, 512),  stride=(341, 341)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.8, direction=['horizontal', 'vertical']),
    dict(type='MultiLevelCrop', crop_size=crop_size, cat_max_ratio=0.75, level_list=[1,2,4,6]),
    dict(type='PackSegMultiInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', scale=(1024, 1024)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='PackSegInputs')
]

train_dataloader = dict(batch_size=4, num_workers=16, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=2, num_workers=16, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1000, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
)

'''
check_layers = ['decode']
custom_hooks = [
    dict(type='CheckGradHook', check_layers=check_layers)
]
'''
