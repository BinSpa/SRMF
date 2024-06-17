_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/fbp_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa
model = dict(
    type='ShapeEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(type='Shape_SegformerHead',in_channels=[64, 128, 320, 512], num_classes=25),
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
    dict(type='RandomFlip', prob=0.8, direction=['horizontal', 'vertical']),
    dict(type='Shape_Detect', multi_size=5, cat_max_ratio=0.8, ignore_index=255),
    dict(type='PackSegShapeInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(batch_size=16, num_workers=8, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=2, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=50)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=50),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
)