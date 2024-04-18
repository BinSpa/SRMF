_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/gid_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)

data_preprocessor = dict(
    type='MultiSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='MCEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=3,
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        type='MCFusionSegformerHead', 
        in_channels=[64, 128, 320, 512], 
        num_classes=6,
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg = dict(mode='slide',crop_size=(512, 512),stride=(341, 341),levels=[1,2,3,4]))

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
    dict(type='MultiLevelCrop', crop_size=crop_size, cat_max_ratio=0.75, level_list=[1,2,3,4], withlocal=True),
    dict(type='PackSegMultiInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='MultiLevelCrop', crop_size=crop_size, cat_max_ratio=0.75, level_list=[1,2,3,4], withlocal=True),
    dict(type='PackSegInputs')
]

train_dataloader = dict(batch_size=2, num_workers=2, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=4, num_workers=4, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

resume = True
load_from = "/home/rsr/gyl/RS_Code/mmseg_exp/onlyfusion_enforce_gid/first_iter_160000.pth"

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=320000, val_interval=16000)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
)

'''
check_layers = ['decode']
custom_hooks = [
    dict(type='CheckGradHook', check_layers=check_layers)
]
'''