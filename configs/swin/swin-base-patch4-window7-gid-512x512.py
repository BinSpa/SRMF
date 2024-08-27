_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/gid_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_20220317-e9b98025.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=25),
    auxiliary_head=dict(in_channels=512, num_classes=6),
    test_cfg = dict(mode='slide',crop_size=(512, 512),  stride=(341, 341)))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
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

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=8,num_workers=10)
val_dataloader = dict(batch_size=1,num_workers=2)
test_dataloader = val_dataloader

# train cfg and default hooks
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
    logger=dict(type='LoggerHook', interval=1000, log_metric_by_epoch=False),
    # test visualizer
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1),
)

# test visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer', alpha=1.0)