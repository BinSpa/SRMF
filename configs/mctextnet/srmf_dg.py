_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/deepglobe_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)

clip_text = "/data1/gyl/RS_Code/mmseg_exp/Code/cliph14_txt54.pt"

# clip_text = "/data1/gyl/RS_Code/mmseg_exp/Code/new_cliph14_txt54.pt"
data_preprocessor = dict(
    type='MultiSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    clip_text=clip_text,
    size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
model = dict(
    type='TextEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=3,
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(type='MCText_SegformerHead', in_channels=[64, 128, 320, 512], num_classes=7, text_nums=54),
    test_cfg = dict(mode='slide',crop_size=(512, 512),  stride=(341, 341)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=5e-6, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=3000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.8,
        begin=3000,
        end=160000,
        by_epoch=False,
    )
]

# record_path = '/data1/gyl/RS_DATASET/boxes_jsonl/urur_record.jsonl'
boxes_path = '/data1/gyl/RS_DATASET/boxes_jsonl/deepglobe_boxes.jsonl'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='ColorJittering'),
    dict(type='Samhq_boxes', boxes_path=boxes_path, select_num=4, keep_gsd=True, ifmc=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.8, direction=['horizontal', 'vertical']),
    dict(type='MultiLevelCrop', crop_size=crop_size, cat_max_ratio=0.75, level_list=[1,2,3,4]),
    dict(type='PackSegMultiInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', scale=(1024, 1024)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='PackSegInputs')
]

train_dataloader = dict(batch_size=1, num_workers=20, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=4, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
    logger=dict(type='LoggerHook', interval=1000, log_metric_by_epoch=False),
    # visualization=dict(type='SegVisualizationHook', draw=True, interval=10)
)
# test
# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer', alpha=1.0)