_base_ = ['./bundlelane_r18.py']

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
)

model = dict(
    bundle_cfg=dict(
        inject_iters=[0],
    ),
)

optimizer = dict(type='Adam', lr=5e-5)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=2500)

load_from = 'pretrained/openlane_anchor3dlane++_r18.pth'
resume_from = None
work_dir = 'output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_bs8'
