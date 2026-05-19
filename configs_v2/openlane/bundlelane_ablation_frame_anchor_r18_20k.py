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

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=5000)
work_dir = 'output/ablation/openlane/bundlelane_frame_anchor_r18_20k_bs8'
