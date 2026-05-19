_base_ = ['./anchor3dlane++_r18.py']

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
)

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=5000)
work_dir = 'output/ablation/openlane/anchor3dlanepp_baseline_r18_20k_bs8'
