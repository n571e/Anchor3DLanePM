_base_ = ['./anchor3dlane++_r18.py']

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
)

optimizer = dict(type='Adam', lr=5e-5)
runner = dict(type='IterBasedRunner', max_iters=30000)
checkpoint_config = dict(by_epoch=False, interval=5000)

load_from = None
resume_from = 'output/ablation/openlane/anchor3dlanepp_baseline_warm_r18_10k_bs8/iter_10000.pth'
work_dir = 'output/ablation/openlane/anchor3dlanepp_baseline_warm_r18_30k_resume_bs8'
