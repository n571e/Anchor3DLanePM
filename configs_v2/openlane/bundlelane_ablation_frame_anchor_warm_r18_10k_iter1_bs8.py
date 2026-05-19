_base_ = ['./bundlelane_ablation_frame_anchor_warm_r18_10k_bs8.py']

model = dict(
    bundle_cfg=dict(
        inject_iters=[1],
    ),
)

work_dir = 'output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_iter1_bs8'
