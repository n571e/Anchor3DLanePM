_base_ = ['./bundlelane_ablation_frame_anchor_warm_r18_10k_bs8.py']

model = dict(
    bundle_cfg=dict(
        inject_strength=0.1,
    ),
)

work_dir = 'output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_soft01_bs8'
