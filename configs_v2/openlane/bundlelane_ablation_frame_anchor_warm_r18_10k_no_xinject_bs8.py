_base_ = ['./bundlelane_ablation_frame_anchor_warm_r18_10k_bs8.py']

model = dict(
    bundle_cfg=dict(
        inject_x=False,
        inject_z=True,
    ),
)

work_dir = 'output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_no_xinject_bs8'
