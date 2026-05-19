_base_ = ['./bundlelane_ablation_frame_anchor_warm_r18_10k_bs8.py']

model = dict(
    bundle_cfg=dict(
        frame_x_loss_weight=0.05,
    ),
)

work_dir = 'output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_xloss005_bs8'
